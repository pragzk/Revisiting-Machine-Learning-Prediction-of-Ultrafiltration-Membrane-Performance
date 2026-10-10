"""
SCRIPT 2 of 4 -- Repeated-seed testing of the ACTUAL reported best solo
model per target (not just the untuned baseline screen, which already has
this check). Produces a real mean +/- std per target to replace the
single-seed=1000 comparison against Gao et al.

Covers the SOLO model currently reported as best per target. Ensembles are
a further layer not wrapped into this harness -- that's a separate,
later extension if you want it.

Run from the same directory as full_pipeline_part1_data.py and
full_pipeline_part2_tuning.py. Run 01_reconcile_flux_recovery.py first,
or pass --flux-recovery-model to skip that dependency.

Usage:
    python3 02_repeated_seed_tuned.py --flux-recovery-model HistGradientBoosting
    python3 02_repeated_seed_tuned.py --flux-recovery-model HistGradientBoosting --n-seeds-perf 30 --n-seeds-prop 15
"""
import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import full_pipeline_part1_data as p1
import full_pipeline_part2_tuning as p2

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

PERFORMANCE_TARGETS = p1.PERFORMANCE_TARGETS
PROPERTY_TARGETS = p1.PROPERTY_TARGETS


def load_sheet1_features():
    df1 = p1.canonicalize_categoricals(p1.load_sheet1())
    df1 = p1.add_chemical_family_features(df1)
    df1 = p1.add_interaction_features(df1)
    exclude = PERFORMANCE_TARGETS + [p1.IRREVERSIBLE_FOULING_COL, "DOI", "SMILES"]
    feature_cols = [c for c in df1.columns if c not in exclude]
    numeric_cols = df1[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = df1[feature_cols].select_dtypes(exclude="number").columns.tolist()
    return df1, numeric_cols, categorical_cols


def load_sheet2_features():
    df2 = p1.canonicalize_categoricals(p1.load_sheet2())
    df2 = p1.add_chemical_family_features(df2)
    exclude = PROPERTY_TARGETS + ["DOI", "SMILES"]
    feature_cols = [c for c in df2.columns if c not in exclude]
    numeric_cols = df2[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = df2[feature_cols].select_dtypes(exclude="number").columns.tolist()
    return df2, numeric_cols, categorical_cols


def build_best_solo_model(flux_recovery_model):
    """(model_name, log_target) per target, matching Tables 1-2's solo winners."""
    return {
        "water permeability (LMH/bar)": ("LightGBM", True),
        "organic compound removal (%)": ("GradientBoosting", False),
        "flux decline ratio (%)": ("GradientBoosting", False),
        "flux recovery ratio (%)": (flux_recovery_model, False),
        "reversible fouling ratio (%)": ("GradientBoosting", False),
        "overall porosity (%)": ("GradientBoosting", False),
        "mean pore radius (nm)": ("KernelRidge_rbf", False),
        "contact angle (degree)": ("GradientBoosting", False),
    }


def main(n_trials=25, n_seeds_perf=30, n_seeds_prop=15, flux_recovery_model="HistGradientBoosting"):
    df1, num1, cat1 = load_sheet1_features()
    df2, num2, cat2 = load_sheet2_features()
    best_solo_model = build_best_solo_model(flux_recovery_model)

    seeds_perf = list(range(1000, 1000 + n_seeds_perf))
    seeds_prop = list(range(1000, 1000 + n_seeds_prop))
    results = {}

    for target in PERFORMANCE_TARGETS:
        model_name, log_target = best_solo_model[target]
        mask = df1[target].notna()
        X = df1.loc[mask, num1 + cat1]
        y = df1.loc[mask, target]
        t0 = time.time()
        if log_target:
            df = p2.tuned_repeated_splits_logtarget(
                model_name, X, y, num1, cat1, seeds=seeds_perf, n_trials=n_trials
            )
        else:
            df, _ = p2.tuned_repeated_splits(
                model_name, X, y, num1, cat1, seeds=seeds_perf, n_trials=n_trials
            )
        results[target] = {
            "model": model_name, "r2_mean": float(df.r2.mean()), "r2_std": float(df.r2.std()),
            "rmse_mean": float(df.rmse.mean()), "rmse_std": float(df.rmse.std()),
            "n_seeds": len(seeds_perf), "seconds": time.time() - t0,
        }
        print(f"[perf] {target[:25]:25s} {model_name:18s} "
              f"R2={results[target]['r2_mean']:.3f}+-{results[target]['r2_std']:.3f} "
              f"({results[target]['seconds']:.0f}s)")

    for target in PROPERTY_TARGETS:
        model_name, _ = best_solo_model[target]
        mask = df2[target].notna()
        X = df2.loc[mask, num2 + cat2]
        y = df2.loc[mask, target]
        t0 = time.time()
        df, _ = p2.tuned_repeated_splits(
            model_name, X, y, num2, cat2, seeds=seeds_prop, n_trials=n_trials
        )
        results[target] = {
            "model": model_name, "r2_mean": float(df.r2.mean()), "r2_std": float(df.r2.std()),
            "rmse_mean": float(df.rmse.mean()), "rmse_std": float(df.rmse.std()),
            "n_seeds": len(seeds_prop), "seconds": time.time() - t0,
        }
        print(f"[prop] {target[:25]:25s} {model_name:18s} "
              f"R2={results[target]['r2_mean']:.3f}+-{results[target]['r2_std']:.3f} "
              f"({results[target]['seconds']:.0f}s)")

    with open(OUT_DIR / "repeated_seed_tuned_models.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved outputs/repeated_seed_tuned_models.json")
    print("Use r2_mean +/- r2_std here (not the single seed=1000 value) as the")
    print("defensible per-target comparison against Gao et al. going forward.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--n-seeds-perf", type=int, default=30)
    parser.add_argument("--n-seeds-prop", type=int, default=15)
    parser.add_argument("--flux-recovery-model", type=str, required=True,
                         help="Winning model name from 01_reconcile_flux_recovery.py's output.")
    args = parser.parse_args()
    main(n_trials=args.n_trials, n_seeds_perf=args.n_seeds_perf,
         n_seeds_prop=args.n_seeds_prop, flux_recovery_model=args.flux_recovery_model)
