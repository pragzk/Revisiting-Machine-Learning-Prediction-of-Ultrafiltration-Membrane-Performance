"""
SCRIPT 1 of 4 -- Fixed-protocol reconciliation for flux recovery ratio.

Runs every tunable model under ONE identical protocol (same seed, same
n_trials, same inner CV) so there's exactly one number per model and no
ambiguity about which historical run "counts". Whichever model wins THIS
run replaces both the old 0.638 and the disputed 0.619 -- don't keep
either prior number afterward.

Run from the same directory as full_pipeline_part1_data.py and
full_pipeline_part2_tuning.py.

Usage:
    python3 01_reconcile_flux_recovery.py
    python3 01_reconcile_flux_recovery.py --n-trials 25 --seed 1000
"""
import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import full_pipeline_part1_data as p1
import full_pipeline_part2_tuning as p2

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def load_sheet1_features():
    df1 = p1.canonicalize_categoricals(p1.load_sheet1())
    df1 = p1.add_chemical_family_features(df1)
    df1 = p1.add_interaction_features(df1)
    exclude = p1.PERFORMANCE_TARGETS + [p1.IRREVERSIBLE_FOULING_COL, "DOI", "SMILES"]
    feature_cols = [c for c in df1.columns if c not in exclude]
    numeric_cols = df1[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = df1[feature_cols].select_dtypes(exclude="number").columns.tolist()
    return df1, numeric_cols, categorical_cols


def main(n_trials=25, seed=1000):
    df1, num1, cat1 = load_sheet1_features()
    target = "flux recovery ratio (%)"
    mask = df1[target].notna()
    X = df1.loc[mask, num1 + cat1]
    y = df1.loc[mask, target]

    results = {}
    for model_name in p2.TUNABLE_MODELS:
        r = p2.tune_with_train_test_metrics(
            model_name, X, y, num1, cat1, seed=seed, n_trials=n_trials, log_target=False
        )
        results[model_name] = r
        print(f"{model_name:22s} test_r2={r['test_r2']:.4f}  train_r2={r['train_r2']:.4f}  "
              f"test_rmse={r['test_rmse']:.3f}")

    best_model = max(results, key=lambda m: results[m]["test_r2"])
    print(f"\nRECONCILED BEST for flux recovery ratio: {best_model} "
          f"(test_r2={results[best_model]['test_r2']:.4f})")
    print("Replace BOTH the old 0.638 (HistGradientBoosting) and the disputed "
          "0.619 (LightGBM) with this run's number -- don't keep either prior value.")

    with open(OUT_DIR / "flux_recovery_reconciled.json", "w") as f:
        json.dump({"seed": seed, "n_trials": n_trials, "all_models": results,
                    "best_model": best_model}, f, indent=2, default=str)
    print("Saved outputs/flux_recovery_reconciled.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    main(n_trials=args.n_trials, seed=args.seed)
