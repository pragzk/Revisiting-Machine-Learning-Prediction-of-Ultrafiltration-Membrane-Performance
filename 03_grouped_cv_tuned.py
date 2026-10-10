"""
SCRIPT 3 of 4 -- DOI-grouped, leakage-safe CV of the ACTUAL reported best
model per target (not just the untuned baseline screen, which already has
this check). Tunes fresh on each fold's training data (nested, leakage-safe)
so the tuning itself never sees the held-out source paper either.

This is the check that tells you whether the "beats Gao" claim survives
when an entire literature source is held out, rather than a random 20%
of rows that may share a source with the training set.

Run from the same directory as full_pipeline_part1_data.py and
full_pipeline_part2_tuning.py. Run 01_reconcile_flux_recovery.py first,
or pass --flux-recovery-model to skip that dependency.

Usage:
    python3 03_grouped_cv_tuned.py --flux-recovery-model HistGradientBoosting
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

import full_pipeline_part1_data as p1
import full_pipeline_part2_tuning as p2

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

PERFORMANCE_TARGETS = p1.PERFORMANCE_TARGETS


def load_sheet1_features():
    df1 = p1.canonicalize_categoricals(p1.load_sheet1())
    df1 = p1.add_chemical_family_features(df1)
    df1 = p1.add_interaction_features(df1)
    exclude = PERFORMANCE_TARGETS + [p1.IRREVERSIBLE_FOULING_COL, "DOI", "SMILES"]
    feature_cols = [c for c in df1.columns if c not in exclude]
    numeric_cols = df1[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = df1[feature_cols].select_dtypes(exclude="number").columns.tolist()
    return df1, numeric_cols, categorical_cols


def build_best_solo_model(flux_recovery_model):
    return {
        "water permeability (LMH/bar)": ("LightGBM", True),
        "organic compound removal (%)": ("GradientBoosting", False),
        "flux decline ratio (%)": ("GradientBoosting", False),
        "flux recovery ratio (%)": (flux_recovery_model, False),
        "reversible fouling ratio (%)": ("GradientBoosting", False),
    }


def tuned_group_kfold(model_name, X, y, groups, numeric_cols, categorical_cols,
                       n_splits=5, n_trials=25, seed=1000, log_target=False, fold_timeout=None):
    """Same idea as p2.evaluate_model_group_kfold, but tunes fresh (nested,
    leakage-safe) on each fold's training data instead of using library
    defaults."""
    gkf = GroupKFold(n_splits=n_splits)
    records = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        y_train_fit = np.log1p(y_train) if log_target else y_train

        best_params, inner_r2, pre, model = p2.tune_model_nested(
            model_name, X_train, y_train_fit, numeric_cols, categorical_cols,
            n_trials=n_trials, seed=seed, timeout=fold_timeout,
        )
        n_done = len(best_params)  # rough proxy; actual trial count in study
        to_str = f'{fold_timeout}s' if fold_timeout else 'unlimited'
        print(f"    fold {fold}: tuning done (timeout={to_str}, best_params={best_params})")
        Xva = pre.transform(X_val)
        pred = model.predict(Xva)
        if log_target:
            pred = np.expm1(pred)
        records.append({"fold": fold, "r2": r2_score(y_val, pred), "n_val": len(y_val)})
        print(f"    fold {fold}: r2={records[-1]['r2']:.3f} (n_val={len(y_val)})")
    return pd.DataFrame(records)


def main(n_trials=25, seed=1000, flux_recovery_model="HistGradientBoosting", fold_timeout=120):
    df1, num1, cat1 = load_sheet1_features()
    best_solo_model = build_best_solo_model(flux_recovery_model)

    doi_mask = df1["DOI"].notna()
    results = {}
    for target in PERFORMANCE_TARGETS:
        model_name, log_target = best_solo_model[target]
        mask = df1[target].notna() & doi_mask
        X = df1.loc[mask, num1 + cat1]
        y = df1.loc[mask, target]
        groups = df1.loc[mask, "DOI"]

        print(f"[grouped/tuned] {target} ({model_name})")
        to_str = f'{fold_timeout}s' if fold_timeout else 'unlimited'
        print(f"  timeout_per_fold={to_str}")
        df = tuned_group_kfold(model_name, X, y, groups, num1, cat1,
                                 n_trials=n_trials, seed=seed, log_target=log_target,
                                 fold_timeout=fold_timeout if fold_timeout else None)
        results[target] = {
            "model": model_name, "grouped_r2_mean": float(df.r2.mean()),
            "grouped_r2_std": float(df.r2.std()), "per_fold_r2": df.r2.tolist(),
        }
        print(f"  -> grouped R2 = {results[target]['grouped_r2_mean']:.3f} "
              f"+- {results[target]['grouped_r2_std']:.3f}\n")

    with open(OUT_DIR / "grouped_cv_tuned_models.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved outputs/grouped_cv_tuned_models.json")
    print("This answers 'do the TUNED models beat Gao under leakage-safe")
    print("evaluation', not just the untuned screen.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--flux-recovery-model", type=str, required=True,
                         help="Winning model name from 01_reconcile_flux_recovery.py's output.")
    parser.add_argument("--fold-timeout", type=int, default=120,
                         help="Max seconds Optuna spends tuning per fold (0=unlimited). "
                              "At ~85s/trial empirically, 120s => ~1 trial/fold; 600s => ~7 trials/fold.")
    args = parser.parse_args()
    main(n_trials=args.n_trials, seed=args.seed, flux_recovery_model=args.flux_recovery_model,
         fold_timeout=args.fold_timeout)
