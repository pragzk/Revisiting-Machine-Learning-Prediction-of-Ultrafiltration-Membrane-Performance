"""
Driver script — the piece missing from your handoff. Calls the actual
functions defined in full_pipeline_part2_tuning.py to reproduce:
  - 06_train_test_batch1.json      (tune_with_train_test_metrics, all targets)
  - 08_ensemble_train_metrics.json (build_and_eval_ensemble_n_with_train)
  - 02/09_baseline_*.csv           (evaluate_model_repeated_splits +
                                     evaluate_model_group_kfold, 16-model zoo)

Run from the same directory as full_pipeline_part1_data.py and
full_pipeline_part2_tuning.py (both are imported directly, no packaging).

Usage:
    python3 run_full_pipeline.py --stage smoke     # ~1 min, sanity check only
    python3 run_full_pipeline.py --stage baseline  # 16-model screen, slow
    python3 run_full_pipeline.py --stage tuned     # Optuna-tuned single models
    python3 run_full_pipeline.py --stage ensemble  # weighted ensembles
    python3 run_full_pipeline.py --stage all       # everything, in order
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import full_pipeline_part1_data as p1
import full_pipeline_part2_tuning as p2

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

PERFORMANCE_TARGETS = p1.PERFORMANCE_TARGETS
PROPERTY_TARGETS = p1.PROPERTY_TARGETS

# Which models get the FULL Optuna treatment for the "tuned" stage. Restricted
# to p2.TUNABLE_MODELS because that's all suggest_params()/make_model_from_params()
# actually support -- AdaBoost, SVR_poly, Lasso, PLS, KNN, BayesianRidge,
# GaussianProcess only ever run untuned (baseline stage) in this codebase.
CANDIDATE_TUNED_MODELS = p2.TUNABLE_MODELS

# Per your paper/README: only water permeability gets the log1p wrapper.
LOG_TARGETS = {"water permeability (LMH/bar)"}

# Per-target ensemble membership, matching what your README/checkpoint files
# say was actually used (2-model for properties + organic removal, 3-model
# for flux decline). Edit this if your own notes say otherwise.
ENSEMBLE_MEMBERS = {
    "flux decline ratio (%)": ["GradientBoosting", "RandomForest", "ExtraTrees"],
    "organic compound removal (%)": ["GradientBoosting", "LightGBM", "SVR_rbf"],
    "mean pore radius (nm)": ["KernelRidge_rbf", "RandomForest"],
    "overall porosity (%)": ["GradientBoosting", "KernelRidge_rbf"],
    "contact angle (degree)": ["GradientBoosting", "KernelRidge_rbf"],
}


def load_sheet1_features():
    df1 = p1.canonicalize_categoricals(p1.load_sheet1())
    df1 = p1.add_chemical_family_features(df1)
    df1 = p1.add_interaction_features(df1)
    exclude = PERFORMANCE_TARGETS + [p1.IRREVERSIBLE_FOULING_COL, "DOI", "SMILES"]
    feature_cols = [c for c in df1.columns if c not in exclude]
    numeric_cols = df1[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = df1[feature_cols].select_dtypes(exclude="number").columns.tolist()
    return df1, feature_cols, numeric_cols, categorical_cols


def load_sheet2_features():
    df2 = p1.canonicalize_categoricals(p1.load_sheet2())
    df2 = p1.add_chemical_family_features(df2)
    exclude = PROPERTY_TARGETS + ["DOI", "SMILES"]
    feature_cols = [c for c in df2.columns if c not in exclude]
    numeric_cols = df2[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = df2[feature_cols].select_dtypes(exclude="number").columns.tolist()
    return df2, feature_cols, numeric_cols, categorical_cols


# --------------------------------------------------------------------------
# STAGE 1: baseline 16-model screen -> 02_baseline_*.csv / 09_baseline_*.csv
# --------------------------------------------------------------------------
def run_baseline_stage(n_seeds_perf=30, n_seeds_prop=15, n_groupfolds=5):
    df1, _, num1, cat1 = load_sheet1_features()
    df2, _, num2, cat2 = load_sheet2_features()

    seeds_perf = list(range(1000, 1000 + n_seeds_perf))
    seeds_prop = list(range(1000, 1000 + n_seeds_prop))

    repeated_rows, grouped_rows = [], []

    # ---- performance targets (Sheet 1), random splits + DOI-grouped CV ----
    doi_mask = df1["DOI"].notna()
    for target in PERFORMANCE_TARGETS:
        mask = df1[target].notna() & doi_mask
        X = df1.loc[mask, num1 + cat1]
        y = df1.loc[mask, target]
        groups = df1.loc[mask, "DOI"]
        for model_name, factory in p2.BASELINE_MODEL_ZOO.items():
            rep = p2.evaluate_model_repeated_splits(factory, X, y, num1, cat1, seeds_perf)
            rep["target"] = target
            rep["model"] = model_name
            repeated_rows.append(rep)

            grp = p2.evaluate_model_group_kfold(factory, X, y, groups, num1, cat1, n_splits=n_groupfolds)
            grp["target"] = target
            grp["model"] = model_name
            grouped_rows.append(grp)
            print(f"[baseline/perf] {target[:25]:25s} {model_name:22s} "
                  f"R2={rep.r2.mean():.3f}  grouped R2={grp.r2.mean():.3f}")

    repeated_all = pd.concat(repeated_rows, ignore_index=True)
    grouped_all = pd.concat(grouped_rows, ignore_index=True)
    repeated_all.to_csv(OUT_DIR / "02_baseline_repeated_ALL.csv", index=False)
    grouped_all.to_csv(OUT_DIR / "02_baseline_grouped_ALL.csv", index=False)

    summary = (
        repeated_all.groupby(["target", "model"])["r2"].agg(r2_mean="mean", r2_std="std").reset_index()
    )
    rmse_summary = repeated_all.groupby(["target", "model"])["rmse"].mean().reset_index(name="rmse_mean")
    summary = summary.merge(rmse_summary, on=["target", "model"])
    grouped_summary = grouped_all.groupby(["target", "model"])["r2"].mean().reset_index(name="grouped_r2_mean")
    grouped_rmse = grouped_all.groupby(["target", "model"])["rmse"].mean().reset_index(name="grouped_rmse_mean")
    summary = summary.merge(grouped_summary, on=["target", "model"]).merge(grouped_rmse, on=["target", "model"])
    summary.to_csv(OUT_DIR / "02_baseline_summary_by_model.csv", index=False)

    # ---- property targets (Sheet 2), random splits only (no DOI grouping shown for these in your files) ----
    prop_rows = []
    for target in PROPERTY_TARGETS:
        mask = df2[target].notna()
        X = df2.loc[mask, num2 + cat2]
        y = df2.loc[mask, target]
        for model_name, factory in p2.BASELINE_MODEL_ZOO.items():
            rep = p2.evaluate_model_repeated_splits(factory, X, y, num2, cat2, seeds_prop)
            rep["target"] = target
            rep["model"] = model_name
            prop_rows.append(rep)
            print(f"[baseline/prop] {target[:25]:25s} {model_name:22s} R2={rep.r2.mean():.3f}")

    prop_all = pd.concat(prop_rows, ignore_index=True)
    prop_all.to_csv(OUT_DIR / "09_baseline_repeated_properties_ALL.csv", index=False)
    prop_summary = (
        prop_all.groupby(["target", "model"])["r2"].agg(r2_mean="mean", r2_std="std").reset_index()
    )
    prop_rmse = prop_all.groupby(["target", "model"])["rmse"].mean().reset_index(name="rmse_mean")
    prop_summary = prop_summary.merge(prop_rmse, on=["target", "model"])
    prop_summary.to_csv(OUT_DIR / "09_baseline_summary_properties.csv", index=False)

    print("\nBaseline stage done -> outputs/02_baseline_*.csv, outputs/09_baseline_*.csv")


# --------------------------------------------------------------------------
# STAGE 2: Optuna-tuned single models, seed=1000 -> 06_train_test_batch1.json
# --------------------------------------------------------------------------
def run_tuned_stage(n_trials=25, seed=1000):
    df1, _, num1, cat1 = load_sheet1_features()
    df2, _, num2, cat2 = load_sheet2_features()

    results = {}

    for target in PERFORMANCE_TARGETS:
        mask = df1[target].notna()
        X = df1.loc[mask, num1 + cat1]
        y = df1.loc[mask, target]
        log_target = target in LOG_TARGETS

        best = None
        for model_name in CANDIDATE_TUNED_MODELS:
            r = p2.tune_with_train_test_metrics(
                model_name, X, y, num1, cat1, seed=seed, n_trials=n_trials, log_target=log_target
            )
            print(f"[tuned/perf] {target[:25]:25s} {model_name:22s} "
                  f"test_r2={r['test_r2']:.3f} train_r2={r['train_r2']:.3f}")
            
            clean_target = target.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
            pd.DataFrame({'split': ['train']*len(r['y_train']) + ['test']*len(r['y_test']), 'y_true': list(r['y_train']) + list(r['y_test']), 'y_pred': list(r['pred_train']) + list(r['pred_test'])}).to_csv(OUT_DIR / f"predictions_{clean_target}_{model_name}.csv", index=False)
            r.pop('y_train', None); r.pop('pred_train', None); r.pop('y_test', None); r.pop('pred_test', None)

            if best is None or r["test_r2"] > best["test_r2"]:
                best = r
        results[target] = best

    for target in PROPERTY_TARGETS:
        mask = df2[target].notna()
        X = df2.loc[mask, num2 + cat2]
        y = df2.loc[mask, target]

        best = None
        for model_name in CANDIDATE_TUNED_MODELS:
            r = p2.tune_with_train_test_metrics(
                model_name, X, y, num2, cat2, seed=seed, n_trials=n_trials, log_target=False
            )
            print(f"[tuned/prop] {target[:25]:25s} {model_name:22s} "
                  f"test_r2={r['test_r2']:.3f} train_r2={r['train_r2']:.3f}")
            
            clean_target = target.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
            pd.DataFrame({'split': ['train']*len(r['y_train']) + ['test']*len(r['y_test']), 'y_true': list(r['y_train']) + list(r['y_test']), 'y_pred': list(r['pred_train']) + list(r['pred_test'])}).to_csv(OUT_DIR / f"predictions_{clean_target}_{model_name}.csv", index=False)
            r.pop('y_train', None); r.pop('pred_train', None); r.pop('y_test', None); r.pop('pred_test', None)

            if best is None or r["test_r2"] > best["test_r2"]:
                best = r
        results[target] = best

    with open(OUT_DIR / "06_train_test_batch1.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nTuned stage done -> outputs/06_train_test_batch1.json")


# --------------------------------------------------------------------------
# STAGE 3: weighted ensembles -> 08_ensemble_train_metrics.json
# --------------------------------------------------------------------------
def run_ensemble_stage(n_trials=25, seed=1000):
    df1, _, num1, cat1 = load_sheet1_features()
    df2, _, num2, cat2 = load_sheet2_features()

    results = {}
    for target, members in ENSEMBLE_MEMBERS.items():
        if target in PERFORMANCE_TARGETS:
            df, num_cols, cat_cols = df1, num1, cat1
        else:
            df, num_cols, cat_cols = df2, num2, cat2
        mask = df[target].notna()
        X = df.loc[mask, num_cols + cat_cols]
        y = df.loc[mask, target]

        r = p2.build_and_eval_ensemble_n_with_train(
            X, y, num_cols, cat_cols, members, seed=seed, n_trials=n_trials
        )
        results[target] = r
        print(f"[ensemble] {target[:25]:25s} members={members} "
              f"test_r2={r['test_r2']:.3f} train_r2={r['train_r2']:.3f}")

    with open(OUT_DIR / "08_ensemble_train_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nEnsemble stage done -> outputs/08_ensemble_train_metrics.json")


# --------------------------------------------------------------------------
# Fast sanity check before committing to a full run
# --------------------------------------------------------------------------
def run_smoke_stage():
    df1, _, num1, cat1 = load_sheet1_features()
    target = "water permeability (LMH/bar)"
    X = df1[num1 + cat1]
    y = df1[target]
    res, params_log = p2.tuned_repeated_splits(
        "ExtraTrees", X, y, num1, cat1, seeds=[1000, 1001], n_trials=5, inner_splits=3
    )
    print(res)
    print("Smoke test OK -- data loads, features build, tuning runs end to end.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "baseline", "tuned", "ensemble", "all"], required=True)
    parser.add_argument("--n-trials", type=int, default=25, help="Optuna trials per (target, model)")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--n-seeds-perf", type=int, default=30, help="baseline stage only")
    parser.add_argument("--n-seeds-prop", type=int, default=15, help="baseline stage only")
    args = parser.parse_args()

    if args.stage == "smoke":
        run_smoke_stage()
    elif args.stage == "baseline":
        run_baseline_stage(n_seeds_perf=args.n_seeds_perf, n_seeds_prop=args.n_seeds_prop)
    elif args.stage == "tuned":
        run_tuned_stage(n_trials=args.n_trials, seed=args.seed)
    elif args.stage == "ensemble":
        run_ensemble_stage(n_trials=args.n_trials, seed=args.seed)
    elif args.stage == "all":
        run_smoke_stage()
        run_baseline_stage(n_seeds_perf=args.n_seeds_perf, n_seeds_prop=args.n_seeds_prop)
        run_tuned_stage(n_trials=args.n_trials, seed=args.seed)
        run_ensemble_stage(n_trials=args.n_trials, seed=args.seed)
