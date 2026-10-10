"""
Main driver: Dataset C (membrane properties + operational conditions -> performance).

*** IMPORTANT — READ BEFORE QUOTING THESE NUMBERS ANYWHERE ***
Dataset C's inputs are the membrane's ACTUAL MEASURED properties (mean pore
radius, overall porosity, contact angle) plus operational conditions. In a
real design workflow you don't know a membrane's properties until you've
already fabricated and tested it -- so this is NOT the same task as Gao et
al.'s Table 1 (which predicts performance from a fabrication *recipe*,
before anything is made). This is closer to the paper's own Table S7
setup. Treat these results as an "oracle-properties upper bound" -- useful
for understanding how predictive properties are of performance -- and keep
them in a SEPARATE table from the Dataset A vs. Gao-Table-1 comparison.
No PAPER_TABLE_C benchmark dict is defined below on purpose: we don't have
the paper's own Table S7 numbers (only descriptive text, not the exact
values), so there is nothing honest to compare against numerically here.

Same checkpointed-per-(target,model) design as run_dataset_a.py/run_dataset_b.py.
Stacking is DELIBERATELY excluded from MODEL_NAMES -- see chat history: it was
consistently the worst or most unstable model on Dataset A/B (up to R2=-0.765,
std=0.53) while costing 10x the runtime of every other model combined.

Usage:
    python3 run_dataset_c.py --target-idx 0
    python3 run_dataset_c.py --target-idx 0 --model LightGBM
    python3 run_dataset_c.py --summary
"""
from __future__ import annotations
import time
import warnings
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

from data_loader import load_dataset_c
from feature_builder import build_features_dataset_c
from modeling import (
    make_lightgbm_pipeline, make_extratrees_pipeline, make_svr_pipeline,
    make_histgb_pipeline, make_xgboost_pipeline, make_catboost_pipeline,
)
from tuning import tune_lightgbm, tune_histgb, tune_extratrees, tune_svr, tune_xgboost, tune_catboost
from evaluation import repeated_kfold_eval, single_split_eval, results_to_dataframe

# NOTE: point this at whatever folder your run_dataset_a.py / run_dataset_b.py
# are actually writing to (check the RESULTS_DIR line in those files) so all
# three datasets' CSVs land in one place for build_final_table.py to read.
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

N_REPEATS = 5
N_SPLITS = 5
N_TRIALS_GBM = 8
N_TRIALS_OTHER = 6

CV_CSV = RESULTS_DIR / "dataset_c_repeated_cv.csv"
SINGLE_CSV = RESULTS_DIR / "dataset_c_single_split.csv"
PARAMS_JSON = RESULTS_DIR / "dataset_c_tuned_params.json"

# Stacking excluded on purpose -- see module docstring.
MODEL_NAMES = ["LightGBM", "HistGB", "ExtraTrees", "SVR",
               "XGBoost_reproduction", "CatBoost_reproduction"]


def _load_tuned_params():
    return json.loads(PARAMS_JSON.read_text()) if PARAMS_JSON.exists() else {}


def _save_tuned_params(d):
    PARAMS_JSON.write_text(json.dumps(d, indent=2, default=str))


def _already_done_pairs():
    if not CV_CSV.exists():
        return set()
    df = pd.read_csv(CV_CSV)
    return set(zip(df["target"], df["model"]))


def _get_or_tune(target, model_name, X, y, cat_cols, num_cols, seed=0):
    all_params = _load_tuned_params()
    target_params = all_params.get(target, {})
    if model_name in target_params:
        return target_params[model_name]

    mask = y.notna()
    Xf, yf = X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)

    if model_name == "LightGBM":
        p = tune_lightgbm(Xf, yf, cat_cols, num_cols, n_trials=N_TRIALS_GBM, seed=seed)
    elif model_name == "HistGB":
        p = tune_histgb(Xf, yf, cat_cols, num_cols, n_trials=N_TRIALS_OTHER, seed=seed)
    elif model_name == "ExtraTrees":
        p = tune_extratrees(Xf, yf, cat_cols, num_cols, n_trials=N_TRIALS_OTHER, seed=seed)
    elif model_name == "SVR":
        p = tune_svr(Xf, yf, cat_cols, num_cols, n_trials=N_TRIALS_OTHER, seed=seed)
    elif model_name == "XGBoost_reproduction":
        p = tune_xgboost(Xf, yf, cat_cols, num_cols, n_trials=N_TRIALS_GBM, seed=seed)
    elif model_name == "CatBoost_reproduction":
        p = tune_catboost(Xf, yf, cat_cols, num_cols, n_trials=N_TRIALS_OTHER, seed=seed)
    else:
        raise ValueError(model_name)

    all_params.setdefault(target, {})[model_name] = p
    _save_tuned_params(all_params)
    return p


def _factory_for(model_name, target, X, y, cat_cols, num_cols):
    factories = {
        "LightGBM": make_lightgbm_pipeline, "HistGB": make_histgb_pipeline,
        "ExtraTrees": make_extratrees_pipeline, "SVR": make_svr_pipeline,
        "XGBoost_reproduction": make_xgboost_pipeline, "CatBoost_reproduction": make_catboost_pipeline,
    }
    if model_name not in factories:
        raise ValueError(model_name)
    return factories[model_name], _get_or_tune(target, model_name, X, y, cat_cols, num_cols)


def run_one(target: str, model_name: str, quick: bool = False):
    dataset = load_dataset_c()
    X, Y, categorical_cols, numeric_cols = build_features_dataset_c(dataset)
    y = Y[target]

    done = _already_done_pairs()
    if (target, model_name) in done:
        print(f"SKIP (already done): {target} / {model_name}")
        return

    t0 = time.time()
    n_repeats = 2 if quick else N_REPEATS
    factory, kwargs = _factory_for(model_name, target, X, y, categorical_cols, numeric_cols)
    print(f"tuning/lookup done at {time.time()-t0:.1f}s, params={kwargs}")

    res = repeated_kfold_eval(
        factory, X, y, categorical_cols, numeric_cols,
        model_name=model_name, target_name=target,
        n_repeats=n_repeats, n_splits=N_SPLITS, random_state=0, factory_kwargs=kwargs,
    )
    single = single_split_eval(
        factory, X, y, categorical_cols, numeric_cols,
        model_name=model_name, target_name=target, test_size=0.2, random_state=42,
        factory_kwargs=kwargs,
    )
    print(f"[{time.time()-t0:6.1f}s] {target[:30]:30s} | {model_name:24s} | "
          f"CV R2={np.mean(res.r2_scores):+.3f}+-{np.std(res.r2_scores):.3f} | "
          f"CV RMSE={np.mean(res.rmse_scores):7.3f}+-{np.std(res.rmse_scores):.3f} | "
          f"single-split R2={single['test_r2']:+.3f}")

    cv_df = results_to_dataframe([res])
    cv_df.to_csv(CV_CSV, mode="a", header=not CV_CSV.exists(), index=False)
    pd.DataFrame([single]).to_csv(SINGLE_CSV, mode="a", header=not SINGLE_CSV.exists(), index=False)
    print(f">>> checkpointed {target} / {model_name}")


def run_all_models_for_target(target: str, quick: bool = False, time_budget_s: float = 240):
    t0 = time.time()
    for model_name in MODEL_NAMES:
        if time.time() - t0 > time_budget_s:
            print(f"Time budget ({time_budget_s}s) reached, stopping early for this call.")
            break
        run_one(target, model_name, quick=quick)


def print_summary():
    if not CV_CSV.exists():
        print("No results yet.")
        return
    cv_df = pd.read_csv(CV_CSV)
    print("\n=== SUMMARY (from disk): Dataset C repeated CV mean R2 ===")
    print("*** ORACLE-PROPERTIES UPPER BOUND -- not the same task as Gao Table 1, do not compare directly ***")
    pivot_r2 = cv_df.pivot_table(index="model", columns="target", values="r2_mean")
    print(pivot_r2.round(3).to_string())
    print("\n=== SUMMARY (from disk): Dataset C repeated CV mean RMSE ===")
    pivot_rmse = cv_df.pivot_table(index="model", columns="target", values="rmse_mean")
    print(pivot_rmse.round(3).to_string())
    print(f"\nCompleted (target, model) pairs: {len(cv_df)} / {len(load_dataset_c()['targets']) * len(MODEL_NAMES)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-idx", type=int, default=None)
    parser.add_argument("--model", type=str, default=None, choices=MODEL_NAMES)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--time-budget", type=float, default=240)
    args = parser.parse_args()

    if args.summary:
        print_summary()
    elif args.target_idx is not None:
        targets = load_dataset_c()["targets"]
        if not (0 <= args.target_idx < len(targets)):
            print(f"ERROR: --target-idx must be in [0, {len(targets)-1}] for Dataset C. "
                  f"Valid targets: {list(enumerate(targets))}")
        else:
            target = targets[args.target_idx]
            if args.model:
                run_one(target, args.model, quick=args.quick)
            else:
                run_all_models_for_target(target, quick=args.quick, time_budget_s=args.time_budget)
    else:
        parser.print_help()
