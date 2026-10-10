"""
SCRIPT 4 of 4 -- Ablation: separate the feature-engineering effect from the
model-selection (GradientBoosting) effect. 2x2 per target:
    {old features, new features} x {old algorithm (XGBoost), new algorithm}

CatBoost is deliberately excluded: this project's own notes record it
hanging 2+ hours on a single (target, model) pair at this dataset size.
XGBoost alone stands in for "the original paper's algorithm family" here.

Requires: pip install xgboost

Run from the same directory as full_pipeline_part1_data.py and
full_pipeline_part2_tuning.py.

Usage:
    python3 04_ablation.py
    python3 04_ablation.py --n-trials 25 --seed 1000
"""
import argparse
import json
import warnings
from pathlib import Path

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

import full_pipeline_part1_data as p1
import full_pipeline_part2_tuning as p2

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

PERFORMANCE_TARGETS = p1.PERFORMANCE_TARGETS

# Model to use as "new algorithm" per target -- matches Tables 1-2's solo
# winners. Update flux recovery ratio's entry once Script 1 has an answer.
BEST_SOLO_MODEL = {
    "water permeability (LMH/bar)": ("LightGBM", True),
    "organic compound removal (%)": ("GradientBoosting", False),
    "flux decline ratio (%)": ("GradientBoosting", False),
    "flux recovery ratio (%)": ("HistGradientBoosting", False),
    "reversible fouling ratio (%)": ("GradientBoosting", False),
}


def load_sheet1_features_new():
    df1 = p1.canonicalize_categoricals(p1.load_sheet1())
    df1 = p1.add_chemical_family_features(df1)
    df1 = p1.add_interaction_features(df1)
    exclude = PERFORMANCE_TARGETS + [p1.IRREVERSIBLE_FOULING_COL, "DOI", "SMILES"]
    feature_cols = [c for c in df1.columns if c not in exclude]
    numeric_cols = df1[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = df1[feature_cols].select_dtypes(exclude="number").columns.tolist()
    return df1, numeric_cols, categorical_cols


def load_sheet1_features_old():
    """Same rows, WITHOUT chemical-family taxonomy or interaction features --
    just canonicalized categoricals, raw additive column as-is."""
    df1 = p1.canonicalize_categoricals(p1.load_sheet1())
    exclude = PERFORMANCE_TARGETS + [p1.IRREVERSIBLE_FOULING_COL, "DOI", "SMILES"]
    feature_cols = [c for c in df1.columns if c not in exclude]
    numeric_cols = df1[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = df1[feature_cols].select_dtypes(exclude="number").columns.tolist()
    return df1, numeric_cols, categorical_cols


def make_xgb_factory():
    from xgboost import XGBRegressor
    return lambda: XGBRegressor(n_estimators=400, random_state=42, verbosity=0)


def main(n_trials=25, seed=1000):
    df1_new, num1_new, cat1_new = load_sheet1_features_new()
    df1_old, num1_old, cat1_old = load_sheet1_features_old()
    xgb_factory = make_xgb_factory()
    results = {}

    for target in PERFORMANCE_TARGETS:
        model_name, log_target = BEST_SOLO_MODEL[target]
        row = {}

        for feat_label, (df, num_cols, cat_cols) in [
            ("old_features", (df1_old, num1_old, cat1_old)),
            ("new_features", (df1_new, num1_new, cat1_new)),
        ]:
            mask = df[target].notna()
            X = df.loc[mask, num_cols + cat_cols]
            y = df.loc[mask, target]

            # old algorithm family (XGBoost, untuned defaults)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
            pre = p2.build_preprocessor(num_cols, cat_cols)
            Xtr = pre.fit_transform(X_train)
            Xte = pre.transform(X_test)
            xgb_model = xgb_factory()
            xgb_model.fit(Xtr, y_train)
            xgb_r2 = r2_score(y_test, xgb_model.predict(Xte))
            row[f"{feat_label}__old_algo_xgboost"] = xgb_r2

            # new algorithm zoo, tuned
            r = p2.tune_with_train_test_metrics(
                model_name, X, y, num_cols, cat_cols, seed=seed, n_trials=n_trials, log_target=log_target
            )
            row[f"{feat_label}__new_algo_{model_name.lower()}"] = r["test_r2"]

            print(f"[ablation] {target[:22]:22s} {feat_label:14s} "
                  f"XGBoost={xgb_r2:.3f}  {model_name}={r['test_r2']:.3f}")

        results[target] = row

    with open(OUT_DIR / "ablation_features_vs_algorithm.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved outputs/ablation_features_vs_algorithm.json")
    print("Read as a 2x2 per target: if new_features moves the number more than")
    print("new_algo does (holding the other fixed), that's evidence for the")
    print("feature-engineering story; if new_algo moves it more, that's evidence")
    print("for the algorithm-selection story. Report the actual deltas.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    main(n_trials=args.n_trials, seed=args.seed)
