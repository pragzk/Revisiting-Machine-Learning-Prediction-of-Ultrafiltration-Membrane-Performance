"""
STANDALONE experiment -- does not touch modeling.py / evaluation.py / etc.
Tests whether giving the model the polymer's actual molecular structure
(RDKit Morgan fingerprints from its SMILES) helps predict contact angle,
overall porosity, and mean pore radius -- vs. only knowing "which of 11
polymers" as a bare category, which is all your current pipeline sees.

Requirements: pip install rdkit-pypi optuna scikit-learn category_encoders lightgbm

Usage:
    python rdkit_contact_angle_experiment.py --xlsx path/to/your/da_data.xlsx
"""
import argparse
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


# --------------------------------------------------------------------
# RDKit Morgan fingerprints for the polymer SMILES
# --------------------------------------------------------------------
def smiles_to_fp_array(smiles, n_bits=64, radius=2):
    """Compact Morgan fingerprint. 64 bits (not the usual 1024/2048) on
    purpose -- we only have 11 distinct polymers and ~300 rows total, a
    huge fingerprint would just be 990+ near-constant columns of noise
    for a tree model to sift through. Returns None if SMILES unparseable.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def add_polymer_fingerprints(df, smiles_col, n_bits=64):
    df = df.copy()
    fps = df[smiles_col].apply(lambda s: smiles_to_fp_array(s, n_bits=n_bits))
    n_failed = fps.isna().sum()
    if n_failed:
        print(f"WARNING: {n_failed} SMILES failed to parse -- those rows get all-zero fingerprint "
              f"(equivalent to 'unknown polymer'), check them manually if this number is large.")
    fp_matrix = np.vstack([f if f is not None else np.zeros(n_bits, dtype=int) for f in fps])
    fp_cols = [f"polymer_fp_{i}" for i in range(n_bits)]
    fp_df = pd.DataFrame(fp_matrix, columns=fp_cols, index=df.index)
    return pd.concat([df, fp_df], axis=1), fp_cols


# --------------------------------------------------------------------
# Leakage-safe preprocessing (fit on train fold only, same discipline as everywhere else)
# --------------------------------------------------------------------
def build_preprocessor(numeric_cols, categorical_cols):
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)])


def suggest_et_params(trial):
    return dict(n_estimators=trial.suggest_int("n_estimators", 100, 800),
                max_depth=trial.suggest_int("max_depth", 3, 30),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                max_features=trial.suggest_float("max_features", 0.3, 1.0))


def tune_and_eval_one_seed(X, y, numeric_cols, categorical_cols, seed, n_trials=25, inner_splits=4, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    inner_kf = KFold(n_splits=inner_splits, shuffle=True, random_state=seed)

    def objective(trial):
        params = suggest_et_params(trial)
        scores = []
        for tr_idx, va_idx in inner_kf.split(X_train):
            pre = build_preprocessor(numeric_cols, categorical_cols)
            Xtr = pre.fit_transform(X_train.iloc[tr_idx])
            Xva = pre.transform(X_train.iloc[va_idx])
            m = ExtraTreesRegressor(random_state=42, n_jobs=-1, **params)
            m.fit(Xtr, y_train.iloc[tr_idx])
            scores.append(r2_score(y_train.iloc[va_idx], m.predict(Xva)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    pre = build_preprocessor(numeric_cols, categorical_cols)
    Xtr = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)
    model = ExtraTreesRegressor(random_state=42, n_jobs=-1, **study.best_params)
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    return {"seed": seed, "r2": r2_score(y_test, pred), "rmse": rmse(y_test, pred),
            "mae": mean_absolute_error(y_test, pred), "best_params": study.best_params}


GAO_TABLE_2 = {"overall porosity (%)": 0.66, "mean pore radius (nm)": 0.87, "contact angle (degree)": 0.76}


def load_sheet2(xlsx_path):
    df = pd.read_excel(xlsx_path, sheet_name="2", header=1).dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", required=True, help="path to your da_data.xlsx (or es2c05404_si_002.xlsx)")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--n-bits", type=int, default=64, help="fingerprint size")
    args = parser.parse_args()

    df = load_sheet2(args.xlsx)
    smiles_col = [c for c in df.columns if "SMILES" in c.upper()][0]
    print(f"Using SMILES column: '{smiles_col}'")
    df, fp_cols = add_polymer_fingerprints(df, smiles_col, n_bits=args.n_bits)

    prop_targets = ["mean pore radius (nm)", "overall porosity (%)", "contact angle (degree)"]
    exclude = prop_targets + ["DOI", smiles_col]

    seeds = list(range(1000, 1000 + args.n_seeds))
    all_results = []

    for target in prop_targets:
        y = df[target].dropna()
        base_feature_cols = [c for c in df.columns if c not in exclude and c not in fp_cols]

        for variant_name, feature_cols in [
            ("baseline (polymer as bare category, no fingerprint)", base_feature_cols),
            ("with_polymer_fingerprint", base_feature_cols + fp_cols),
        ]:
            X = df.loc[y.index, feature_cols].copy()
            numeric_cols = X.select_dtypes(include="number").columns.tolist()
            categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
            for c in categorical_cols:
                X[c] = X[c].astype(str)

            print(f"\n--- {target} | {variant_name} ---")
            for seed in seeds:
                res = tune_and_eval_one_seed(X, y, numeric_cols, categorical_cols, seed,
                                              n_trials=args.n_trials)
                res["target"] = target
                res["variant"] = variant_name
                all_results.append(res)
                print(f"  seed={seed}  R2={res['r2']:.3f}  RMSE={res['rmse']:.3f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("rdkit_experiment_results.csv", index=False)

    print("\n" + "=" * 90)
    print("SUMMARY: baseline vs. with polymer fingerprint (mean R2 across seeds)")
    print("=" * 90)
    summary = results_df.groupby(["target", "variant"])["r2"].agg(["mean", "std"]).round(3)
    print(summary.to_string())
    print("\nGao benchmarks for reference:", GAO_TABLE_2)
    print("\nFull per-seed results saved to rdkit_experiment_results.csv")
