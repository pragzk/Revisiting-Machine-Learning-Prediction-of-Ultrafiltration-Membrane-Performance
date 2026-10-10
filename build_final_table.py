"""
Reads whatever's on disk in results/ across datasets A, B, C and builds ONE
honest final table. Run this any time -- it only reads, never re-fits
anything, so it's free/instant and safe to run after every partial batch.

Usage: python3 build_final_table.py
"""
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"

GAO_TABLE_1 = {  # Dataset A targets
    "water permeability (LMH/bar)": {"r2": 0.83, "rmse": 68.24},
    "organic compound removal (%)": {"r2": 0.84, "rmse": 6.60},
    "flux decline ratio (%)": {"r2": 0.78, "rmse": 9.50},
    "flux recovery ratio (%)": {"r2": 0.62, "rmse": 7.89},
    "reversible fouling ratio (%)": {"r2": 0.73, "rmse": 10.33},
}
GAO_TABLE_2 = {  # Dataset B targets
    "overall porosity (%)": {"r2": 0.66, "rmse": 7.61},
    "mean pore radius (nm)": {"r2": 0.87, "rmse": 7.59},
    "contact angle (degree)": {"r2": 0.76, "rmse": 5.87},
}
GAO_ALL = {**GAO_TABLE_1, **GAO_TABLE_2}


def _load(name):
    cv_path = RESULTS_DIR / f"dataset_{name}_repeated_cv.csv"
    single_path = RESULTS_DIR / f"dataset_{name}_single_split.csv"
    cv = pd.read_csv(cv_path) if cv_path.exists() else pd.DataFrame()
    single = pd.read_csv(single_path) if single_path.exists() else pd.DataFrame()
    return cv, single


def build_ab_table():
    """Dataset A + B vs Gao -- the real 'did we beat the paper' claim.
    Primary column: single-split R2 (apples-to-apples with Gao's own stated
    protocol). Secondary: repeated-CV mean+/-std (the honest, stricter,
    lower-variance number -- what we'd actually trust for a real claim of
    'generalizes', but not what Gao's number itself represents).
    """
    rows = []
    for name in ["a", "b"]:
        cv, single = _load(name)
        if cv.empty or single.empty:
            continue
        for target, gao in GAO_ALL.items():
            cv_t = cv[cv.target == target]
            single_t = single[single.target == target]
            if cv_t.empty and single_t.empty:
                continue
            # best model by single-split R2 (the apples-to-apples metric)
            if not single_t.empty:
                best_row = single_t.loc[single_t.test_r2.idxmax()]
                best_model = best_row["model"]
                best_single_r2 = best_row["test_r2"]
                best_single_rmse = best_row["test_rmse"]
            else:
                best_model, best_single_r2, best_single_rmse = None, None, None
            cv_match = cv_t[cv_t.model == best_model] if best_model else pd.DataFrame()
            rows.append({
                "target": target, "dataset": name.upper(),
                "gao_r2": gao["r2"], "gao_rmse": gao["rmse"],
                "best_model": best_model,
                "our_single_split_r2": round(best_single_r2, 3) if best_single_r2 is not None else None,
                "beats_gao_single_split": (best_single_r2 > gao["r2"]) if best_single_r2 is not None else None,
                "our_repeated_cv_r2_mean": round(cv_match.r2_mean.iloc[0], 3) if len(cv_match) else None,
                "our_repeated_cv_r2_std": round(cv_match.r2_std.iloc[0], 3) if len(cv_match) else None,
                "beats_gao_repeated_cv": (cv_match.r2_mean.iloc[0] > gao["r2"]) if len(cv_match) else None,
                "n_models_evaluated": single_t.model.nunique(),
            })
    return pd.DataFrame(rows)


def build_c_table():
    """Dataset C -- oracle-properties upper bound. NOT comparable to Gao Table 1
    numbers (different task/inputs), so no gao_r2 column here on purpose."""
    cv, single = _load("c")
    if cv.empty:
        return pd.DataFrame()
    rows = []
    for target in cv.target.unique():
        cv_t = cv[cv.target == target]
        single_t = single[single.target == target] if not single.empty else pd.DataFrame()
        best_cv_row = cv_t.loc[cv_t.r2_mean.idxmax()]
        rows.append({
            "target": target, "dataset": "C (oracle properties -- NOT vs Gao Table 1)",
            "best_model": best_cv_row["model"],
            "repeated_cv_r2_mean": round(best_cv_row.r2_mean, 3),
            "repeated_cv_r2_std": round(best_cv_row.r2_std, 3),
            "single_split_r2": round(single_t[single_t.model == best_cv_row["model"]].test_r2.iloc[0], 3)
            if len(single_t[single_t.model == best_cv_row["model"]]) else None,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    ab = build_ab_table()
    print("=" * 100)
    print("MAIN RESULT: Dataset A + B vs Gao et al. Table 1 / Table 2")
    print("=" * 100)
    if not ab.empty:
        print(ab.to_string(index=False))
        n_beat_single = ab.beats_gao_single_split.sum()
        n_beat_cv = ab.beats_gao_repeated_cv.sum()
        print(f"\nBeats Gao on single-split (apples-to-apples) R2: {n_beat_single}/{len(ab)} targets")
        print(f"Beats Gao on repeated-CV (stricter) R2:            {n_beat_cv}/{len(ab)} targets")
        ab.to_csv(RESULTS_DIR / "FINAL_main_comparison_vs_gao.csv", index=False)
    else:
        print("No Dataset A/B results on disk yet.")

    c = build_c_table()
    print("\n" + "=" * 100)
    print("SEPARATE (not a Gao-Table-1 comparison): Dataset C oracle-properties upper bound")
    print("=" * 100)
    if not c.empty:
        print(c.to_string(index=False))
        c.to_csv(RESULTS_DIR / "FINAL_dataset_c_oracle_upper_bound.csv", index=False)
    else:
        print("No Dataset C results on disk yet.")
