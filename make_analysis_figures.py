"""
Regenerates the analysis figures (heatmap, leakage comparison, overfitting
gap, seed variance, discrepancy-resolution bars) directly from your
pipeline's own output files. Run this any time those files change --
it only reads, never re-fits anything.

Expected inputs (all produced by run_full_pipeline.py):
    outputs/02_baseline_summary_by_model.csv
    outputs/09_baseline_summary_properties.csv
    outputs/02_baseline_repeated_ALL.csv
    outputs/09_baseline_repeated_properties_ALL.csv
    outputs/06_train_test_batch1.json

Usage:
    python3 make_analysis_figures.py
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.size"] = 10

IN_DIR = Path("outputs")
OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    "ExtraTrees", "RandomForest", "GradientBoosting", "HistGradientBoosting",
    "LightGBM", "AdaBoost", "GaussianProcess", "Ridge", "BayesianRidge", "PLS",
    "Lasso", "ElasticNet", "KernelRidge_rbf", "SVR_rbf", "SVR_poly", "KNN",
]
PERF_TARGETS = [
    "water permeability (LMH/bar)", "organic compound removal (%)",
    "flux decline ratio (%)", "flux recovery ratio (%)", "reversible fouling ratio (%)",
]
PROP_TARGETS = ["overall porosity (%)", "mean pore radius (nm)", "contact angle (degree)"]
ALL_TARGETS = PERF_TARGETS + PROP_TARGETS


def short(t):
    return t.split(" (")[0]


# --------------------------------------------------------------------------
# Figure 1: 16-model x 8-target untuned R2 heatmap
# --------------------------------------------------------------------------
def make_heatmap():
    perf = pd.read_csv(IN_DIR / "02_baseline_summary_by_model.csv")
    prop = pd.read_csv(IN_DIR / "09_baseline_summary_properties.csv")

    heat = pd.DataFrame(index=MODEL_ORDER, columns=ALL_TARGETS, dtype=float)
    for t in PERF_TARGETS:
        for m in MODEL_ORDER:
            row = perf[(perf.target == t) & (perf.model == m)]
            if len(row):
                heat.loc[m, t] = row["r2_mean"].values[0]
    for t in PROP_TARGETS:
        for m in MODEL_ORDER:
            row = prop[(prop.target == t) & (prop.model == m)]
            if len(row):
                heat.loc[m, t] = row["r2_mean"].values[0]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(heat.values, cmap="RdYlGn", vmin=-0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(ALL_TARGETS)))
    ax.set_xticklabels([short(t) for t in ALL_TARGETS], rotation=35, ha="right")
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER)
    for i in range(len(MODEL_ORDER)):
        for j in range(len(ALL_TARGETS)):
            v = heat.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                         color="white" if v < 0.3 or v > 0.85 else "black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean untuned test $R^2$")
    ax.set_title("Untuned Baseline $R^2$ -- 16 Models x 8 Targets")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_baseline_heatmap.png", dpi=200)
    plt.close()
    print("Saved fig_baseline_heatmap.png")


# --------------------------------------------------------------------------
# Figure 2: random-split vs DOI-grouped R2 (leakage check)
# --------------------------------------------------------------------------
def make_leakage_comparison():
    perf = pd.read_csv(IN_DIR / "02_baseline_summary_by_model.csv")

    # top untuned model per target, from the summary table itself -- no hardcoding needed
    labels, rand_vals, grp_vals = [], [], []
    for t in PERF_TARGETS:
        sub = perf[perf.target == t].sort_values("r2_mean", ascending=False)
        top = sub.iloc[0]
        labels.append(f"{short(t)}\n({top['model']})")
        rand_vals.append(top["r2_mean"])
        grp_vals.append(top["grouped_r2_mean"])

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, rand_vals, width, label="Random 80/20 split (mean of seeds)", color="#4C72B0")
    ax.bar(x + width / 2, grp_vals, width, label="DOI-grouped 5-fold CV", color="#C44E52")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("$R^2$")
    ax.set_title("Random-Split vs. DOI-Grouped $R^2$: Literature-Source Leakage Check")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_leakage_comparison.png", dpi=200)
    plt.close()
    print("Saved fig_leakage_comparison.png")


# --------------------------------------------------------------------------
# Figure 3: train vs test R2 (overfitting gap) from the tuned-model JSON
# --------------------------------------------------------------------------
def make_overfitting_gap():
    with open(IN_DIR / "06_train_test_batch1.json") as f:
        tuned = json.load(f)

    targets = list(tuned.keys())
    train_r2 = [tuned[t]["train_r2"] for t in targets if "train_r2" in tuned[t]]
    test_r2 = [tuned[t]["test_r2"] for t in targets if "train_r2" in tuned[t]]
    labels = [short(t) for t in targets if "train_r2" in tuned[t]]

    if len(labels) < len(targets):
        skipped = [short(t) for t in targets if "train_r2" not in tuned[t]]
        print(f"NOTE: skipping targets with no train_r2 recorded: {skipped}")

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, train_r2, width, label="Train $R^2$", color="#4C72B0")
    ax.bar(x + width / 2, test_r2, width, label="Test $R^2$", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("$R^2$")
    ax.set_ylim(0, 1.05)
    ax.set_title("Train vs. Test $R^2$: Overfitting Gap (Tuned Models, seed=1000)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_overfitting_gap.png", dpi=200)
    plt.close()
    print("Saved fig_overfitting_gap.png")


# --------------------------------------------------------------------------
# Figure 4: seed-to-seed variance boxplots
# --------------------------------------------------------------------------
def make_seed_variance():
    perf_rep = pd.read_csv(IN_DIR / "02_baseline_repeated_ALL.csv")
    prop_rep = pd.read_csv(IN_DIR / "09_baseline_repeated_properties_ALL.csv")
    perf_summary = pd.read_csv(IN_DIR / "02_baseline_summary_by_model.csv")
    prop_summary = pd.read_csv(IN_DIR / "09_baseline_summary_properties.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    data0, labels0 = [], []
    for t in PERF_TARGETS:
        top_model = perf_summary[perf_summary.target == t].sort_values(
            "r2_mean", ascending=False).iloc[0]["model"]
        vals = perf_rep[(perf_rep.target == t) & (perf_rep.model == top_model)]["r2"].values
        data0.append(vals)
        labels0.append(f"{short(t)}\n({top_model})")
    axes[0].boxplot(data0, tick_labels=labels0)
    axes[0].set_ylabel("$R^2$ (30 seeds)")
    axes[0].set_title("Performance Targets")
    axes[0].tick_params(axis="x", labelsize=8, rotation=20)

    data1, labels1 = [], []
    for t in PROP_TARGETS:
        top_model = prop_summary[prop_summary.target == t].sort_values(
            "r2_mean", ascending=False).iloc[0]["model"]
        vals = prop_rep[(prop_rep.target == t) & (prop_rep.model == top_model)]["r2"].values
        data1.append(vals)
        labels1.append(f"{short(t)}\n({top_model})")
    axes[1].boxplot(data1, tick_labels=labels1)
    axes[1].set_ylabel("$R^2$ (15 seeds)")
    axes[1].set_title("Property Targets")
    axes[1].tick_params(axis="x", labelsize=8, rotation=20)

    fig.suptitle("Seed-to-Seed Variance in Test $R^2$ (Untuned Baseline)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_seed_variance.png", dpi=200)
    plt.close()
    print("Saved fig_seed_variance.png")


if __name__ == "__main__":
    make_heatmap()
    make_leakage_comparison()
    make_overfitting_gap()
    make_seed_variance()
    print("\nAll figures regenerated in outputs/figures/")
