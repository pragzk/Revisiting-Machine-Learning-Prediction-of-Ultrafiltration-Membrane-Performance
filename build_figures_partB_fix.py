import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = 'outputs/figures'

# -----------------
# 4. fig_ablation_2x2.png
# -----------------
# Load repeated_seed to get target names (Performance targets)
with open('outputs/repeated_seed_tuned_models.json', 'r') as f:
    rep_seed = json.load(f)
gao_vals = {
    'water permeability (LMH/bar)': 0.83,
    'organic compound removal (%)': 0.84,
    'flux decline ratio (%)': 0.78,
    'flux recovery ratio (%)': 0.62,
    'reversible fouling ratio (%)': 0.73,
    'overall porosity (%)': 0.66,
    'mean pore radius (nm)': 0.87,
    'contact angle (degree)': 0.76
}
targets = list(gao_vals.keys())

with open('outputs/ablation_features_vs_algorithm.json', 'r') as f:
    abl = json.load(f)

perf_targets = [t for t in targets if t in abl]

oldF_oldA = []
newF_oldA = []
oldF_newA = []
newF_newA = []

for t in perf_targets:
    target_data = abl[t]
    
    # Identify the actual keys based on prefixes
    val_oldF_oldA = None
    val_newF_oldA = None
    val_oldF_newA = None
    val_newF_newA = None
    
    for k, v in target_data.items():
        if k.startswith("old_features__old_algo_"): val_oldF_oldA = v
        elif k.startswith("new_features__old_algo_"): val_newF_oldA = v
        elif k.startswith("old_features__new_algo_"): val_oldF_newA = v
        elif k.startswith("new_features__new_algo_"): val_newF_newA = v
        
    oldF_oldA.append(val_oldF_oldA)
    newF_oldA.append(val_newF_oldA)
    oldF_newA.append(val_oldF_newA)
    newF_newA.append(val_newF_newA)

x = np.arange(len(perf_targets))
w = 0.2
plt.figure(figsize=(14, 6))
plt.bar(x - 1.5*w, oldF_oldA, w, label='Old Features + XGBoost')
plt.bar(x - 0.5*w, newF_oldA, w, label='New Features + XGBoost')
plt.bar(x + 0.5*w, oldF_newA, w, label='Old Features + New Algorithm')
plt.bar(x + 1.5*w, newF_newA, w, label='New Features + New Algorithm')
plt.xticks(x, [t.split(' (')[0] for t in perf_targets])
plt.ylabel("Test R2$")
plt.figtext(0.5, 0.01, "Flux recovery new algorithm = HistGradientBoosting (fixed in script)", ha="center", fontsize=10)
plt.legend()
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(f'{FIG_DIR}/fig_ablation_2x2.png', dpi=300)
plt.close()

# -----------------
# 5. fig_leakage_tuned.png
# -----------------
with open('outputs/grouped_cv_tuned_models.json', 'r') as f:
    grp_seed = json.load(f)

grp_means = [grp_seed[t]['grouped_r2_mean'] for t in perf_targets]
grp_stds = [grp_seed[t]['grouped_r2_std'] for t in perf_targets]
rep_means_perf = [rep_seed[t]['r2_mean'] for t in perf_targets]
rep_stds_perf = [rep_seed[t]['r2_std'] for t in perf_targets]

x = np.arange(len(perf_targets))
width = 0.35
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, rep_means_perf, width, yerr=rep_stds_perf, label='Random Split (Repeated)', capsize=5)
plt.bar(x + width/2, grp_means, width, yerr=grp_stds, label='DOI-Grouped 5-Fold CV', capsize=5)
plt.axhline(0, color='black', linewidth=0.8)
plt.xticks(x, [t.split(' (')[0] for t in perf_targets])
plt.ylabel("R2$")
plt.legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig_leakage_tuned.png', dpi=300)
plt.close()

