import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = 'outputs/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------
# 1. fig_overfitting_gap.png
# -----------------
df_metrics = pd.read_csv('outputs/figures/seed1000_exact_metrics.csv')
x = np.arange(len(df_metrics))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, df_metrics['train_R2'], width, label='Train R2', color='#4C72B0')
plt.bar(x + width/2, df_metrics['test_R2'], width, label='Test R2', color='#C44E52')

plt.ylabel("R2$")
# No title inside the PNG
plt.xticks(x, df_metrics['target'], rotation=25, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig_overfitting_gap.png', dpi=300)
plt.close()

# -----------------
# 2. parity_<target>.png x8
# -----------------
units = {
    'water permeability (LMH/bar)': 'LMH/bar',
    'organic compound removal (%)': '%',
    'flux decline ratio (%)': '%',
    'flux recovery ratio (%)': '%',
    'reversible fouling ratio (%)': '%',
    'overall porosity (%)': '%',
    'mean pore radius (nm)': 'nm',
    'contact angle (degree)': 'degree'
}
for idx, row in df_metrics.iterrows():
    t = row['target']
    m = row['model']
    test_r2 = row['test_R2']
    
    # Exact names: parity_water_permeability.png, parity_organic_compound_removal.png...
    safe_t2 = t.split(' (')[0].replace(' ', '_')
    
    fname = f'outputs/predictions_{t.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")}_{m}.csv'
    if not os.path.exists(fname):
        print(f"MISSING: {fname}")
        continue
    df_p = pd.read_csv(fname)
    
    train_mask = df_p['split'] == 'train'
    test_mask = df_p['split'] == 'test'
    
    plt.figure(figsize=(6, 6))
    plt.scatter(df_p.loc[train_mask, 'y_true'], df_p.loc[train_mask, 'y_pred'], alpha=0.5, label='Train', color='blue')
    plt.scatter(df_p.loc[test_mask, 'y_true'], df_p.loc[test_mask, 'y_pred'], alpha=0.5, label='Test', color='orange')
    
    min_val = min(df_p['y_true'].min(), df_p['y_pred'].min())
    max_val = max(df_p['y_true'].max(), df_p['y_pred'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    unit_str = units[t]
    plt.xlabel(f"True {t.split(' (')[0]} ({unit_str})")
    plt.ylabel(f"Predicted {t.split(' (')[0]} ({unit_str})")
    
    plt.title(f"{m} (Test R2$ = {test_r2:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/parity_{safe_t2}.png', dpi=300)
    plt.close()

# -----------------
# 3. fig_repeated_seed_vs_gao.png
# -----------------
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
rep_means = [rep_seed[t]['r2_mean'] for t in targets]
rep_stds = [rep_seed[t]['r2_std'] for t in targets]
seed1000_r2 = []
for t in targets:
    r = df_metrics[df_metrics['target'] == t]['test_R2'].values[0]
    seed1000_r2.append(r)
gao = [gao_vals[t] for t in targets]

y = np.arange(len(targets))
plt.figure(figsize=(10, 8))
plt.errorbar(rep_means, y, xerr=rep_stds, fmt='o', color='#4C72B0', label='Repeated-seed mean ± 1 SD')
plt.scatter(seed1000_r2, y, color='red', marker='x', s=100, label='Seed 1000 Test R2')
plt.scatter(gao, y, color='black', marker='D', s=80, label='Gao et al. Baseline')
plt.yticks(y, targets)
plt.xlabel("Test R2$")
plt.legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig_repeated_seed_vs_gao.png', dpi=300)
plt.close()

# -----------------
# 4. fig_ablation_2x2.png
# -----------------
with open('outputs/ablation_features_vs_algorithm.json', 'r') as f:
    abl = json.load(f)

perf_targets = [t for t in targets if t in abl]
oldF_oldA = [abl[t]['old_features__old_algo_xgboost'] for t in perf_targets]
newF_oldA = [abl[t]['new_features__old_algo_xgboost'] for t in perf_targets]
oldF_newA = [abl[t]['old_features__new_algo'] for t in perf_targets]
newF_newA = [abl[t]['new_features__new_algo'] for t in perf_targets]

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

# -----------------
# 6. fig_seed_variance_tuned.png
# -----------------
# Check if repeated_seed_tuned_models.json has per-seed test R2.
if 'per_seed_r2' in rep_seed[targets[0]]:
    print("Found per_seed_r2")
else:
    print("fig_seed_variance_tuned.png skipped: No per_seed_r2 found in repeated_seed_tuned_models.json (only mean/std).")
