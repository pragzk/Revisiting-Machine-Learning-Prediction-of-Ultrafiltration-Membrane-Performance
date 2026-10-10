import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

OUT_DIR = Path('outputs')
FIG_DIR = OUT_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)

def plot_baseline_heatmap():
    if not (OUT_DIR / '02_baseline_summary_by_model.csv').exists():
        print("Skipping heatmap: baseline CSV not found.")
        return
    df1 = pd.read_csv(OUT_DIR / '02_baseline_summary_by_model.csv')
    df2 = pd.read_csv(OUT_DIR / '09_baseline_summary_properties.csv')
    df = pd.concat([df1, df2])
    pivot = df.pivot(index='model', columns='target', values='r2_mean')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, cmap='RdYlGn', fmt=".2f", vmin=-0.5, vmax=1.0)
    plt.title("Untuned Baseline $R^2$ — 16 Models x 8 Targets")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_baseline_heatmap.png', dpi=300)
    plt.close()
    print("Generated fig_baseline_heatmap.png")

def plot_leakage_comparison():
    if not (OUT_DIR / '02_baseline_summary_by_model.csv').exists():
        return
    df1 = pd.read_csv(OUT_DIR / '02_baseline_summary_by_model.csv')
    best_idx = df1.groupby('target')['grouped_r2_mean'].idxmax()
    best_df = df1.loc[best_idx].copy()
    
    labels = []
    random_scores = []
    grouped_scores = []
    
    for _, row in best_df.iterrows():
        labels.append(f"{row['target'].split(' (')[0]}\n({row['model']})")
        random_scores.append(row['r2_mean'])
        grouped_scores.append(row['grouped_r2_mean'])
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, random_scores, width, label='Random 80/20 split (mean of seeds)', color='#4C72B0')
    plt.bar(x + width/2, grouped_scores, width, label='DOI-grouped 5-fold CV', color='#C44E52')
    
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel("$R^2$")
    plt.title("Random-Split vs. DOI-Grouped $R^2$: Literature-Source Leakage Check")
    plt.xticks(x, labels, rotation=0, fontsize=9)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_leakage_comparison.png', dpi=300)
    plt.close()
    print("Generated fig_leakage_comparison.png")

def plot_overfitting_gap():
    json_path = OUT_DIR / '06_train_test_batch1.json'
    if not json_path.exists():
        return
    with open(json_path, 'r') as f:
        best_models = json.load(f)
        
    targets = []
    train_r2 = []
    test_r2 = []
    for t, info in best_models.items():
        targets.append(t.split(' (')[0])
        train_r2.append(info['train_r2'])
        test_r2.append(info['test_r2'])
        
    x = np.arange(len(targets))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, train_r2, width, label='Train $R^2$', color='#4C72B0')
    plt.bar(x + width/2, test_r2, width, label='Test $R^2$', color='#C44E52')
    
    plt.ylabel("$R^2$")
    plt.title("Train vs. Test $R^2$: Overfitting Gap (Tuned Models, seed=1000)")
    plt.xticks(x, targets, rotation=25, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_overfitting_gap.png', dpi=300)
    plt.close()
    print("Generated fig_overfitting_gap.png")

def plot_seed_variance():
    path1 = OUT_DIR / '02_baseline_repeated_ALL.csv'
    path2 = OUT_DIR / '09_baseline_repeated_properties_ALL.csv'
    if not (path1.exists() and path2.exists()):
        return
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df = pd.concat([df1, df2])
    
    mean_r2 = df.groupby(['target', 'model'])['r2'].mean().reset_index()
    best_idx = mean_r2.groupby('target')['r2'].idxmax()
    best_models_per_target = mean_r2.loc[best_idx]
    
    perf_targets = [t for t in df1['target'].unique()]
    prop_targets = [t for t in df2['target'].unique()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    data_perf = []
    labels_perf = []
    for t in perf_targets:
        m = best_models_per_target[best_models_per_target['target'] == t]['model'].iloc[0]
        d = df[(df['target'] == t) & (df['model'] == m)]['r2'].dropna()
        data_perf.append(d)
        labels_perf.append(f"{t.split(' (')[0]}\n({m})")
    
    ax1.boxplot(data_perf)
    ax1.set_xticklabels(labels_perf, rotation=15, fontsize=8)
    ax1.set_ylabel("$R^2$ (30 seeds)")
    ax1.set_title("Performance Targets")
    
    data_prop = []
    labels_prop = []
    for t in prop_targets:
        m = best_models_per_target[best_models_per_target['target'] == t]['model'].iloc[0]
        d = df[(df['target'] == t) & (df['model'] == m)]['r2'].dropna()
        data_prop.append(d)
        labels_prop.append(f"{t.split(' (')[0]}\n({m})")
        
    ax2.boxplot(data_prop)
    ax2.set_xticklabels(labels_prop, rotation=15, fontsize=8)
    ax2.set_ylabel("$R^2$ (15 seeds)")
    ax2.set_title("Property Targets")
    
    plt.suptitle("Seed-to-Seed Variance in Test $R^2$ (Untuned Baseline)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_seed_variance.png', dpi=300)
    plt.close()
    print("Generated fig_seed_variance.png")

def plot_resolved_discrepancy(target, out_name, old_tuned_r2, old_tuned_name, ensemble_r2=None):
    csv_path = OUT_DIR / '09_baseline_summary_properties.csv'
    if not csv_path.exists():
        return
    df2 = pd.read_csv(csv_path)
    untuned_row = df2[(df2['target'] == target) & (df2['model'] == 'GradientBoosting')]
    untuned_r2 = untuned_row['r2_mean'].iloc[0]
    untuned_std = untuned_row['r2_std'].iloc[0]
    
    with open(OUT_DIR / '06_train_test_batch1.json', 'r') as f:
        best_models = json.load(f)
    
    tuned_info = best_models[target]
    tuned_model = tuned_info['model']
    tuned_r2 = tuned_info['test_r2']
    
    labels = [
        'Untuned\nGradientBoosting\n(15-seed mean)',
        f'Old Tuned\n{old_tuned_name}\n(previous best)',
        f'Tuned\n{tuned_model}\n(seed=1000)',
    ]
    scores = [untuned_r2, old_tuned_r2, tuned_r2]
    errors = [untuned_std, 0, 0]
    colors = ['#DD8452', '#8C8C8C', '#55A868']
    
    if ensemble_r2 is not None:
        labels.append('Tuned\nEnsemble\n(seed=1000)')
        scores.append(ensemble_r2)
        errors.append(0)
        colors.append('#4C72B0')
        
    plt.figure(figsize=(9, 6))
    bars = plt.bar(labels, scores, color=colors, yerr=errors, capsize=10)
                   
    for i, score in enumerate(scores):
        offset = errors[i] if errors[i] > 0 else 0
        plt.text(i, score + offset + 0.02, f"{score:.3f}", ha='center', va='bottom', fontweight='bold')
    
    plt.ylim(0, 1.0)
    plt.ylabel(f"$R^2$ ({target.split(' (')[0].lower()})")
    plt.title(f"Flagged Discrepancy Resolved: {target.split(' (')[0].title()}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / out_name, dpi=300)
    plt.close()
    print(f"Generated {out_name}")

if __name__ == "__main__":
    plot_baseline_heatmap()
    plot_leakage_comparison()
    plot_overfitting_gap()
    plot_seed_variance()
    plot_resolved_discrepancy('contact angle (degree)', 'fig_contact_angle_resolved.png', 0.728, 'Multi-task ExtraTrees', 0.806)
    plot_resolved_discrepancy('overall porosity (%)', 'fig_porosity_resolved.png', 0.578, 'Multi-task ExtraTrees', 0.697)
    print("All paper figures generated successfully.")
