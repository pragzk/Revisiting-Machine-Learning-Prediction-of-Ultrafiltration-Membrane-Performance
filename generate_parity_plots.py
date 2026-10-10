import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

OUT_DIR = Path('outputs')
FIG_DIR = OUT_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)

# Load best models
with open(OUT_DIR / '06_train_test_batch1.json', 'r') as f:
    best_models = json.load(f)

for target, info in best_models.items():
    model_name = info['model']
    test_r2 = info['test_r2']
    
    clean_target = target.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
    csv_path = OUT_DIR / f"predictions_{clean_target}_{model_name}.csv"
    
    if not csv_path.exists():
        print(f"Skipping {target}: {csv_path} not found")
        continue
        
    df = pd.read_csv(csv_path)
    
    # Plot
    plt.figure(figsize=(6, 6))
    
    # Train
    train_df = df[df['split'] == 'train']
    plt.scatter(train_df['y_true'], train_df['y_pred'], alpha=0.5, label='Train', color='blue', s=20)
    
    # Test
    test_df = df[df['split'] == 'test']
    plt.scatter(test_df['y_true'], test_df['y_pred'], alpha=0.7, label='Test', color='orange', s=20)
    
    # y=x line
    min_val = min(df['y_true'].min(), df['y_pred'].min())
    max_val = max(df['y_true'].max(), df['y_pred'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
    
    plt.title(f"{target}\nBest Model: {model_name} (Test $R^2$: {test_r2:.3f})")
    plt.xlabel(f"Actual {target}")
    plt.ylabel(f"Predicted {target}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fig_path = FIG_DIR / f"parity_{clean_target}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved {fig_path}")

print("All plots generated successfully.")
