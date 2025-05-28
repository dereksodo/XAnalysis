

import pandas as pd
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv("/Users/tianhaozhang/Desktop/XCountryOIRPrediction/results/time_series_metrics.csv")
target_models = ["XGB", "RF", "ARIMA"]
df = df[df["model"].isin(target_models)]

# Define plot setup
metrics = ['rmse/std', 'r2', 'mase', 'da']
titles = ['RMSE / STD', 'R² Score', 'MASE', 'Directional Accuracy']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    for model in target_models:
        values = df[df["model"] == model].groupby("target")[metric].mean()
        ax.plot(values.index, values.values, marker='o', label=model)

    ax.set_title(titles[idx])
    ax.set_xlabel("Indicator")
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)

    # Add feasible region thresholds
    if metric == 'rmse/std':
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1)
    elif metric == 'r2':
        ax.axhline(y=0.6, color='green', linestyle='--', linewidth=1)
    elif metric == 'mase':
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1)
    elif metric == 'da':
        ax.axhline(y=0.7, color='green', linestyle='--', linewidth=1)

    ax.legend()

plt.tight_layout()
out_path = "/Users/tianhaozhang/Desktop/XCountryOIRPrediction/figures/time_series/figure4.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi = 300)
plt.close()
print(f"Saved: {out_path}")