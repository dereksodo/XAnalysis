


import pandas as pd
import matplotlib.pyplot as plt
import os

# Load and filter
df = pd.read_csv("/Users/tianhaozhang/Desktop/XCountryOIRPrediction/results/time_series_metrics.csv")
models = ["XGB", "RF", "ARIMA"]
df = df[df["model"].isin(models)]

# Define metric thresholds
thresholds = {
    "rmse/std": 1.0,
    "r2": 0.6,
    "mase": 1.0,
    "da": 0.7
}

# Function to compute pass/fail per metric
def get_pass_rate(df, metric):
    # For metrics where lower is better (rmse/std, mase), success = value < threshold
    # For metrics where higher is better (r2, da), success = value > threshold
    if "rmse" in metric or "mase" in metric:
        passed = df.groupby(["country", "model"])[metric].apply(lambda x: (x < thresholds[metric]).mean())
    else:
        passed = df.groupby(["country", "model"])[metric].apply(lambda x: (x > thresholds[metric]).mean())
    return passed.unstack()

# Setup plot
metrics = ['rmse/std', 'r2', 'mase', 'da']
titles = ['RMSE / STD', 'R² Score', 'MASE', 'Directional Accuracy']
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    pass_rate = get_pass_rate(df, metric)
    pass_rate.plot(kind='bar', ax=ax)
    ax.set_title(titles[idx])
    ax.set_ylabel("Success Rate")
    # Add feasible region thresholds
    if metric == 'rmse/std':
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1)
    elif metric == 'r2':
        ax.axhline(y=0.6, color='green', linestyle='--', linewidth=1)
    elif metric == 'mase':
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1)
    elif metric == 'da':
        ax.axhline(y=0.7, color='green', linestyle='--', linewidth=1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.tight_layout()
output_path = "/Users/tianhaozhang/Desktop/XCountryOIRPrediction/figures/time_series/figure7.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300)
plt.close()
print(f"Saved figure: {output_path}")