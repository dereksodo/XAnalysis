

import pandas as pd
import matplotlib.pyplot as plt
import os

# Load metrics
df = pd.read_csv("/Users/tianhaozhang/Desktop/XCountryOIRPrediction/results/time_series_metrics.csv")

# Filter models of interest
models = ["XGB", "RF", "ARIMA"]
df = df[df["model"].isin(models)]

# Define feasibility function
thresholds = {
    "rmse/std": lambda x: x < 1.0,
    "r2": lambda x: x > 0.6,
    "mase": lambda x: x < 1.0,
    "da": lambda x: x > 0.7
}

def compute_feasibility(row):
    indicators = [
        thresholds["rmse/std"](row["rmse/std"]),
        thresholds["r2"](row["r2"]),
        thresholds["mase"](row["mase"]),
        thresholds["da"](row["da"])
    ]
    return int(sum(indicators) >= 3)

df["feasibility"] = df.apply(compute_feasibility, axis=1)

# Compute mean feasibility by country and model
grouped = df.groupby(["country", "model"])["feasibility"].mean().unstack()

# Plot
fig, ax = plt.subplots(figsize=(14, 6))
grouped.plot(kind="bar", ax=ax)
ax.set_ylabel("Feasibility Rate")
ax.set_title("Feasibility Rate by Country and Model")
ax.axhline(y=0.5, linestyle="--", color="green", linewidth=1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.tight_layout()

# Save
output_path = "/Users/tianhaozhang/Desktop/XCountryOIRPrediction/figures/time_series/figure6.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300)
plt.close()
print(f"Saved figure: {output_path}")