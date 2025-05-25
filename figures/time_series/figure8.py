


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load time series results
df = pd.read_csv("/Users/tianhaozhang/Desktop/XCountryOIRPrediction/results/time_series_metrics.csv")

# Define feasibility thresholds
thresholds = {
    "rmse/std": lambda x: x < 1.0,
    "r2": lambda x: x > 0.6,
    "mase": lambda x: x < 1.0,
    "da": lambda x: x > 0.7
}

# Compute feasibility score (0 to 4)
def compute_feasibility_score(row):
    checks = [
        thresholds["rmse/std"](row["rmse/std"]),
        thresholds["r2"](row["r2"]),
        thresholds["mase"](row["mase"]),
        thresholds["da"](row["da"]),
    ]
    return sum(checks)

df["feasibility_score"] = df.apply(compute_feasibility_score, axis=1)

# Keep best-performing model per (country, target)
idx = df.groupby(["country", "target"])["feasibility_score"].idxmax()
df_best = df.loc[idx]

# Pivot to matrix: countries x indicators, sorted by average feasibility
heatmap_data = df_best.pivot(index="country", columns="target", values="feasibility_score").fillna(0)
# Sort countries (rows) by predictability (descending)
heatmap_data = heatmap_data.loc[heatmap_data.mean(axis=1).sort_values(ascending=False).index]
# Sort indicators (columns) by predictability (ascending)
heatmap_data = heatmap_data[heatmap_data.mean(axis=0).sort_values(ascending=True).index]

# Plot heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".0f",
    cmap="Blues",
    cbar_kws={"label": "Feasibility Score (0–4)"},
    xticklabels=True,
    yticklabels=True
)
plt.xticks(rotation=45)
plt.title("Heatmap of Per-Indicator Feasibility Scores by Country (Best Model per Cell)")
plt.xlabel("Indicator")
plt.ylabel("Country")

# Save figure
output_path = "/Users/tianhaozhang/Desktop/XCountryOIRPrediction/figures/time_series/figure8.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()
print(f"Saved heatmap to: {output_path}")