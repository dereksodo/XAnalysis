import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.DataProcessing.utils import selected_features, Paths
from src.DataProcessing.evaluate import compute_metrics  # y_true / y_pred → metrics

# ---------------------------- CONFIG ---------------------------------
FIGURE_DATA = Paths.FIGURE_DIR
BASE_DIR = os.path.join(FIGURE_DATA, "Chapter5", "predictions")
FEATURE_LIST = selected_features.SF

BETA = np.array([2.0, 2.0, 1.0, 5.0])
THRESH = {"rmse_std": 1., "r2": .6, "mase": 1., "da": .7}

# ---------------------- COLLECT METRICS ------------------------------
records = []
for model in os.listdir(BASE_DIR):
    model_path = os.path.join(BASE_DIR, model)
    if not os.path.isdir(model_path):
        continue
    for feat in FEATURE_LIST:
        pattern = os.path.join(model_path, f"*{feat}*.csv")
        for csv in glob.glob(pattern):
            try:
                df_tmp = pd.read_csv(csv, usecols=["y_true", "y_pred"])
            except Exception:
                continue
            m = compute_metrics(df_tmp["y_true"], df_tmp["y_pred"])
            records.append({
                "model": model,
                "feature": feat,
                "rmse_std": m["rmse/std"],
                "r2": m["r2"],
                "mase": m["mase"],
                "da": m["da"],
            })

if not records:
    raise RuntimeError("No CSV files found – check predictions directory")

df = pd.DataFrame(records)

# -------------------------------------------------------- Guiding Score
a1 = -(df["rmse_std"] - THRESH["rmse_std"]) / THRESH["rmse_std"]
a2 =  (df["r2"]        - THRESH["r2"])       / THRESH["r2"]
a3 = -(df["mase"]      - THRESH["mase"])     / THRESH["mase"]
a4 =  (df["da"]        - THRESH["da"])       / THRESH["da"]

a_mat = np.vstack([a1, a2, a3, a4]).T
guiding_mat = 1 / (1 + np.exp(-a_mat)) * BETA
df["guiding_score"] = guiding_mat.sum(axis=1)
for i, n in enumerate(["rmse_std","r2","mase","da"]):
    df[f"contrib_{n}"] = guiding_mat[:, i]

# -------------------------------------------------------- aggregate & plot
agg = df.groupby("model")[["guiding_score",
                           "contrib_rmse_std","contrib_r2",
                           "contrib_mase","contrib_da"]].mean()
agg = agg.sort_values("guiding_score", ascending=False)

# -------------------------------------------------- 统一绘图 --------------------------------------------------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 6))

# -------- (a) 左：堆叠柱 --------
stack_cols = ["contrib_rmse_std", "contrib_r2", "contrib_mase", "contrib_da"]
labels = ["RMSE/STD", "$R^2$", "MASE", "DA"]
xs = np.arange(len(agg))
bottom = np.zeros(len(agg))
for col, lbl in zip(stack_cols, labels):
    ax0.bar(xs, agg[col], bottom=bottom, width=0.6, label=lbl)
    bottom += agg[col].values
for x, tot in zip(xs, agg["guiding_score"]):
    ax0.text(x, tot + 0.1, f"{tot:.2f}", ha="center", va="bottom")
ax0.axhline(2.0, ls="--", color="g", label="Threshold 2.0")
ax0.set_ylabel("Average Contribution (Guiding Score)")
ax0.set_ylim(0, 7)
ax0.set_xticks(xs);  ax0.set_xticklabels(agg.index)
ax0.set_title("(a) Metric Contributions per Model")
ax0.legend(fontsize="small")

# -------- (b) 右：箱线图 --------
box_data = [df[df["model"] == m]["guiding_score"].values for m in agg.index]
ax1.boxplot(box_data,
            positions=xs, widths=0.6, showfliers=True, patch_artist=True,
            boxprops=dict(facecolor="lightblue", color="blue"),
            medianprops=dict(color="red"))
ax1.axhline(2.0, ls="--", color="g", label="Threshold 2.0")
ax1.set_xticks(xs); ax1.set_xticklabels(agg.index)
ax1.set_ylabel("Guiding Score across 10 Features")
ax1.set_title("(b) Guiding Score Distribution (Boxplot)")
ax1.legend(fontsize="small")

fig.tight_layout()
out = os.path.join(FIGURE_DATA, "Chapter5","guiding_score_combined.png")
fig.savefig(out, dpi=300)
print("Saved combined figure to", out)