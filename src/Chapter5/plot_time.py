import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.DataProcessing.evaluate import compute_metrics 

from src.DataProcessing.utils import selected_features, FeatureNames, Paths

# --------------------------- CONFIG -----------------------------------------
FIGURE_DATA = Paths.FIGURE_DIR
BASE_DIR = os.path.join(FIGURE_DATA, "Chapter5", "predictions")
FEATURE_LIST = selected_features.SF  # 10 个指标代码
FEATURE_NAMES = [FeatureNames.code2name[c] for c in FEATURE_LIST]

# 6 个模型目录（自动扫描）
MODEL_DIRS = ['Rolling_XGB_lag3', 'Rolling_XGB_lag2', 'Naive', 'ARIMA']
print("Model folders detected:", MODEL_DIRS)

# --------------------------- METRICS CONFIG ---------------------------------
METRICS = [
    ("RMSE/STD", 1, "lower"),
    ("R2", 0.6, "higher"),
    ("MASE", 1, "lower"),
    ("DA", 0.7, "higher"),
]
metrics_values = {m[0]: {model: [np.nan] * len(FEATURE_LIST) for model in MODEL_DIRS} for m in METRICS}


for model in MODEL_DIRS:
    print(f"\n[INFO] Processing model: {model}")
    model_path = os.path.join(BASE_DIR, model)
    for f_idx, feature in enumerate(FEATURE_LIST):
        csv_files = glob(os.path.join(model_path, f"*{feature}*.csv"))
        df_list = []
        for csv in csv_files:
            try:
                tmp = pd.read_csv(csv, usecols=["country_code", "year", "y_true", "y_pred"])
                df_list.append(tmp)
            except Exception as e:
                print(f"[WARN] Could not read {csv}: {e}")
        if not df_list:
            continue
        big_df = pd.concat(df_list, ignore_index=True)
        metric_dict = compute_metrics(big_df["y_true"].values, big_df["y_pred"].values)

        rmse_std = metric_dict["rmse/std"]
        r2       = metric_dict["r2"]
        mase     = metric_dict["mase"]
        da       = metric_dict["da"]
        for metric_name, _, _ in METRICS:
            val = locals()[metric_name.lower().replace("/", "_") if metric_name != "R2" else "r2"]
            metrics_values[metric_name][model][f_idx] = val
    print(f"[INFO] Completed model: {model}")

# --------------------------- PLOTTING ---------------------------------------

plt.figure(figsize=(16, 10))
for idx, (metric, threshold, _) in enumerate(METRICS, 1):
    plt.subplot(2, 2, idx)
    for model in MODEL_DIRS:
        xs = range(len(FEATURE_LIST))
        plt.plot(xs, metrics_values[metric][model], marker="o", label=model)
    threshold_handle = plt.axhline(
        y=threshold,
        color="r" if _ == "lower" else "g",
        linestyle="--",
        label=f"Threshold ={threshold}"
    )
    plt.title(metric)
    plt.xlabel("Feature")
    plt.ylabel(metric)
    plt.xticks(ticks=range(len(FEATURE_LIST)), labels=FEATURE_NAMES, rotation=30)
    if idx == 1:
        # full legend: models + threshold
        plt.legend()
    else:
        # show only threshold legend entry
        plt.legend(handles=[threshold_handle], labels=[f"Threshold = {threshold}"])
    plt.grid(True)

plt.tight_layout()
plt.suptitle("Model Comparison on Each Metric (Avg over All Countries & Years)", fontsize=18, y=1.04)
output_path = os.path.join(BASE_DIR, "featurewise_model_comparison.png")
plt.savefig(output_path, bbox_inches="tight", dpi=300)
print(f"Figure saved to {output_path}")
#plt.show()