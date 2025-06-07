"""
plot_1.py
------------------------------------------------
生成两张图：
1. 跨国溢出权重矩阵 M 的后验均值热力图
2. 每个指标在原始尺度下的 RMSE 柱状图

运行：
    python -m src.Chapter6.plot_1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
from src.DataProcessing.utils import Paths, get_scores, selected_features, FeatureNames, NamesAndCodes
from src.DataProcessing.evaluate import compute_metrics

# --------------------------------------------------
# 1. 路径
# --------------------------------------------------
PANEL_PATH     = os.path.join(Paths.DATA_DIR, "panel_1990_2020.csv")
INTERPOLATION_PATH = os.path.join(Paths.DATA_DIR, "interpolation_top13.csv")
df_interp = pd.read_csv(INTERPOLATION_PATH)

# --------------------------------------------------
# 2. 基本配置
# --------------------------------------------------
COUNTRIES = NamesAndCodes.country_codes
FEATURES  = selected_features.SF

# --------------------------------------------------
# 3. 加载预测数据
# --------------------------------------------------
PREDICTION_FILES = [
    os.path.join(Paths.FIGURE_DIR,"Chapter6", "bayesian_prediction_identity.csv"),
    os.path.join(Paths.FIGURE_DIR,"Chapter6", "bayesian_prediction_zero.csv"),
    os.path.join(Paths.FIGURE_DIR,"Chapter6", "bayesian_prediction_uniform.csv")
]

# Initialize dictionaries to hold metrics for each model
rmse_std_identity = []
mase_identity = []
r2_identity = []
da_identity = []

rmse_std_zero = []
mase_zero = []
r2_zero = []
da_zero = []

rmse_std_uniform = []
mase_uniform = []
r2_uniform = []
da_uniform = []

# Load naive metrics later
rmse_std_naive = []
mase_naive = []
r2_naive = []
da_naive = []

for idx, pred_file in enumerate(PREDICTION_FILES):
    df = pd.read_csv(pred_file)
    FEATURES = df["feature"].unique().tolist()
    COUNTRIES = df["country"].unique().tolist()
    COUNTRIES.sort()
    FEATURES.sort()

    rmse_std = []
    mase = []
    r2 = []
    da = []

    for feature in FEATURES:
        sub = df[df["feature"] == feature]
        y_true = sub["y_true"].values
        y_pred = sub["y_pred"].values
        y_naive = []
        for _, row in sub.iterrows():
            try:
                value = df_interp[
                    (df_interp["country_code"] == row["country"]) &
                    (df_interp["year"] == row["year"])
                ][row["feature"]].values[0]
                y_naive.append(value)
            except:
                y_naive.append(np.nan)
        y_naive = np.array(y_naive)

        mask = ~np.isnan(y_naive)
        y_true_masked = y_true[mask]
        y_pred_masked = y_pred[mask]
        y_naive_masked = y_naive[mask]

        metrics = compute_metrics(y_true_masked, y_pred_masked, y_naive_masked)
        rmse_std.append(metrics["rmse/std"])
        mase.append(metrics["mase"])
        r2.append(metrics["r2"])
        da.append(metrics["da"])

    if idx == 0:
        rmse_std_identity = rmse_std
        mase_identity = mase
        r2_identity = r2
        da_identity = da
    elif idx == 1:
        rmse_std_zero = rmse_std
        mase_zero = mase
        r2_zero = r2
        da_zero = da
    elif idx == 2:
        rmse_std_uniform = rmse_std
        mase_uniform = mase
        r2_uniform = r2
        da_uniform = da

# naive 方法的指标
naive_metric_dict = {m: [] for m in ["rmse/std", "mase", "r2", "da"]}

for feature_inner in FEATURES:
    metrics_list = {m: [] for m in ["rmse/std", "mase", "r2", "da"]}
    for country in COUNTRIES:
        filename = os.path.join(Paths.FIGURE_DIR, "Chapter5", "predictions", "Naive", f"{country}_{feature_inner}.csv")
        if not os.path.exists(filename):
            continue
        df_naive = pd.read_csv(filename)
        y_true = df_naive["y_true"].values
        y_pred = df_naive["y_pred"].values
        if len(y_true) < 2:
            continue
        metrics = compute_metrics(y_true, y_pred)
        metrics_list["rmse/std"].append(metrics["rmse/std"])
        metrics_list["mase"].append(metrics["mase"])
        metrics_list["r2"].append(metrics["r2"])
        metrics_list["da"].append(metrics["da"])
    for m in ["rmse/std", "mase", "r2", "da"]:
        if metrics_list[m]:
            naive_metric_dict[m].append(np.mean(metrics_list[m]))
        else:
            naive_metric_dict[m].append(np.nan)

rmse_std_naive = naive_metric_dict["rmse/std"]
mase_naive = naive_metric_dict["mase"]
r2_naive = naive_metric_dict["r2"]
da_naive = naive_metric_dict["da"]

# --------------------------------------------------
# 5. 画图
# --------------------------------------------------
metric_names = ["rmse/std", "mase", "r2", "da"]
metric_scores = [
    (rmse_std_identity, rmse_std_zero, rmse_std_uniform, rmse_std_naive),
    (mase_identity, mase_zero, mase_uniform, mase_naive),
    (r2_identity, r2_zero, r2_uniform, r2_naive),
    (da_identity, da_zero, da_uniform, da_naive)
]
# 指标阈值
THRESH = {"rmse/std": 1.0, "r2": 0.6, "mase": 1.0, "da": 0.7}
# 更人类可读的标题
plot_titles = {
    "rmse/std": "Standardised RMSE",
    "r2":       "R²",
    "mase":     "MASE",
    "da":       "Directional Accuracy"
}

plt.rcParams.update({"font.size": 10})
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
x = np.arange(len(FEATURES))
width = 0.6

for idx, (m, scores_tuple) in enumerate(zip(metric_names, metric_scores)):
    r, c = divmod(idx, 2)
    ax = axes[r, c]
    scores_identity, scores_zero, scores_uniform, scores_naive = scores_tuple
    ax.bar(x - width * 1.5 / 3, scores_identity, width=width / 4, label="Bayesian-Identity")
    ax.bar(x - width / 3, scores_zero, width=width / 4, label="Bayesian-Zero")
    ax.bar(x + width / 3, scores_uniform, width=width / 4, label="Bayesian-Uniform")
    ax.bar(x + width * 1.5 / 3, scores_naive, width=width / 4, label="Naive")
    ax.set_title(plot_titles[m])
    ax.set_xticks(x)
    # 将特征名转换为人类可读格式
    readable_labels = [FeatureNames.code2name.get(f, f) for f in FEATURES]
    ax.set_xticklabels(readable_labels, rotation=45)
    ax.set_ylabel(m)
    # 绘制基准线
    thresh = THRESH[m]
    color = "red" if m in ["rmse/std", "mase"] else "green"
    ax.axhline(thresh, linestyle="--", color=color, linewidth=1, label="Threshold")
    ax.legend()

fig.suptitle("Bayesian GVAR Performance Metrics by Feature", fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
FIG_DIR = os.path.join(Paths.FIGURE_DIR, "Chapter6", "metrics_from_csv.png")
fig.savefig(FIG_DIR, dpi=300, bbox_inches="tight")
print(f"[INFO] Figures saved to {FIG_DIR}")