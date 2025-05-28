

"""
Plot per-country per-feature best Guiding Score heatmap.
Traverses time series prediction CSVs, computes metrics, selects best score per (country, feature).
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.DataProcessing.utils import Paths, get_scores, selected_features, FeatureNames
from src.DataProcessing.evaluate import compute_metrics

# ----------------------------- CONFIGURATION -----------------------------
# Models to consider
MODEL_LIST = ["Rolling_XGB_lag3", "Rolling_XGB_lag2", "ARIMA", "Naive"]
# Features and their display names (should match your project)
FEATURE_LIST = selected_features.SF
FEATURE_NAMES = [FeatureNames.code2name[c] for c in FEATURE_LIST]
# Metrics thresholds and weights (from thesis)
BETA = [2.0, 2.0, 1.0, 5.0]
THRESHOLDS = {
    "rmse_std": 1.0,
    "r2": 0.6,
    "mase": 1.0,
    "da": 0.7,
}
# Path to prediction CSVs
FIGURE_ROOT = os.path.join(Paths.FIGURE_DIR, "Chapter5","predictions")
HEATMAP_PATH = os.path.join(Paths.FIGURE_DIR, "Chapter5","heat_map.png")
# ----------------------------- MAIN LOGIC -----------------------------
def main():
    # Map: (country_code, feature) -> best guiding score
    best_score = {}
    all_countries = set()
    # Traverse all models and their CSVs
    for model in MODEL_LIST:
        model_dir = os.path.join(FIGURE_ROOT, model)

        # ---- 新读取：逐 feature 深度递归 glob，与 plot_time 一致 ----
        for feature_code in FEATURE_LIST:
            pattern = os.path.join(model_dir, "**", f"*{feature_code}*.csv")
            
            for csv_path in glob.glob(pattern, recursive=True):
                # 判断国家代码：优先文件名前缀，否则取上级文件夹名
                base = os.path.basename(csv_path)
                if "_" in base:
                    country_code = base.split("_", 1)[0]
                else:
                    country_code = os.path.basename(os.path.dirname(csv_path))

                # 粗略过滤：要求国家代码 3 位
                if len(country_code) != 3:
                    continue

                all_countries.add(country_code)
                try:
                    df = pd.read_csv(csv_path, usecols=["y_true", "y_pred"])
                    print(f"processing {csv_path}")
                    guiding_score = get_scores.guiding_score(compute_metrics(df["y_true"], df["y_pred"]))
                    key = (country_code, feature_code)
                    if (key not in best_score) or (guiding_score > best_score[key]):
                        best_score[key] = guiding_score
                except Exception:
                    # 读取失败直接跳过
                    continue
    # Prepare sorted lists
        # ------------------ 构造矩阵并按均值排序 ------------------
    country_list = sorted(all_countries)
    feature_list = FEATURE_LIST

    # 先填一个未排序矩阵
    mat = np.full((len(country_list), len(feature_list)), np.nan)
    for i, country in enumerate(country_list):
        for j, feat in enumerate(feature_list):
            mat[i, j] = best_score.get((country, feat), np.nan)

    # 计算每个国家的平均 Guiding Score（忽略 NaN）
    row_means = np.nanmean(mat, axis=1)
    # 按均值降序排序索引
    sort_idx = np.argsort(row_means)[::-1]

    # 重新排 country_list 和矩阵行
    country_list = [country_list[i] for i in sort_idx]
    mat = mat[sort_idx, :]

    # ----------------------------- PLOTTING -----------------------------
    fig, ax = plt.subplots(figsize=(1.2*len(feature_list), 0.5*len(country_list) + 2))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", interpolation="nearest")
    # X-ticks: feature names
    ax.set_xticks(np.arange(len(feature_list)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=40, ha="right", fontsize=9)
    # Y-ticks: country codes
    ax.set_yticks(np.arange(len(country_list)))
    ax.set_yticklabels(country_list, fontsize=9)
    # Title
    ax.set_title("Per-Country Guiding Score (best model)", fontsize=15, pad=15)
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03)
    cbar.set_label("Best Guiding Score", fontsize=11)
    plt.tight_layout()
    # Save
    os.makedirs(os.path.dirname(HEATMAP_PATH), exist_ok=True)
    plt.savefig(HEATMAP_PATH, dpi=300)
    print(f"Heatmap saved to: {HEATMAP_PATH}")

if __name__ == "__main__":
    main()