"""
Plot best Guiding Score by feature (aggregate 1960-2020).

x-axis: 10 indicator codes
y-axis: best Guiding Score over Naive / ARIMA / Rolling-XGB-lag2 / Rolling-XGB-lag3
One poly-line per country in COUNTRY_GRP.
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

from src.DataProcessing.utils import Paths, get_scores, selected_features, FeatureNames
from src.DataProcessing.evaluate import compute_metrics

# ------------------------------------------------------------------ #
MODEL_LIST   = ["Rolling_XGB_lag3", "Rolling_XGB_lag2", "ARIMA", "Naive"]
FEATURE_LIST = selected_features.SF
COUNTRY_GRP  = ["BRA", "RUS", "IND", "CHN", "ZAF"]          # 修改这里即可换国家
# Two study windows: full history vs recent decade
PERIODS = [("1960-2020", 1960, 2020), ("2010-2020", 2010, 2020)]
PRED_ROOT = os.path.join(Paths.FIGURE_DIR, "Chapter5", "predictions")
FIG_PATH  = os.path.join(Paths.FIGURE_DIR, "Chapter5", "comparison_feature_BRICS.png")
# ------------------------------------------------------------------ #

def collect_feature_scores(start_year: int, end_year: int):
    """
    返回 best_scores[country][feature] 为指定时间段内各模型最大 Guiding Score。
    """
    best_scores = {c: {f: np.nan for f in FEATURE_LIST} for c in COUNTRY_GRP}

    for model in MODEL_LIST:
        mdir = os.path.join(PRED_ROOT, model)
        for feat in FEATURE_LIST:
            for csv in glob.glob(os.path.join(mdir, f"*_{feat}.csv")):
                country = os.path.basename(csv).split("_")[0]
                if country not in COUNTRY_GRP:
                    continue

                df = pd.read_csv(csv, usecols=["year", "y_true", "y_pred"])
                df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

                score = get_scores.guiding_score(
                    compute_metrics(df["y_true"].values, df["y_pred"].values)
                )
                if np.isnan(best_scores[country][feat]) or score > best_scores[country][feat]:
                    best_scores[country][feat] = score
    return best_scores

def main():
    period_scores = {lbl: collect_feature_scores(lo, hi) for lbl, lo, hi in PERIODS}
    x = np.arange(len(FEATURE_LIST))

    fig, ax = plt.subplots(figsize=(1.3 * len(FEATURE_LIST), 6))

    # 颜色：每国一个固定颜色
    base_cmap = cm.get_cmap("tab10", len(COUNTRY_GRP))

    for i, country in enumerate(COUNTRY_GRP):
        color = base_cmap(i)
        for lbl, _lo, _hi in PERIODS:
            alpha = 1.0 if lbl == "1960-2020" else 0.2
            y = [period_scores[lbl][country][f] for f in FEATURE_LIST]
            ax.plot(x, y, "-", marker="o", color=color, alpha=alpha)

    # ---------- 自定义双图例 ----------
    # Legend 1: 国家颜色
    country_handles = [Line2D([0], [0], color=base_cmap(i), lw=2)
                       for i in range(len(COUNTRY_GRP))]
    legend1 = ax.legend(country_handles, COUNTRY_GRP,
                        title="Country", loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.add_artist(legend1)

    # Legend 2: 透明度代表时间段
    period_handles = [Line2D([0], [0], color="black", lw=2, alpha=1.0),
                      Line2D([0], [0], color="black", lw=2, alpha=0.2)]
    period_labels  = ["1960–2020", "2010–2020"]
    ax.legend(period_handles, period_labels,
              title="Period", loc="lower left", bbox_to_anchor=(1.02, 0))

    # ---------- 轴/网格/保存 ----------
    ax.set_xticks(x)
    ax.set_xticklabels((FeatureNames.code2name[i] for i in FEATURE_LIST), rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Best Guiding Score")
    ax.set_xlabel("Features")
    ax.set_title("Best Guiding Score by Feature: 1960–2020 vs 2010–2020")
    ax.grid(True, linestyle="--", alpha=0.6)

    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=300)
    print(f"Feature-comparison plot saved to: {FIG_PATH}")

if __name__ == "__main__":
    main()