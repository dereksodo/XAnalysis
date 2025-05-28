import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from src.DataProcessing.utils import Paths
results_dir = os.path.join(Paths.FIGURE_DIR, "Chapter4")

def plot_average_guiding_scores():
    periods = ['1960_2020', '2010_2020']
    model_names = [
        'LR', 'LWR', 'SVR', 'Ridge', 'Lasso', 'ElasticNet', 'RF', 'KNN', 'XGB'
    ]

    data = []
    from src.DataProcessing.utils import get_scores
    def compute_guiding_score_row(row):
        metrics = {
            "rmse/std": row["rmse/std"],
            "r2": row["r2"],
            "mase": row["mase"],
            "da": row["da"]
        }
        return get_scores.guiding_score(metrics)
    for period in periods:
        for model in model_names:
            result_file = os.path.join(results_dir, f"{period}_{model}_results.csv")
            if not os.path.exists(result_file):
                continue
            #print(f"Processing {result_file}")
            df = pd.read_csv(result_file)
            # Compute guiding_score if missing, then mean
            if "guiding_score" not in df.columns:
                df["guiding_score"] = df.apply(compute_guiding_score_row, axis=1)
            mean_guiding_score = df["guiding_score"].mean()
            data.append({
                "Period": period,
                "Model": model,
                "Average Guiding Score": mean_guiding_score
            })

    df_all = pd.DataFrame(data)

    # Debug print statement as requested
    print(df_all)

    # 只画一张Average Guiding Score（两组柱状，分period），x轴模型名，柱分period
    x = np.arange(len(model_names))  # 模型序号
    width = 0.35  # 柱宽

    # --- Average Guiding Score plot ---
    # Use explicit bar plotting as requested
    df_1960 = df_all[df_all["Period"] == "1960_2020"].set_index("Model")
    df_2010 = df_all[df_all["Period"] == "2010_2020"].set_index("Model")

    y1 = [df_1960.loc[m, "Average Guiding Score"] if m in df_1960.index else np.nan for m in model_names]
    y2 = [df_2010.loc[m, "Average Guiding Score"] if m in df_2010.index else np.nan for m in model_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bar1 = ax.bar(x - width/2, y1, width, label='1960-2020')
    bar2 = ax.bar(x + width/2, y2, width, label='2010-2020')
    ax.axhline(2.0, color='g', linestyle='--', label='Guiding Score = 2.0')
    ax.set_ylim(0, 3.0)
    ax.set_ylabel("Average Guiding Score")
    ax.set_title("Average Guiding Score for Each Model")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=0, fontsize=10)
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(results_dir, "Average_GuidingScore_figure2.png"), dpi=200)
    #plt.show()


if __name__ == "__main__":
    plot_average_guiding_scores()