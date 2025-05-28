from src.DataProcessing.utils import Paths
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


periods = ['1960_2020', '2010_2020']
model_names = [
    'LR', 'LWR', 'SVR', 'Ridge', 'Lasso', 'ElasticNet', 'RF', 'KNN', 'XGB'
]

results_dir = os.path.join(Paths.FIGURE_DIR, "Chapter4")

def plot_figure1():
    data = []
    for period in periods:
        for model in model_names:
            result_file = os.path.join(results_dir, f"{period}_{model}_results.csv")
            if not os.path.exists(result_file):
                continue
            df = pd.read_csv(result_file)
            mean_rmse_std = df["rmse/std"].mean()
            mean_r2 = df["r2"].mean()
            mean_mase = df["mase"].mean() if "mase" in df.columns else np.nan
            mean_da = df["da"].mean() if "da" in df.columns else np.nan
            data.append({
                "Period": period.replace("_", "-"),
                "Model": model,
                "Average RMSE/STD": mean_rmse_std,
                "Average R2": mean_r2,
                "Average MASE": mean_mase,
                "Average DA": mean_da
            })

    df_all = pd.DataFrame(data)

    x = np.arange(len(model_names))
    width = 0.35

    # Prepare values for each metric and period
    def get_metric_values(metric, period_str):
        return [
            df_all.loc[(df_all["Period"] == period_str) & (df_all["Model"] == m), metric].values[0]
            if not df_all.loc[(df_all["Period"] == period_str) & (df_all["Model"] == m), metric].empty
            else np.nan
            for m in model_names
        ]

    # Create 2x2 subplot for 4 metrics: RMSE/STD, R2, MASE, DA
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    subplot_titles = [
        "Average RMSE/STD for Each Model",
        "Average $R^2$ for Each Model",
        "Average MASE for Each Model",
        "Average DA for Each Model"
    ]
    ylabels = [
        "Average RMSE/STD",
        "Average $R^2$",
        "Average MASE",
        "Average DA"
    ]
    thresholds = [
        (1.0, 'r', 'Feasible region threshold'),    # RMSE/STD
        (0.6, 'g', 'Feasible region threshold'),    # R2
        (1.0, 'r', 'Feasible region threshold'), # MASE
        (0.7, 'g', 'Feasible region threshold')  # DA
    ]
    metrics = [
        "Average RMSE/STD",
        "Average R2",
        "Average MASE",
        "Average DA"
    ]

    for idx, ax in enumerate(axes.flat):
        metric = metrics[idx]
        vals_1960_2020 = get_metric_values(metric, '1960-2020')
        vals_2010_2020 = get_metric_values(metric, '2010-2020')
        bar1 = ax.bar(
            x - width / 2, vals_1960_2020, width, label='1960-2020'
        )
        bar2 = ax.bar(
            x + width / 2, vals_2010_2020, width, label='2010-2020'
        )
        thresh, color, thlabel = thresholds[idx]
        ax.axhline(thresh, color=color, linestyle='--', label=thlabel)
        ax.set_ylabel(ylabels[idx])
        ax.set_title(subplot_titles[idx])
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=0, fontsize=10)
        ax.legend()

    fig.tight_layout()
    output_path = os.path.join(results_dir, "Combined_Metrics_figure1.png")
    plt.savefig(output_path, dpi=200)
    print("Saved figure to", output_path)


if __name__ == "__main__":
    plot_figure1()