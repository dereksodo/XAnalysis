def plot_tuning_metrics(df_res, start_year, end_year, model_name, save_dir):
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from src.DataProcessing.utils import FeatureNames

    # Prepare data
    labels = df_res["target"].map(FeatureNames.code2name)
    if "guiding_metric" in df_res.columns and "guiding_metric_baseline" in df_res.columns:
        guiding_metric = 100 * (df_res["guiding_metric"] - df_res["guiding_metric_baseline"]) / df_res["guiding_metric_baseline"]
        baseline_available = True
    else:
        guiding_metric = df_res[["rmse/std", "r2", "mase", "da"]].mean(axis=1)
        baseline_available = False
    print(f"baseline = {baseline_available}")
    # --- Compute improvements for the four metrics as percentage improvement over baseline ---
    metrics = ["rmse/std", "r2", "mase", "da"]
    metric_names = ["RMSE/STD", "R²", "MASE", "DA"]
    colors = ["#e41a1c", "#4daf4a", "#984ea3", "#ff7f00"]
    # Ensure baseline columns exist for each metric
    for m in metrics:
        if f"{m}_baseline" not in df_res.columns:
            # Need df_base in scope; assume it's available (if not, pass as argument)
            baseline_vals = df_base.set_index("target")[m]
            df_res[f"{m}_baseline"] = df_res["target"].map(baseline_vals)
    # Calculate percentage improvements
    improvements = []
    for m in metrics:
        # Percentage improvement: 100 * (mean(tuned) - mean(baseline)) / abs(mean(baseline))
        improvement = 100 * abs(df_res[m].mean() - df_res[f"{m}_baseline"].mean()) / abs(df_res[f"{m}_baseline"].mean())
        improvements.append(improvement)
    # --- Determine y-axis limits for both subplots ---
    left_max = guiding_metric.max()
    right_max = max(improvements)
    max_ylim = max(left_max, right_max) * 1.2
    if max_ylim == 0:
        max_ylim = 1.0

    # --- Create single figure with two subplots ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left subplot: Guiding metric comparison per feature (with fill and enhanced ticks/limits/legend) ---
    if baseline_available:
        x = np.arange(len(labels))
        baseline_vals = df_res["guiding_metric_baseline"].values
        tuned_vals = df_res["guiding_metric"].values
        axes[0].plot(x, baseline_vals, 'g--o', label='Baseline (untuned)', alpha=0.8)
        axes[0].plot(x, tuned_vals, 'b--o', label='Tuned', alpha=0.8)
        for i in range(len(x)):
            axes[0].fill_between([x[i], x[i]], baseline_vals[i], tuned_vals[i],
                                 color='darkred', alpha=0.4)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=40, ha='right')
        axes[0].set_ylim(min(baseline_vals.min(), tuned_vals.min()) * 0.95,
                         max(baseline_vals.max(), tuned_vals.max()) * 1.05)
        axes[0].legend(loc="best", title="Guiding Metric")
    else:
        x = np.arange(len(labels))
        axes[0].plot(x, guiding_metric, 'b-', label='Guiding Metric')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=40, ha='right')
        axes[0].set_ylim(guiding_metric.min() * 0.95, guiding_metric.max() * 1.05)
        axes[0].legend(loc="best")
    axes[0].set_xlabel("Features")
    axes[0].set_ylabel("Guiding Metric Score")
    axes[0].set_title(f"{start_year}-{end_year} {model_name}\nGuiding Metric Comparison per Feature")

    # --- Right subplot: Overall metric improvement for RMSE/STD, R², MASE, DA ---
    bars = axes[1].bar(metric_names, improvements, color=colors)
    axes[1].set_xlabel("Evaluation Metric")
    axes[1].set_ylabel("Average Improvement (|% change|)")
    axes[1].set_title(f"{start_year}-{end_year} {model_name}\nAverage Absolute Improvement (%) by Metric")
    axes[1].set_ylim(0, 100)
    # Add value labels above each bar, color-matched
    for i, (bar, v, color) in enumerate(zip(bars, improvements, colors)):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}", ha='center', va='bottom',
                     fontsize=10, color=color, fontweight='bold')
    # Add color-coded legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=colors[i], label=metric_names[i]) for i in range(len(metric_names))]
    axes[1].legend(handles=legend_handles, title="Metrics", loc="best")

    plt.tight_layout()
    fig.align_ylabels(axes)

    fname = f"{save_dir}/{start_year}_{end_year}_{model_name}_metrics.png"
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {fname}")


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from src.DataProcessing.utils import Paths, get_scores

    model_names = ['XGB', 'RF']
    year_ranges = [(1960, 2020), (2010, 2020)]

    for start_year, end_year in year_ranges:
        for model_name in model_names:
            result_path = f"{Paths.FIGURE_DIR}/Chapter4tuning/{start_year}_{end_year}_{model_name}_tuning_result.csv"
            save_dir = f"{Paths.FIGURE_DIR}/Chapter4tuning"
            df_res = pd.read_csv(result_path)

            # --- Load baseline file and calculate guiding_metric_baseline ---
            baseline_path = f"{Paths.FIGURE_DIR}/Chapter4/{start_year}_{end_year}_{model_name}_results.csv"
            df_base = pd.read_csv(baseline_path)
            df_base["guiding_metric_baseline"] = df_base.apply(get_scores.guiding_score, axis=1)
            # Merge baseline guiding metric with results
            df_res = df_res.merge(df_base[["target", "guiding_metric_baseline"]], on="target", how="left")
            df_res["guiding_metric"] = df_res.apply(get_scores.guiding_score, axis=1)
            print("NaNs in guiding_metric_baseline:", df_res["guiding_metric_baseline"].isna().sum())
            print("Total rows in df_res:", len(df_res))
            plot_tuning_metrics(df_res, start_year, end_year, model_name, save_dir)