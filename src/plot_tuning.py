def plot_featurewise_improvement(xgb_file, rf_file, default_xgb_file, default_rf_file, save_path):
    """
    Generate a combined figure with two subplots:
      - Left: improvement bar plot (overall, reusing plot_improvement_bars logic)
      - Right: per-feature improvement averaged across metrics (XGB vs RF)
    Args:
        xgb_file: CSV file of XGB tuning results (should include columns 'rmse/std', 'r2', 'mase', 'da')
        rf_file: CSV file of RF tuning results
        default_xgb_file: CSV file of XGB default results (same columns)
        default_rf_file: CSV file of RF default results
        save_path: path to save the output figure
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Load tuning and default results
    xgb_tuned = pd.read_csv(xgb_file).set_index("target")
    rf_tuned = pd.read_csv(rf_file).set_index("target")
    xgb_default = pd.read_csv(default_xgb_file).set_index("target")
    rf_default = pd.read_csv(default_rf_file).set_index("target")

    features = xgb_tuned.index.tolist()
    metrics = ["rmse/std", "r2", "mase", "da"]

    def compute_improve(df_def, df_tun, metric, positive=True):
        if positive:
            return 100 * (df_tun[metric] - df_def[metric]) / df_def[metric]
        else:
            return 100 * (df_def[metric] - df_tun[metric]) / df_def[metric]

    improvements_xgb = []
    improvements_rf = []

    for feat in features:
        imp_vals_xgb = []
        imp_vals_rf = []
        for m in metrics:
            sign = m in ["r2", "da"]
            imp_vals_xgb.append(compute_improve(xgb_default, xgb_tuned, m, sign)[feat])
            imp_vals_rf.append(compute_improve(rf_default, rf_tuned, m, sign)[feat])
        improvements_xgb.append(np.mean(imp_vals_xgb))
        improvements_rf.append(np.mean(imp_vals_rf))

    # Plot combined figure
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Feature-wise average improvement
    width = 0.35
    x = np.arange(len(features))
    axs[0].bar(x - width/2, improvements_xgb, width, label='XGB')
    axs[0].bar(x + width/2, improvements_rf, width, label='RF')
    axs[0].set_ylabel('Avg Improvement (%)')
    axs[0].set_title('Average Improvement per Feature')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(features, rotation=45, ha='right')
    axs[0].legend()
    axs[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)

    # Annotate bars in subplot 0 with improvement values
    for i, val in enumerate(improvements_xgb):
        axs[0].text(x[i] - width/2, val + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=7)
    for i, val in enumerate(improvements_rf):
        axs[0].text(x[i] + width/2, val + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=7)

    # Subplot 2: Overall model improvement (reusing logic from plot_improvement_bars)
    def avg_metric(df): return df[metrics].mean()

    def_model = {'XGB': avg_metric(xgb_default), 'RF': avg_metric(rf_default)}
    tun_model = {'XGB': avg_metric(xgb_tuned), 'RF': avg_metric(rf_tuned)}

    def comp_improve(def_val, tun_val, positive=True):
        return 100 * ((tun_val - def_val) if positive else (def_val - tun_val)) / def_val

    xgb_vals = [
        comp_improve(def_model['XGB']['rmse/std'], tun_model['XGB']['rmse/std'], False),
        comp_improve(def_model['XGB']['r2'], tun_model['XGB']['r2'], True),
        comp_improve(def_model['XGB']['mase'], tun_model['XGB']['mase'], False),
        comp_improve(def_model['XGB']['da'], tun_model['XGB']['da'], True),
    ]
    rf_vals = [
        comp_improve(def_model['RF']['rmse/std'], tun_model['RF']['rmse/std'], False),
        comp_improve(def_model['RF']['r2'], tun_model['RF']['r2'], True),
        comp_improve(def_model['RF']['mase'], tun_model['RF']['mase'], False),
        comp_improve(def_model['RF']['da'], tun_model['RF']['da'], True),
    ]

    labels = ["RMSE/STD", "R²", "MASE", "DA"]
    idx = np.arange(4)
    axs[1].bar(idx - width/2, xgb_vals, width, label='XGB')
    axs[1].bar(idx + width/2, rf_vals, width, label='RF')
    axs[1].set_xticks(idx)
    axs[1].set_xticklabels(labels)
    axs[1].set_ylabel('Improvement (%)')
    axs[1].set_title('Overall Metric Improvement')
    axs[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axs[1].legend()

    # Annotate bars in subplot 1 with improvement values
    for i, val in enumerate(xgb_vals):
        axs[1].text(i - width/2, val + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    for i, val in enumerate(rf_vals):
        axs[1].text(i + width/2, val + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved combined improvement figure to {save_path}")
import matplotlib.pyplot as plt

def plot_metrics(df_res, start_year, end_year, model_name, save_dir):
    plt.figure(figsize=(14, 5))

    # 1. RMSE/STD
    plt.subplot(1, 2, 1)
    plt.bar(df_res["target"], df_res["rmse/std"])
    plt.xlabel('Features')
    plt.ylabel('RMSE/STD')
    plt.title(f'{start_year}-{end_year} {model_name}  RMSE/STD')
    plt.axhline(1.0, color='r', linestyle='--', label='Feasible region threshold')
    plt.xticks(rotation=45, ha='right')
    plt.legend()

    # 2. R^2
    plt.subplot(1, 2, 2)
    plt.bar(df_res["target"], df_res["r2"])
    plt.xlabel('Features')
    plt.ylabel('R²')
    plt.title(f'{start_year}-{end_year} {model_name}  R²')
    plt.axhline(0.6, color='g', linestyle='--', label='Feasible region threshold')
    plt.ylim(-1, 1)
    plt.xticks(rotation=45, ha='right')
    plt.legend()

    plt.tight_layout()
    fname = f"{save_dir}/{start_year}_{end_year}_{model_name}.png"
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {fname}")

    # 3. MASE
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.bar(df_res["target"], df_res["mase"])
    plt.xlabel('Features')
    plt.ylabel('MASE')
    plt.title(f'{start_year}-{end_year} {model_name}  MASE')
    plt.axhline(1.0, color='r', linestyle='--', label='MASE = 1')
    plt.ylim(0, 4.0)
    plt.xticks(rotation=45, ha='right')
    plt.legend()

    # 4. Directional Accuracy (DA)
    plt.subplot(1, 2, 2)
    plt.bar(df_res["target"], df_res["da"])
    plt.xlabel('Features')
    plt.ylabel('Directional Accuracy')
    plt.title(f'{start_year}-{end_year} {model_name}  Directional Accuracy')
    plt.axhline(0.7, color='g', linestyle='--', label='DA = 0.7')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.legend()

    plt.tight_layout()
    fname2 = f"{save_dir}/{start_year}_{end_year}_{model_name}_mase_da.png"
    plt.savefig(fname2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {fname2}")

def plot_comparison_tuned_vs_default(df_summary, start_year, end_year, save_dir):
    """
    Generate four side-by-side bar charts comparing default vs tuned performance for XGB and RF
    across RMSE/STD, R², MASE, and DA.
    Args:
        df_summary: DataFrame with one row per model-version pair, columns: model_version, rmse/std, r2, mase, da
        start_year: int or str, start year of analysis
        end_year: int or str, end year of analysis
        save_dir: directory to save output figures
    """
    import matplotlib.pyplot as plt
    metrics = ['rmse/std', 'r2', 'mase', 'da']
    titles = ['RMSE/STD', 'R²', 'MASE', 'Directional Accuracy']
    ylims = [(0, 2), (-1, 1), (0, 4), (0, 1)]
    thresholds = [1.0, 0.6, 1.0, 0.7]
    colors = ['r', 'g', 'r', 'g']
    model_labels = ["XGB-Default", "XGB-Tuned", "RF-Default", "RF-Tuned"]

    # Ensure the order of rows matches model_labels
    values_matrix = []
    for label in model_labels:
        row = df_summary[df_summary['model_version'] == label]
        if row.empty:
            # If not present, append NaN
            values_matrix.append([float('nan')] * len(metrics))
        else:
            values_matrix.append([row.iloc[0][metric] for metric in metrics])
    values_matrix = list(zip(*values_matrix))  # shape: metrics x models

    for i, metric in enumerate(metrics):
        plt.figure(figsize=(8, 5))
        values = [df_summary[df_summary['model_version'] == label][metric].values[0]
                  if not df_summary[df_summary['model_version'] == label].empty else float('nan')
                  for label in model_labels]
        plt.bar(model_labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.axhline(thresholds[i], color=colors[i], linestyle='--', label=f'Threshold = {thresholds[i]}')
        plt.ylim(ylims[i])
        plt.ylabel(titles[i])
        plt.title(f'{titles[i]} Comparison: Default vs Tuned ({start_year}–{end_year})')
        plt.xticks(rotation=0)
        plt.legend()
        plt.tight_layout()
        fname = f"{save_dir}/{start_year}_{end_year}_compare_{metric.replace('/', '_')}.png"
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved figure to {fname}")


# New function: plot_improvement_bars
def plot_improvement_bars(xgb_file, rf_file, default_metrics, save_path):
    """
    Load XGB and RF tuning results, compare to default metrics, and plot improvement bars.
    Args:
        xgb_file: CSV file of XGB tuning results (should include columns 'rmse/std', 'r2', 'mase', 'da')
        rf_file: CSV file of RF tuning results
        default_metrics: dict, e.g. {'XGB': {...}, 'RF': {...}} with keys 'rmse/std', 'r2', 'mase', 'da'
        save_path: path to save the output barplot
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Load tuning results
    xgb_df = pd.read_csv(xgb_file)
    rf_df = pd.read_csv(rf_file)

    # Compute average metrics for each model
    metrics = ['rmse/std', 'r2', 'mase', 'da']
    xgb_avg = xgb_df[metrics].mean()
    rf_avg = rf_df[metrics].mean()

    # Default metrics should be a dict like: {'XGB': {...}, 'RF': {...}}
    xgb_default = default_metrics['XGB']
    rf_default = default_metrics['RF']

    # Compute improvement percentage: (default - tuned) / default for error metrics,
    # (tuned - default) / default for R² and DA (higher is better)
    def compute_improvement(default, tuned, positive=True):
        if default == 0:
            return np.nan
        if positive:
            return 100 * (tuned - default) / default
        else:
            return 100 * (default - tuned) / default

    improvements = {
        'XGB': [
            compute_improvement(xgb_default['rmse/std'], xgb_avg['rmse/std'], positive=False),
            compute_improvement(xgb_default['r2'], xgb_avg['r2'], positive=True),
            compute_improvement(xgb_default['mase'], xgb_avg['mase'], positive=False),
            compute_improvement(xgb_default['da'], xgb_avg['da'], positive=True)
        ],
        'RF': [
            compute_improvement(rf_default['rmse/std'], rf_avg['rmse/std'], positive=False),
            compute_improvement(rf_default['r2'], rf_avg['r2'], positive=True),
            compute_improvement(rf_default['mase'], rf_avg['mase'], positive=False),
            compute_improvement(rf_default['da'], rf_avg['da'], positive=True)
        ]
    }

    # Plot
    labels = ['RMSE/STD', 'R²', 'MASE', 'DA']
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, improvements['XGB'], width, label='XGB')
    rects2 = ax.bar(x + width/2, improvements['RF'], width, label='RF')

    ax.set_ylabel('Improvement (%)')
    ax.set_title('Performance Improvement after Tuning')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.axhline(0, color='gray', linewidth=0.8)

    for rect in list(rects1) + list(rects2):
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)

    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved improvement bar chart to {save_path}")

if __name__ == "__main__":
    import pandas as pd

    # Load default model results
    xgb_default_df = pd.read_csv("/Users/tianhaozhang/Desktop/XCountryOIRPrediction/figures/1960_2020_XGB_results.csv")
    rf_default_df = pd.read_csv("/Users/tianhaozhang/Desktop/XCountryOIRPrediction/figures/1960_2020_RF_results.csv")

    # Compute average metrics from default models
    default_metrics = {
        "XGB": {
            'rmse/std': xgb_default_df["rmse/std"].mean(),
            'r2': xgb_default_df["r2"].mean(),
            'mase': xgb_default_df["mase"].mean(),
            'da': xgb_default_df["da"].mean()
        },
        "RF": {
            'rmse/std': rf_default_df["rmse/std"].mean(),
            'r2': rf_default_df["r2"].mean(),
            'mase': rf_default_df["mase"].mean(),
            'da': rf_default_df["da"].mean()
        }
    }

    # Generate combined feature-wise + overall improvement chart
    plot_featurewise_improvement(
        xgb_file="/Users/tianhaozhang/Desktop/XCountryOIRPrediction/figures/chapter4tuning/XGB_tuning_result.csv",
        rf_file="/Users/tianhaozhang/Desktop/XCountryOIRPrediction/figures/chapter4tuning/RF_tuning_result.csv",
        default_xgb_file="/Users/tianhaozhang/Desktop/XCountryOIRPrediction/figures/1960_2020_XGB_results.csv",
        default_rf_file="/Users/tianhaozhang/Desktop/XCountryOIRPrediction/figures/1960_2020_RF_results.csv",
        save_path="/Users/tianhaozhang/Desktop/XCountryOIRPrediction/figures/chapter4tuning/featurewise_and_modelwise_improvement.png"
    )