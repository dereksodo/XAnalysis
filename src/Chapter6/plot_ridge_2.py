

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import new metric computation logic
from src.DataProcessing.evaluate import compute_metrics
from src.DataProcessing.utils import get_scores, Paths, FeatureNames, selected_features

# Set year range constants at the top
START_YEAR = 2005
END_YEAR = 2019

def main():
    # Read results from the correct CSV file path
    df = pd.read_csv(os.path.join(Paths.FIGURE_DIR, "Chapter6", "ridge2_kfold_results.csv"))

    df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)]
    # Compute guiding scores grouped by both country_code and feature_code
    country_feature_scores = []
    for (country_code, feature_code), group in df.groupby(["country_code", "feature_code"]):
        y_true = group["y_true"]
        y_pred = group["y_pred"]
        metrics = compute_metrics(y_true, y_pred)
        gscore = get_scores.guiding_score(metrics)
        country_feature_scores.append({
            "country_code": country_code,
            "feature_code": feature_code,
            "guiding_score": gscore
        })
    df_country_feature_scores = pd.DataFrame(country_feature_scores)
    # Save to CSV
    country_feature_out_path = os.path.join(Paths.FIGURE_DIR, "Chapter6", "ridge2_country_feature_scores.csv")
    df_country_feature_scores.to_csv(country_feature_out_path, index=False)

    # Compute average guiding score per feature_code (across all countries) for Ridge
    df_scores = df_country_feature_scores.groupby("feature_code", as_index=False)["guiding_score"].mean()
    # Compute average guiding score per country_code (across all features) for Ridge
    df_country_scores = df_country_feature_scores.groupby("country_code", as_index=False)["guiding_score"].mean()
    # Sort for better visualization
    df_scores = df_scores.sort_values("guiding_score", ascending=False)
    df_country_scores = df_country_scores.sort_values("guiding_score", ascending=False)

    # -----------------------------------------------------------
    # Load Naive prediction results for each (country, feature)
    # -----------------------------------------------------------
    # Ridge (country, feature) pairs
    ridge_pairs = set(zip(df_country_feature_scores["country_code"], df_country_feature_scores["feature_code"]))
    # Directory containing Naive predictions
    naive_dir = os.path.join(Paths.FIGURE_DIR, "chapter5", "predictions", "Naive")
    # Build mapping: (country, feature) -> file path
    naive_files = glob.glob(os.path.join(naive_dir, "*.csv"))
    # Map (country_code, feature_code) to file path
    naive_file_map = {}
    for fpath in naive_files:
        fname = os.path.basename(fpath)
        if fname.endswith(".csv"):
            parts = fname[:-4].split("_", 1)
            if len(parts) == 2:
                country_code, feature_code = parts
                naive_file_map[(country_code, feature_code)] = fpath

    # For each (country, feature) in Ridge results, if corresponding Naive file exists, compute guiding score
    naive_country_feature_scores = []
    for country_code, feature_code in ridge_pairs:
        key = (country_code, feature_code)
        if key in naive_file_map:
            try:
                df_naive = pd.read_csv(naive_file_map[key])
                # Expect columns: y_true, y_pred, maybe year
                # Only keep years in START_YEAR..END_YEAR
                if "year" in df_naive.columns:
                    df_naive = df_naive[(df_naive["year"] >= START_YEAR) & (df_naive["year"] <= END_YEAR)]
                y_true = df_naive["y_true"]
                y_pred = df_naive["y_pred"]
                metrics = compute_metrics(y_true, y_pred)
                gscore = get_scores.guiding_score(metrics)
                naive_country_feature_scores.append({
                    "country_code": country_code,
                    "feature_code": feature_code,
                    "guiding_score": gscore
                })
            except Exception as e:
                # Could log error here, but just skip for now
                continue
    df_naive_country_feature_scores = pd.DataFrame(naive_country_feature_scores)
    # Compute average guiding score per feature_code (across all countries) for Naive
    if not df_naive_country_feature_scores.empty:
        df_naive_scores = df_naive_country_feature_scores.groupby("feature_code", as_index=False)["guiding_score"].mean()
    else:
        # If no Naive results found, create empty DataFrame
        df_naive_scores = pd.DataFrame(columns=["feature_code", "guiding_score"])

    # -----------------------------------------------------------
    # Prepare country-wise average guiding scores for Ridge and Naive
    # -----------------------------------------------------------
    # Ridge: average guiding score per country (already computed)
    ridge_country_scores = df_country_feature_scores.groupby("country_code", as_index=False)["guiding_score"].mean()
    # Naive: average guiding score per country (across all features available)
    if not df_naive_country_feature_scores.empty:
        naive_country_scores = df_naive_country_feature_scores.groupby("country_code", as_index=False)["guiding_score"].mean()
    else:
        naive_country_scores = pd.DataFrame(columns=["country_code", "guiding_score"])

    # Align country codes for plotting (intersection only, sorted for clarity)
    countries = sorted(set(ridge_country_scores["country_code"]) & set(naive_country_scores["country_code"]))
    ridge_avg_scores_c = []
    naive_avg_scores_c = []
    for cc in countries:
        ridge_val = ridge_country_scores[ridge_country_scores["country_code"] == cc]["guiding_score"].values
        naive_val = naive_country_scores[naive_country_scores["country_code"] == cc]["guiding_score"].values
        ridge_avg_scores_c.append(ridge_val[0] if len(ridge_val) > 0 else np.nan)
        naive_avg_scores_c.append(naive_val[0] if len(naive_val) > 0 else np.nan)

    # For left plot (by feature): retain previous logic for completeness
    features = sorted(set(df_scores["feature_code"]) & set(df_naive_scores["feature_code"]))
    ridge_avg_scores = []
    naive_avg_scores = []
    for feat in features:
        ridge_val = df_scores[df_scores["feature_code"] == feat]["guiding_score"].values
        naive_val = df_naive_scores[df_naive_scores["feature_code"] == feat]["guiding_score"].values
        ridge_avg_scores.append(ridge_val[0] if len(ridge_val) > 0 else np.nan)
        naive_avg_scores.append(naive_val[0] if len(naive_val) > 0 else np.nan)

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Guiding Score by Feature - Ridge vs Naive (line chart)
    x_labels = [FeatureNames.code2name.get(code, code) for code in selected_features.SF]
    axes[0].plot(features, ridge_avg_scores, marker="o", label="Ridge", color="skyblue")
    axes[0].plot(features, naive_avg_scores, marker="o", label="Naive", color="orange")
    axes[0].set_xticklabels(x_labels, rotation=45, ha="right")
    axes[0].set_ylabel("Average Guiding Score")
    axes[0].set_xlabel("Feature")
    axes[0].set_title("Average Guiding Score by Feature: Ridge vs Naive")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend()

    # Right plot: Guiding Score by Country - Ridge vs Naive (line chart)
    axes[1].plot(countries, ridge_avg_scores_c, marker="o", label="Ridge", color="skyblue")
    axes[1].plot(countries, naive_avg_scores_c, marker="o", label="Naive", color="orange")
    axes[1].set_ylabel("Average Guiding Score")
    axes[1].set_xlabel("Country")
    axes[1].set_title("Average Guiding Score by Country: Ridge vs Naive")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(Paths.FIGURE_DIR, "Chapter6", "guiding_score_ridge2.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    # ------------------------------------------------------------------------
    # Additional Figure: 2x2 subplots, one for each metric (RMSE/std, R², MASE, DA), Ridge2 vs Naive
    # ------------------------------------------------------------------------
    # For each (country, feature), get Ridge2 and Naive predictions, compute metrics
    # We'll aggregate by feature (or by (country, feature)), then average metrics for Ridge2 and Naive
    # First, for each (country, feature): compute Ridge2 and Naive metrics
    ridge_metrics = []
    # Build mapping for quick lookup
    ridge_group = df.groupby(["country_code", "feature_code"])
    for (country_code, feature_code), group in ridge_group:
        y_true = group["y_true"]
        y_pred = group["y_pred"]
        metrics = compute_metrics(y_true, y_pred)
        ridge_metrics.append({
            "country_code": country_code,
            "feature_code": feature_code,
            "rmse_std": metrics["rmse/std"],
            "r2": metrics["r2"],
            "mase": metrics["mase"],
            "da": metrics["da"]
        })
    df_ridge_metrics = pd.DataFrame(ridge_metrics)
    # For Naive, repeat for all available (country, feature)
    naive_metrics = []
    for country_code, feature_code in ridge_pairs:
        key = (country_code, feature_code)
        if key in naive_file_map:
            try:
                df_naive = pd.read_csv(naive_file_map[key])
                if "year" in df_naive.columns:
                    df_naive = df_naive[(df_naive["year"] >= START_YEAR) & (df_naive["year"] <= END_YEAR)]
                y_true = df_naive["y_true"]
                y_pred = df_naive["y_pred"]
                metrics = compute_metrics(y_true, y_pred)
                naive_metrics.append({
                    "country_code": country_code,
                    "feature_code": feature_code,
                    "rmse_std": metrics["rmse/std"],
                    "r2": metrics["r2"],
                    "mase": metrics["mase"],
                    "da": metrics["da"]
                })
            except Exception:
                continue
    df_naive_metrics = pd.DataFrame(naive_metrics)

    # List of metrics and their display names
    metric_keys = ["rmse_std", "r2", "mase", "da"]
    metric_titles = ["RMSE/std", "R²", "MASE", "DA"]

    # For each metric, for each feature (or (country, feature)), average across countries
    # We'll plot by feature (indicator), average value for Ridge2 and Naive
    features_metrics = sorted(set(df_ridge_metrics["feature_code"]) & set(df_naive_metrics["feature_code"]))
    # Prepare 2x2 subplots
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    axes2f = axes2.flatten()
    for idx, (metric, title) in enumerate(zip(metric_keys, metric_titles)):
        ax = axes2f[idx]
        ridge_vals = []
        naive_vals = []
        for feat in features_metrics:
            ridge_feat_vals = df_ridge_metrics[df_ridge_metrics["feature_code"] == feat][metric].mean()
            naive_feat_vals = df_naive_metrics[df_naive_metrics["feature_code"] == feat][metric].mean()
            ridge_vals.append(ridge_feat_vals)
            naive_vals.append(naive_feat_vals)
        x_labels = [FeatureNames.code2name.get(code, code) for code in features_metrics]
        ax.plot(features_metrics, ridge_vals, marker="o", label="Ridge2", color="skyblue", linestyle="-")
        ax.plot(features_metrics, naive_vals, marker="s", label="Naive", color="orange", linestyle="--")
        ax.set_xticks(features_metrics)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_xlabel("Feature")
        ax.set_ylabel(f"Average {title}")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.5)
    # Hide any unused subplots (shouldn't happen, but for safety)
    for j in range(len(metric_keys), 4):
        fig2.delaxes(axes2f[j])
    fig2.tight_layout()
    out_path2 = os.path.join(Paths.FIGURE_DIR, "Chapter6", "ridge2_vs_naive_by_metric.png")
    fig2.savefig(out_path2, dpi=300)
    plt.close(fig2)

if __name__ == "__main__":
    main()