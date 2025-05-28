import pandas as pd
import numpy as np
import os
from src.DataProcessing.utils import selected_features, Paths
from src.DataProcessing.models import get_models
from src.DataProcessing.evaluate import compute_metrics
from sklearn.model_selection import KFold
from src.Chapter4.plot import plot_metrics

def run_pipeline(year_range, model_name, data_path, save_dir):
    df = pd.read_csv(data_path)
    feature_cols = selected_features.SF
    start_year, end_year = year_range

    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

    results = []
    for target in feature_cols:
        X = df[[col for col in feature_cols if col != target]]
        y = df[target]



        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        metrics_list = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = get_models.get_model(model_name)
            model.fit(X_train, y_train)
            if model_name == 'LWR':
                y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)
            y_pred = np.where(np.isnan(y_pred) | np.isinf(y_pred), y_train.mean(), y_pred)

            metrics = compute_metrics(y_test.to_numpy(), y_pred)
            metrics_list.append(metrics)

        avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}

        results.append({
            "target": target,
            **avg_metrics
        })

    df_res = pd.DataFrame(results)
    
    plot_metrics(df_res, start_year, end_year, model_name, save_dir)

    period_str = f"{start_year}_{end_year}"
    result_path = f"{save_dir}/{period_str}_{model_name}_results.csv"
    print(f"Saved to {result_path}")
    df_res.to_csv(result_path, index=False)

def main():
    data_path = os.path.join(Paths.DATA_DIR,"interpolation_top13.csv")
    save_dir = os.path.join(Paths.FIGURE_DIR, "Chapter4")
    year_ranges = [(1960, 2020), (2010, 2020)]
    model_names = ['LR', 'LWR', 'SVR', 'Ridge', 'Lasso', 'ElasticNet', 'RF', 'KNN', 'XGB']
    for year_range in year_ranges:
        for model_name in model_names:
            run_pipeline(year_range, model_name, data_path, save_dir)

if __name__ == "__main__":
    main()