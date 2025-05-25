import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

from utils import selected_features
from models import get_models
from evaluate import compute_metrics


# --- 训练与评估主流程 ---
def run_pipeline(year_range, model_name, data_path, save_dir):
    df = pd.read_csv(data_path)
    id_cols = ["country_code", "year"]
    feature_cols = selected_features.SF
    start_year, end_year = year_range

    # 筛选年份
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

    results = []
    for target in feature_cols:
        X = df[[col for col in feature_cols if col != target]]
        y = df[target]

        # 随机分割（你也可以按年份分割，比如train:前80%，test:后20%）
        from sklearn.model_selection import KFold

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
            # 防nan
            y_pred = np.where(np.isnan(y_pred) | np.isinf(y_pred), y_train.mean(), y_pred)

            metrics = compute_metrics(y_test.to_numpy(), y_pred)
            metrics_list.append(metrics)

        # 计算各指标的平均值
        avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}

        # 保存每个特征的K折平均表现
        results.append({
            "target": target,
            **avg_metrics
        })

    df_res = pd.DataFrame(results)

    from plot import plot_metrics
    plot_metrics(df_res, start_year, end_year, model_name, save_dir)

    # Save results in the expected format for figure2.py
    period_str = f"{start_year}_{end_year}"
    result_path = f"{save_dir}/{period_str}_{model_name}_results.csv"
    df_res.to_csv(result_path, index=False)

# --- 主循环 ---
def main():
    data_path = "/Users/tianhaozhang/Desktop/XCountryOIRPrediction/data/interpolation_top13.csv"
    save_dir = "/Users/tianhaozhang/Desktop/XCountryOIRPrediction/figures"
    year_ranges = [(1960, 2020), (2010, 2020)]
    # 支持更多模型：LR, LWR, SVR, Ridge, Lasso, ElasticNet, RF, KNN, XGB
    model_names = ['LR', 'LWR', 'SVR', 'Ridge', 'Lasso', 'ElasticNet', 'RF', 'KNN', 'XGB']  # 可扩展

    for year_range in year_ranges:
        for model_name in model_names:
            run_pipeline(year_range, model_name, data_path, save_dir)

if __name__ == "__main__":
    main()