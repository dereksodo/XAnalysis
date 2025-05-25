

import pandas as pd
from feature_engineering_time import create_lag_features
from utils import selected_features
from models import get_models
from models_time import get_models as get_models_time, fit_arima_model, predict_arima
from evaluate import compute_metrics
from sklearn.model_selection import train_test_split
import os
from plot_time import plot_prediction_vs_actual
import numpy as np

def run_time_series_pipeline(
    data_path,
    model_list=["XGB", "RF", "ARIMA"],
    lags=[1, 2],
    test_size=0.2,
    save_path="/Users/tianhaozhang/Desktop/XCountryOIRPrediction/results/time_series_metrics.csv"
):
    df = pd.read_csv(data_path)
    df_lagged = create_lag_features(df, lags=lags)

    results = []

    for country in df_lagged["country"].unique():
        df_country = df_lagged[df_lagged["country"] == country]
        for feature in selected_features.SF:
            y = df_country[feature]
            X = df_country[
                [f"{f}_lag{l}" for f in selected_features.SF for l in lags if f != feature]
            ]
            if len(X) < 5:
                continue
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, shuffle=False, test_size=test_size
            )

            for model_name in model_list:
                forecast_results = []
                start_idx = 5  # 至少用前5年数据起步
                for t in range(start_idx, len(X) - 1):
                    X_train, y_train = X.iloc[:t], y.iloc[:t]
                    X_test, y_test = X.iloc[t:t+1], y.iloc[t:t+1]

                    # Skip if any NaN in train/test
                    if y_test.isnull().values.any() or y_train.isnull().values.any() or X_train.isnull().values.any() or X_test.isnull().values.any():
                        continue

                    print(f"Training {model_name} for {country}, predicting year {df_country.iloc[t]['year']}")

                    if model_name == "ARIMA":
                        model_fit = fit_arima_model(y_train)
                        y_pred = predict_arima(model_fit, steps=1)
                    else:
                        model = get_models_time(model_name) if model_name in ["XGB", "RF"] else get_models(model_name)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                    y_val = y_test.values[0]
                    y_hat = y_pred[0] if isinstance(y_pred, (np.ndarray, list)) else y_pred.iloc[0]

                    if np.isnan(y_val) or np.isnan(y_hat):
                        continue

                    forecast_results.append((y_val, y_hat))

                y_true_all = np.array([true for true, pred in forecast_results])
                y_pred_all = np.array([pred for true, pred in forecast_results])
                years_all = df_country.iloc[start_idx+1:len(X)]["year"].values
                min_len = min(len(y_true_all), len(y_pred_all), len(years_all))
                y_true_all = y_true_all[:min_len]
                y_pred_all = y_pred_all[:min_len]
                years_all = years_all[:min_len]

                # 评估
                metrics = compute_metrics(y_true_all, y_pred_all)
                metrics.update({
                    "target": feature,
                    "model": model_name,
                    "country": country
                })
                results.append(metrics)

                # 画图
                plot_dir = f"/Users/tianhaozhang/Desktop/XCountryOIRPrediction/figures/time_series/{model_name}"
                os.makedirs(plot_dir, exist_ok=True)
                plot_path = os.path.join(plot_dir, f"{country}_{feature}_rolling_forecast.png")
                plot_prediction_vs_actual(
                    y_true=y_true_all,
                    y_pred=y_pred_all,
                    years=years_all,
                    title=f"{country} - {feature} rolling forecast using {model_name}",
                    save_path=plot_path
                )
                print(f"Saved rolling forecast plot to {plot_path}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.DataFrame(results).to_csv(save_path, index=False)
    print(f"Saved results to {save_path}")


if __name__ == "__main__":
    run_time_series_pipeline(
        data_path="/Users/tianhaozhang/Desktop/XCountryOIRPrediction/data/interpolation_top13.csv",
        model_list=["XGB", "RF", "ARIMA"],
        lags=[1, 2],
        test_size=0.2,
        save_path="/Users/tianhaozhang/Desktop/XCountryOIRPrediction/results/time_series_metrics.csv"
    )