import pandas as pd
from src.Chapter5.feature_engineering_time import create_lag_features
from src.DataProcessing.utils import selected_features, Paths
from src.DataProcessing.models import get_models
from src.Chapter5.models_time import get_models as get_models_time, fit_arima_model, predict_arima
from sklearn.model_selection import train_test_split
import os
import numpy as np

def run_time_series_pipeline(
    data_path,
    model_list=["Naive", "ARIMA", "Rolling_XGB_lag2", "Rolling_XGB_lag3"],
    save_path=os.path.join("figures", "Chapter5", "pipeline_time_comparison.csv"),
):
    """
    Runs a rolling prediction pipeline for time series models on each country and feature.
    Splits the process into three clear parts:
      1. Preprocess data with lag features using create_lag_features (df_lagged).
      2. For models that do NOT require lag features (Naive, ARIMA), predict using original df.
      3. For models that require lag features (XGB_lag2, XGB_lag3, Rolling), use df_lagged.
    Saves all predictions from all models/countries/features as separate CSVs under /figures/Chapter5/.
    """
    df = pd.read_csv(data_path)
    export_dir = os.path.join("figures", "Chapter5")
    os.makedirs(export_dir, exist_ok=True)

    # --- 1. Preprocess data with lag features ---
    lag2 = [1, 2]
    lag3 = [1, 2, 3]
    df_lag2 = create_lag_features(df, lags=lag2)
    df_lag3 = create_lag_features(df, lags=lag3)

    # --- 2. Models that do NOT require lag features ---
    for model_name in model_list:
        if model_name in ["Naive", "ARIMA"]:
            for country_code in df["country_code"].unique():
                df_country = df[df["country_code"] == country_code]
                for feature in selected_features.SF:
                    pred_dir = os.path.join("figures", "Chapter5", "predictions", model_name)
                    os.makedirs(pred_dir, exist_ok=True)
                    save_path = os.path.join(pred_dir, f"{country_code}_{feature}.csv")
                    if os.path.exists(save_path):
                        print(f"Skipping {save_path}, already exists.")
                        continue
                    y = df_country[feature]
                    if len(y) < 5:
                        continue
                    results = []
                    for t in range(5, len(y) - 1):
                        y_val = y.iloc[t]
                        year_val = df_country.iloc[t]["year"]
                        y_hat = None
                        # Add training log for ARIMA and Naive (so only ARIMA triggers fit)
                        print(f"Training {model_name} for {country_code} - {feature}, year {df_country.iloc[t]['year']}")
                        if model_name == "Naive":
                            y_hat = y.iloc[t - 1]
                        elif model_name == "ARIMA":
                            # Warning: statsmodels issues a FutureWarning when no datetime index is used.
                            # This does not affect results but may require using a DateTime index in future versions.
                            y_train = y.iloc[:t]
                            if y_train.isnull().any():
                                continue
                            model_fit = fit_arima_model(y_train)
                            pred = predict_arima(model_fit, steps=1)
                            y_hat = pred[0] if isinstance(pred, np.ndarray) else pred.iloc[0]
                        if y_hat is None or np.isnan(y_val) or np.isnan(y_hat):
                            continue
                        results.append({
                            "country_code": country_code,
                            "year": year_val,
                            "model": model_name,
                            "target": feature,
                            "y_true": y_val,
                            "y_pred": y_hat,
                            "lag": "-"
                        })
                    if results:
                        # Save to per-model-country-feature CSV
                        pd.DataFrame(results).to_csv(save_path, index=False)
                        print(f"Saved predictions to {save_path}")

    # --- 3. Models that require lag features ---
    for model_name in model_list:
        # Skip non‑rolling lag models entirely
        if ("lag2" in model_name or "lag3" in model_name) and "Rolling" not in model_name:
            continue
        if "lag2" in model_name:
            lags = lag2
            lag_label = "lag2"
            df_lagged = df_lag2
        elif "lag3" in model_name:
            lags = lag3
            lag_label = "lag3"
            df_lagged = df_lag3
        else:
            continue  # already handled Naive/ARIMA above
        for country_code in df_lagged["country_code"].unique():
            df_country = df_lagged[df_lagged["country_code"] == country_code]
            for feature in selected_features.SF:
                pred_dir = os.path.join("figures", "Chapter5", "predictions", model_name)
                os.makedirs(pred_dir, exist_ok=True)
                save_path = os.path.join(pred_dir, f"{country_code}_{feature}.csv")
                if os.path.exists(save_path):
                    print(f"Skipping {save_path}, already exists.")
                    continue
                y = df_country[feature]
                # Include target's own lagged features in predictors
                X = df_country[
                    [f"{f}_lag{l}" for f in selected_features.SF for l in lags]
                ] if lags else None
                if len(y) < 5:
                    continue
                results = []
                static_model = None  # for static lag models
                for t in range(5, len(y) - 1):
                    y_val = y.iloc[t]
                    year_val = df_country.iloc[t]["year"]
                    y_hat = None
                    if X is None or X.iloc[:t].isnull().any().any() or X.iloc[t:t+1].isnull().any().any():
                        continue
                    # ---------------- prediction logic ----------------
                    if "Rolling" in model_name:
                        # Rolling: fit every step using data up to year t‑1
                        model = get_models_time("XGB")
                        model.fit(X.iloc[:t], y.iloc[:t])
                        y_hat = model.predict(X.iloc[t:t+1])[0]
                    else:
                        # Static: fit once on the first 5 years, then reuse
                        if t == 5:
                            static_model = get_models_time("XGB")
                            static_model.fit(X.iloc[:5], y.iloc[:5])
                        if static_model is None:
                            continue  # not enough data to fit
                        y_hat = static_model.predict(X.iloc[t:t+1])[0]
                    if y_hat is None or np.isnan(y_val) or np.isnan(y_hat):
                        continue
                    results.append({
                        "country_code": country_code,
                        "year": year_val,
                        "model": model_name,
                        "target": feature,
                        "y_true": y_val,
                        "y_pred": y_hat,
                        "lag": lag_label
                    })
                if results:
                    # Save to per-model-country-feature CSV
                    pd.DataFrame(results).to_csv(save_path, index=False)
                    print(f"Saved predictions to {save_path}")


# Add a main block to run the pipeline with the specified parameters
if __name__ == "__main__":
    run_time_series_pipeline(
        data_path=os.path.join(Paths.DATA_DIR,"interpolation_top13.csv")
    )