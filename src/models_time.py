
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import warnings

def get_models(model_name):
    if model_name == "XGB":
        return XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    elif model_name == "RF":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            max_features="sqrt",
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def fit_arima_model(y_train):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            model = ARIMA(y_train, order=(2, 0, 0))
            model_fit = model.fit()
            return model_fit
        except Exception:
            return None

def predict_arima(model_fit, steps):
    if model_fit is None:
        return np.full(steps, np.nan)
    forecast = model_fit.forecast(steps=steps)
    return forecast