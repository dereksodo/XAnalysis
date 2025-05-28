from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

import numpy as np

class LWR:
    def __init__(self, tau=1.0):
        self.tau = tau

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X_test = np.array(X)
        y_pred = np.zeros(X_test.shape[0])
        mean_y = np.mean(self.y_train)
        eps = 1e-6

        for i, x in enumerate(X_test):
            diff = self.X_train - x
            weights = np.exp(-np.sum(diff**2, axis=1) / (2 * self.tau**2))
            if np.sum(weights) < eps:
                y_pred[i] = mean_y
                continue
            W = np.diag(weights)
            XTWX = self.X_train.T @ W @ self.X_train
            try:
                if np.linalg.cond(XTWX) > 1e8:
                    theta = np.linalg.pinv(XTWX) @ self.X_train.T @ W @ self.y_train
                else:
                    theta = np.linalg.inv(XTWX) @ self.X_train.T @ W @ self.y_train
                y_pred[i] = x @ theta
            except Exception:
                y_pred[i] = mean_y
            if np.isnan(y_pred[i]) or np.isinf(y_pred[i]):
                y_pred[i] = mean_y
        return y_pred


class get_models:
    def get_model(model_name):
        if model_name == 'LR':
            return LinearRegression()
        elif model_name == 'Ridge':
            return Ridge(alpha=1.0)
        elif model_name == 'Lasso':
            return Lasso(alpha=0.1)
        elif model_name == 'ElasticNet':
            return ElasticNet(alpha=0.1, l1_ratio=0.5)
        elif model_name == 'RF':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_name == 'KNN':
            return KNeighborsRegressor(n_neighbors=5)
        elif model_name == 'SVR':
            return SVR(kernel='rbf', C=1.0, epsilon=0.1)
        elif model_name == 'XGB':
            return XGBRegressor(n_estimators=100, random_state=42)
        elif model_name == 'LWR':
            return LWR(tau=1.0)
        else:
            raise ValueError(f"Unknown model: {model_name}")

class get_models_tuning:
    @staticmethod
    def get_model_tuning(model_name, params):
        if model_name == "XGB":
            from xgboost import XGBRegressor
            return XGBRegressor(**params)
        elif model_name == "RF":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**params)
    
class get_models_tuning_time:
    def get_model_tuning(model_name, param_grid, scoring=None, refit=True):
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestRegressor
        from xgboost import XGBRegressor

        if model_name == "XGB":
            estimator = XGBRegressor()
        elif model_name == "RF":
            estimator = RandomForestRegressor()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        model = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=5,
            refit=refit,
            n_jobs=-1
        )
        return model
    


def fit_and_predict(model, X_train, y_train, X_test):

    if hasattr(model, 'fit') and hasattr(model, 'predict'):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred
    else:
        raise ValueError("Model does not support fit/predict interface.")


"""
from models import get_model, fit_and_predict

model = get_model('lr')
y_pred = fit_and_predict(model, X_train, y_train, X_test)

model = get_model('svm', C=1.0, kernel='rbf')
y_pred = fit_and_predict(model, X_train, y_train, X_test)

model = get_model('lwr', tau=0.5)
y_pred = fit_and_predict(model, X_train, y_train, X_test)
"""