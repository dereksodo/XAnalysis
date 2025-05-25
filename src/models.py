from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

# 可选：自定义Locally Weighted LR（LWR）
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

# 支持更多模型可按需添加
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
    def get_model_tuning(model_name, parameters):
        if model_name == 'XGB':
            base_model = XGBRegressor(random_state=42)
        elif model_name == 'RF':
            base_model = RandomForestRegressor(random_state=42)
        else:
            raise ValueError("Only XGB and RF support tuning.")
        model = GridSearchCV(base_model, parameters, cv=5, scoring='r2', return_train_score=True)
        return model
    

# 可选：统一fit/predict接口，便于主程序调用
def fit_and_predict(model, X_train, y_train, X_test):
    # sklearn的接口
    if hasattr(model, 'fit') and hasattr(model, 'predict'):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred
    else:
        raise ValueError("Model does not support fit/predict interface.")

# 可选：主程序/Notebook调用示例
"""
from models import get_model, fit_and_predict

model = get_model('lr')
y_pred = fit_and_predict(model, X_train, y_train, X_test)

model = get_model('svm', C=1.0, kernel='rbf')
y_pred = fit_and_predict(model, X_train, y_train, X_test)

model = get_model('lwr', tau=0.5)
y_pred = fit_and_predict(model, X_train, y_train, X_test)
"""