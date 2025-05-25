# pipeline_tuning.py: Only tune XGB and RF with GridSearchCV, save results to ./chapter4tuning
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from evaluate import compute_metrics
from models import get_models_tuning
from utils import selected_features

def load_data(data_path):
    df = pd.read_csv(data_path)
    feature_cols = selected_features.SF
    df = df.dropna(subset=feature_cols)
    X = df[feature_cols]
    data = {}
    for target in feature_cols:
        X_input = X.drop(columns=[target])
        y = X[target]
        data[target] = train_test_split(X_input, y, test_size=0.2, random_state=42)
    return data

def run_tuning(model_name, param_grid, data_dict, save_dir):
    results = []
    os.makedirs(save_dir, exist_ok=True)
    cnt = 0
    for target, (X_train, X_test, y_train, y_test) in data_dict.items():
        cnt += 1
        model = get_models_tuning.get_model_tuning(model_name, param_grid)
        print(f"Trial {cnt} : Running GridSearchCV for {model_name} on target: {target} with {len(param_grid)} parameters...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test.to_numpy(), y_pred)
        results.append({
            "target": target,
            "best_params": model.best_params_,
            **metrics
        })
        # Save full grid line results
        df_grid = pd.DataFrame(model.cv_results_)
        df_grid.to_csv(os.path.join(save_dir, f"{model_name}_grid_{target}.csv"), index=False)
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(save_dir, f"{model_name}_tuning_result.csv"), index=False)

def main():
    data_path = "/Users/tianhaozhang/Desktop/XCountryOIRPrediction/data/interpolation_top13.csv"
    save_dir = "/Users/tianhaozhang/Desktop/XCountryOIRPrediction/figures/chapter4tuning"
    data_dict = load_data(data_path)

    param_grids = {
        "XGB": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0]
        },
        "RF": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2"]
        }
    }

    for model_name, param_grid in param_grids.items():
        run_tuning(model_name, param_grid, data_dict, save_dir)

if __name__ == "__main__":
    main()