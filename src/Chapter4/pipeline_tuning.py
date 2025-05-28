# pipeline_tuning.py: Only tune XGB and RF with GridSearchCV, save results to ./Chapter4tuning
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.DataProcessing.evaluate import compute_metrics
from src.DataProcessing.models import get_models_tuning
from src.DataProcessing.utils import selected_features, Paths, get_scores
from sklearn.model_selection import ParameterGrid
from src.Chapter4.plot_tuning import plot_tuning_metrics

def load_data(data_path, year_range):
    df = pd.read_csv(data_path)
    start_year, end_year = year_range
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    feature_cols = selected_features.SF
    df = df.dropna(subset=feature_cols)
    X = df[feature_cols]
    data = {}
    for target in feature_cols:
        X_input = X.drop(columns=[target])
        y = X[target]
        data[target] = train_test_split(X_input, y, test_size=0.2, random_state=42)
    return data

def run_tuning(model_name, param_grid, data_dict, save_dir, start_year, end_year):
    results = []
    os.makedirs(save_dir, exist_ok=True)
    cnt = 0

    for target, (X_train, X_test, y_train, y_test) in data_dict.items():
        cnt += 1
        print(f"Trial {cnt}: Tuning {model_name} on target '{target}' with {len(list(ParameterGrid(param_grid)))} combinations...")

        best_score = -float('inf')
        best_params = None
        best_metrics = None
        grid_result = []

        for params in ParameterGrid(param_grid):
            #print(f"Parameters currently: {params}")
            model = get_models_tuning.get_model_tuning(model_name, params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = compute_metrics(y_test.to_numpy(), y_pred)
            score = get_scores.guiding_score(metrics)

            metrics_row = {"target": target, "params": params, **metrics, "guiding_score": score}
            grid_result.append(metrics_row)

            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = metrics

        results.append({
            "target": target,
            "best_params": best_params,
            **best_metrics
        })
        print(f"Saved as {start_year}_{end_year}_{model_name}_grid_{target}.csv")
        pd.DataFrame(grid_result).to_csv(os.path.join(save_dir, f"{start_year}_{end_year}_{model_name}_grid_{target}.csv"), index=False)

    df_res = pd.DataFrame(results)
    #plot_tuning_metrics(df_res, start_year, end_year, model_name, save_dir)
    # Ensure guiding_score is included in the results DataFrame (if present)
    if "guiding_score" in df_res.columns:
        df_res["guiding_score"] = df_res["guiding_score"]
    df_res.to_csv(os.path.join(save_dir, f"{start_year}_{end_year}_{model_name}_tuning_result.csv"), index=False)
    print(f"Tuning result for {model_name} saved to {save_dir}")

def main():
    data_path = os.path.join(Paths.DATA_DIR, "interpolation_top13.csv")
    save_dir = os.path.join(Paths.FIGURE_DIR, "Chapter4tuning")

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

    for year_range in [(1960, 2020), (2010, 2020)]:
        start_year, end_year = year_range
        data_dict = load_data(data_path, year_range)
        for model_name, param_grid in param_grids.items():
            run_tuning(model_name, param_grid, data_dict, save_dir, start_year, end_year)

if __name__ == "__main__":
    main()
