#
# Cross-feature codes for constructing cross-country vectors
CROSS_FEATURE_CODES = ['NE.IMP.GNFS.ZS', 'NE.EXP.GNFS.ZS']


import os
import numpy as np
import pandas as pd
from src.DataProcessing.utils import Paths

# Constants
DATA_PATH = os.path.join(Paths.DATA_DIR, "interpolation_top13.csv")
LAG = 2
TRAIN_YEARS = 15
VAL_YEARS = 3
START_YEAR = 1990
END_YEAR = 2020
YEARS = list(range(START_YEAR, END_YEAR + 1))

def build_lag_features(df, target_col, lags):
    """Add lag features for target_col using the specified lags."""
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df.groupby("country_code")[target_col].shift(lag)
    return df

def get_cross_features(df, exclude_col, lags):
    """
    For each lag, get all other indicator lag features except the target_col.
    Assumes all indicator columns are present.
    """
    indicator_cols = [col for col in df.columns if col not in ["country_code", "year"] and not col.endswith("_lag1") and not col.endswith("_lag2")]
    features = []
    for lag in lags:
        for col in indicator_cols:
            if col == exclude_col:
                continue
            lag_col = f"{col}_lag{lag}"
            if lag_col in df.columns:
                features.append(lag_col)
    return features

def get_fold_years():
    folds = []
    max_start = END_YEAR - (TRAIN_YEARS + VAL_YEARS) + 1
    for i in range(START_YEAR, max_start + 1, VAL_YEARS):
        train_range = list(range(i, i + TRAIN_YEARS))
        val_range = list(range(i + TRAIN_YEARS, i + TRAIN_YEARS + VAL_YEARS))
        folds.append((train_range, val_range))
    return folds
def main():
    df = pd.read_csv(DATA_PATH)
    feature_cols = [col for col in df.columns if col not in ["country_code", "year"]]
    all_results = []

    # Get all unique country_code and feature_code pairs
    country_codes = df["country_code"].unique()
    lambda_A_list = [0.1, 1.0, 10.0]
    lambda_M_list = [0.1, 1.0, 10.0]
    results_rows = []
    param_rows = []
    # Import metric and guiding score functions
    from src.DataProcessing.evaluate import compute_metrics
    from src.DataProcessing.utils import get_scores

    for country_code in country_codes:
        for feature_code in feature_cols:
            print(f"Tuning for {country_code} - {feature_code}")
            # Prepare data for this country and feature
            df_sub = df[df["country_code"] == country_code].copy()
            # Skip if not enough data
            if df_sub.shape[0] < (LAG + TRAIN_YEARS + VAL_YEARS):
                continue
            # Build lag features for all indicators
            indicator_cols = [col for col in df_sub.columns if col not in ["country_code", "year"]]
            for col in indicator_cols:
                df_sub = build_lag_features(df_sub, col, lags=[1,2])
            df_sub = df_sub.dropna().reset_index(drop=True)
            folds = get_fold_years()
            tau = 2  # LAG
            best_score = None
            best_result = None
            best_params = None
            # Precompute cross-feature indices in indicator_cols
            cross_feature_indices = []
            for code in CROSS_FEATURE_CODES:
                if code in indicator_cols:
                    cross_feature_indices.append(indicator_cols.index(code))
            # Grid search over lambda_A, lambda_M
            for lambda_A in lambda_A_list:
                for lambda_M in lambda_M_list:
                    fold_rows = []
                    all_y_true = []
                    all_y_pred = []
                    for train_years, val_years in folds:
                        group = df_sub.sort_values("year").reset_index(drop=True)
                        Y_c = group[feature_code].values
                        m = len(indicator_cols)
                        indicator_idx = indicator_cols.index(feature_code)
                        Z = []
                        for t in range(len(group)):
                            z_vec = np.zeros(m)
                            # For each cross feature, copy the value from other features (i.e., from indicator_cols) at t-1 if available.
                            for idx, code in zip(cross_feature_indices, CROSS_FEATURE_CODES):
                                if idx == indicator_idx:
                                    continue  # Don't use the current feature itself
                                if t-1 >= 0:
                                    z_vec[idx] = group.iloc[t-1][code]
                                else:
                                    z_vec[idx] = 0.0
                            Z.append(z_vec)
                        Z = np.stack(Z)
                        X = []
                        Y_target = []
                        years = []
                        for t in range(tau, len(Y_c)):
                            h_t = []
                            for i in range(1, tau + 1):
                                h_t.append([Y_c[t - i]])
                            h_t.append(Z[t - 1])
                            X.append(np.concatenate(h_t))
                            Y_target.append(Y_c[t])
                            years.append(group.iloc[t]["year"])
                        if len(X) == 0:
                            continue
                        X = np.stack(X)
                        Y_arr = np.stack(Y_target)
                        years_arr = np.array(years)
                        is_train = np.isin(years_arr, train_years)
                        is_val = np.isin(years_arr, val_years)
                        if np.sum(is_train) == 0 or np.sum(is_val) == 0:
                            continue
                        X_train = X[is_train]
                        y_train = Y_arr[is_train]
                        X_val = X[is_val]
                        y_val = Y_arr[is_val]
                        # Ridge regression with separate regularization for self and cross
                        # The first tau features are self-history, rest are cross
                        n_features = X_train.shape[1]
                        # Construct regularization matrix
                        reg = np.ones(n_features)
                        reg[:tau] = lambda_A
                        reg[tau:] = lambda_M
                        ridge_matrix = X_train.T @ X_train + np.diag(reg)
                        XTy = X_train.T @ y_train
                        try:
                            beta = np.linalg.solve(ridge_matrix, XTy)
                        except np.linalg.LinAlgError:
                            continue
                        y_pred = X_val @ beta
                        fold_result = pd.DataFrame({
                            "country_code": country_code,
                            "year": years_arr[is_val],
                            "y_true": y_val,
                            "y_pred": y_pred,
                            "feature_code": feature_code,
                            "lambda_A": lambda_A,
                            "lambda_M": lambda_M,
                        })
                        fold_rows.append(fold_result)
                        all_y_true.append(y_val)
                        all_y_pred.append(y_pred)
                    if fold_rows:
                        result_df = pd.concat(fold_rows, ignore_index=True)
                        # Use guiding score as selection criterion
                        y_true_full = np.concatenate(all_y_true) if all_y_true else np.array([])
                        y_pred_full = np.concatenate(all_y_pred) if all_y_pred else np.array([])
                        if y_true_full.size == 0:
                            continue
                        metrics = compute_metrics(y_true_full, y_pred_full)
                        guiding = get_scores.guiding_score(metrics)
                        # Pick best (highest guiding score)
                        if (best_score is None) or (guiding > best_score):
                            best_score = guiding
                            best_result = result_df.copy()
                            best_params = {"lambda_A": lambda_A, "lambda_M": lambda_M}
            if best_result is not None:
                # Compute and print guiding score for this country-feature
                y_true = best_result["y_true"].values
                y_pred = best_result["y_pred"].values
                metrics = compute_metrics(y_true, y_pred)
                guiding = get_scores.guiding_score(metrics)
                print(f"Guiding score for {country_code} - {feature_code}: {guiding:.4f}")
                results_rows.append(best_result)
                param_rows.append({
                    "country_code": country_code,
                    "feature_code": feature_code,
                    "lambda_A": best_params["lambda_A"],
                    "lambda_M": best_params["lambda_M"],
                    "guiding_score": best_score
                })
    if results_rows:
        final_df = pd.concat(results_rows, ignore_index=True)
        output_path = os.path.join(Paths.FIGURE_DIR,"Chapter6", "ridge2_kfold_results.csv")
        final_df.to_csv(output_path, index=False)
        print(f"Saved full results to {output_path}")
        param_path = os.path.join(Paths.FIGURE_DIR,"Chapter6", "ridge2_kfold_best_params.csv")
        pd.DataFrame(param_rows).to_csv(param_path, index=False)
        print(f"Saved best params to {param_path}")
    else:
        print("No results found.")

if __name__ == "__main__":
    main()