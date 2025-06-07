###########################
# Cross-feature codes used in cross-country interaction
CROSS_FEATURE_CODES = ['NE.IMP.GNFS.ZS', 'NE.EXP.GNFS.ZS']
###########################


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

def run_kfold(df, indicator):
    """
    For a given indicator, perform K-fold Ridge regression prediction using lagged features.
    Returns a DataFrame with columns: country_code, year, actual, predicted.
    """
    results = []
    # Build lag features for all indicators
    indicator_cols = [col for col in df.columns if col not in ["country_code", "year"]]
    for col in indicator_cols:
        df = build_lag_features(df, col, lags=[1,2])
    df = df.dropna().reset_index(drop=True)
    folds = get_fold_years()

    tau = 2  # LAG
    for train_years, val_years in folds:
        # For each country, build the time series input matrix as in the theoretical model
        fold_rows = []
        for country_code, group in df.groupby("country_code"):
            group = group.sort_values("year").reset_index(drop=True)
            # Extract Y_c: target indicator for this country
            Y_c = group[indicator].values
            m = len(indicator_cols)
            # Build cross-feature vector Z for each year
            # For each t, Z[t] is a zero vector except for entries corresponding to CROSS_FEATURE_CODES, which are filled from Y of other countries
            # For single-country prediction, we mimic the construction by using the values of the cross-feature codes from the other indicators (not used here, but the structure is prepared)
            # For this code, we fill Z as zero except for the indices in CROSS_FEATURE_CODES
            # Find indices of cross-feature codes in indicator_cols
            cross_feature_indices = []
            for code in CROSS_FEATURE_CODES:
                if code in indicator_cols:
                    cross_feature_indices.append(indicator_cols.index(code))
            Z = []
            for t in range(len(group)):
                z_vec = np.zeros(m)
                # For each cross-feature code, fill in the value from group at t-1 if available, else 0.0
                for idx, code in zip(cross_feature_indices, CROSS_FEATURE_CODES):
                    if t-1 >= 0:
                        z_vec[idx] = group.iloc[t-1][code]
                    else:
                        z_vec[idx] = 0.0
                Z.append(z_vec)
            Z = np.stack(Z)
            # Only consider rows where we have enough history (t >= tau)
            # For each t in tau..len(Y_c)-1, build features and target as described
            X = []
            Y_target = []
            years = []
            for t in range(tau, len(Y_c)):
                h_t = []
                # self-history
                for i in range(1, tau + 1):
                    h_t.append([Y_c[t - i]])
                # cross-feature
                h_t.append(Z[t - 1])
                # flatten and concatenate
                X.append(np.concatenate(h_t))
                Y_target.append(Y_c[t])
                years.append(group.iloc[t]["year"])
            if len(X) == 0:
                continue
            X = np.stack(X)
            Y_arr = np.stack(Y_target)
            # Assign folds to train or val depending on year
            years_arr = np.array(years)
            is_train = np.isin(years_arr, train_years)
            is_val = np.isin(years_arr, val_years)
            # Train
            if np.sum(is_train) == 0 or np.sum(is_val) == 0:
                continue
            X_train = X[is_train]
            y_train = Y_arr[is_train]
            X_val = X[is_val]
            y_val = Y_arr[is_val]
            # Closed-form ridge regression: beta = (X^T X + alpha*I)^-1 X^T y
            alpha = 1
            XTX = X_train.T @ X_train
            n_features = XTX.shape[0]
            ridge_matrix = XTX + alpha * np.eye(n_features)
            XTy = X_train.T @ y_train
            beta = np.linalg.solve(ridge_matrix, XTy)
            y_pred = X_val @ beta
            # Store results
            fold_result = pd.DataFrame({
                "country_code": country_code,
                "year": years_arr[is_val],
                "y_true": y_val,
                "y_pred": y_pred,
            })
            fold_rows.append(fold_result)
        if fold_rows:
            results.append(pd.concat(fold_rows, ignore_index=True))

    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=["country_code","year","y_true","y_pred"])

def main():
    df = pd.read_csv(DATA_PATH)
    feature_cols = [col for col in df.columns if col not in ["country_code", "year"]]
    all_results = []

    for indicator in feature_cols:
        result = run_kfold(df.copy(), indicator=indicator)
        result["feature_code"] = indicator
        all_results.append(result)

    final_df = pd.concat(all_results)
    output_path = os.path.join(Paths.FIGURE_DIR,"Chapter6", "ridge_kfold_results.csv")
    final_df.to_csv(output_path, index=False)
    print(f"Saved full results to {output_path}")

if __name__ == "__main__":
    main()