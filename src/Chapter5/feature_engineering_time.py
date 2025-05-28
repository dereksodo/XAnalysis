import pandas as pd
from src.DataProcessing.utils import selected_features

def create_lag_features(df, lags, target_columns=None):
    """
    Generate lagged features for selected indicators at the country level.

    Parameters:
    - df (pd.DataFrame): Input dataframe with columns ['country_code', 'year'] + selected features.
    - lags (list of int): List of lag periods to generate.
    - target_columns (list of str): Optionally restrict to a subset of selected_features.SF.

    Returns:
    - pd.DataFrame: A new dataframe with lagged features added.
    """
    features = target_columns or selected_features.SF
    # If original data has 'country' but not 'country_code', rename it
    if "country" in df.columns and "country_code" not in df.columns:
        df.rename(columns={"country": "country_code"}, inplace=True)
    df = df[["country_code", "year"] + features].copy()
    df.sort_values(by=["country_code", "year"], inplace=True)

    lagged_frames = []
    for country_code, group in df.groupby("country_code"):
        group = group.copy()
        for feat in features:
            for lag in lags:
                group[f"{feat}_lag{lag}"] = group[feat].shift(lag)
        lagged_frames.append(group)

    df_lagged = pd.concat(lagged_frames).dropna().reset_index(drop=True)
    return df_lagged