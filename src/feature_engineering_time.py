import pandas as pd
from utils import selected_features

def create_lag_features(df, lags=[1, 2], target_columns=None):
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
    # Rename 'country_code' to 'country' for consistency
    df = df.rename(columns={"country_code": "country"})
    df = df[["country", "year"] + features].copy()
    df.sort_values(by=["country", "year"], inplace=True)

    lagged_frames = []
    for country, group in df.groupby("country"):
        group = group.copy()
        for feat in features:
            for lag in lags:
                group[f"{feat}_lag{lag}"] = group[feat].shift(lag)
        lagged_frames.append(group)

    df_lagged = pd.concat(lagged_frames).dropna().reset_index(drop=True)
    return df_lagged