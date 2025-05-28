import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.DataProcessing.utils import Paths
import os

INPUT_PATH = os.path.join(Paths.DATA_DIR, "top13_features.csv")
OUTPUT_PATH = os.path.join(Paths.DATA_DIR, "PCA_top13_features.csv")

def main():
    print(f"***{Paths.BASE_DIR}")
    df = pd.read_csv(INPUT_PATH)
    id_cols = ["country_code", "year"]
    feature_cols = [col for col in df.columns if col not in id_cols]
    

    df_sorted = df.sort_values(['country_code', 'year']).reset_index(drop=True)
    df_filled = df_sorted.copy()
    df_filled[feature_cols] = (
        df_sorted.groupby('country_code')[feature_cols]
        .apply(lambda group: group.interpolate(method='linear', limit_direction='both'))
        .reset_index(level=0, drop=True)
    )
    PATHCUR = os.path.join(Paths.DATA_DIR, "interpolation_top13.csv")
    df_filled.to_csv(PATHCUR, index=False)


    df_ready = df_filled.dropna(subset=feature_cols, how='any').reset_index(drop=True)
    print(f"Data shape after interpolation/dropna: {df_ready.shape}")

    # PCA
    X = df_ready[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.95, svd_solver='full')
    X_pca = pca.fit_transform(X_scaled)

    n_pc = X_pca.shape[1]
    pc_names = [f"PC{i+1}" for i in range(n_pc)]
    df_pca = pd.DataFrame(X_pca, columns=pc_names)
    df_out = pd.concat([df_ready[id_cols].reset_index(drop=True), df_pca], axis=1)
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"PCA completed! Output shape: {df_out.shape}")
    print(f"Saved PCA features to {OUTPUT_PATH}")

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=pc_names
    )
    loadings_path = os.path.join(Paths.DATA_DIR, "pca_feature_loadings.csv")
    loadings.to_csv(loadings_path)
    print(f"Saved PCA feature loadings to {loadings_path}")


    # Sort all component loadings by PC1 absolute value for better visibility
    loadings_sorted = loadings.reindex(loadings["PC1"].abs().sort_values(ascending=False).index)

    # Save all principal component loadings
    all_loadings_path = os.path.join(Paths.DATA_DIR, "pca_feature_loadings_all.csv")
    loadings_sorted.to_csv(all_loadings_path)
    print(f"Saved all PCA feature loadings (sorted by PC1 abs) to {all_loadings_path}")


    explained_var = pca.explained_variance_ratio_

    # Save explained variance ratios to CSV
    explained_var_df = pd.DataFrame({
        "Principal Component": pc_names,
        "Explained Variance Ratio": explained_var
    })
    explained_var_path = os.path.join(Paths.DATA_DIR, "pca_explained_variance_ratio.csv")
    explained_var_df.to_csv(explained_var_path, index=False)
    print(f"Saved explained variance ratios to {explained_var_path}")

    weighted_importance = (loadings.abs() * explained_var).sum(axis=1)
    weighted_importance_sorted = weighted_importance.sort_values(ascending=False)
    weighted_importance_path = os.path.join(Paths.DATA_DIR, "pca_feature_importance_weighted.csv")
    weighted_importance_sorted.to_csv(weighted_importance_path, header=["variance_weighted_importance"])
    print("Variance-weighted feature importance (descending):")
    print(weighted_importance_sorted)
    print(f"Saved variance-weighted feature importance to {weighted_importance_path}")

if __name__ == "__main__":
    main()