import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

INPUT_PATH = "/Users/tianhaozhang/Desktop/XCountryOIRPrediction/data/top13_features.csv"
OUTPUT_PATH = "/Users/tianhaozhang/Desktop/XCountryOIRPrediction/data/PCA_top13_features.csv"

def main():
    df = pd.read_csv(INPUT_PATH)
    id_cols = ["country_code", "year"]
    feature_cols = [col for col in df.columns if col not in id_cols]
    
    # 插值（分国家）
    df_sorted = df.sort_values(['country_code', 'year']).reset_index(drop=True)
    df_filled = df_sorted.copy()
    df_filled[feature_cols] = (
        df_sorted.groupby('country_code')[feature_cols]
        .apply(lambda group: group.interpolate(method='linear', limit_direction='both'))
        .reset_index(level=0, drop=True)
    )
    df_filled.to_csv("/Users/tianhaozhang/Desktop/XCountryOIRPrediction/data/interpolation_top13.csv", index=False)

    # 只剔除极端仍有缺失的行
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

    # 1. 计算所有特征在每个主成分上的PCA loading（系数），保存为 pca_feature_loadings.csv
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=pc_names
    )
    loadings_path = "/Users/tianhaozhang/Desktop/XCountryOIRPrediction/data/pca_feature_loadings.csv"
    loadings.to_csv(loadings_path)
    print(f"Saved PCA feature loadings to {loadings_path}")

    # 2. 输出所有feature在PC1上的原始（带符号）loading，按绝对值排序、打印并保存
    pc1_loading = loadings["PC1"].sort_values(key=lambda x: x.abs(), ascending=False)
    print("Feature loading for PC1 (descending by absolute value, signed):")
    print(pc1_loading)
    pc1_loading_path = "/Users/tianhaozhang/Desktop/XCountryOIRPrediction/data/pca_feature_loading_pc1.csv"
    pc1_loading.to_csv(pc1_loading_path, header=["PC1_loading"])
    print(f"Saved PC1 feature loading (signed) to {pc1_loading_path}")

    # 3. 计算并输出所有feature在所有主成分上的方差加权重要性排序，保存
    explained_var = pca.explained_variance_ratio_
    # 计算加权绝对loadings
    weighted_importance = (loadings.abs() * explained_var).sum(axis=1)
    weighted_importance_sorted = weighted_importance.sort_values(ascending=False)
    weighted_importance_path = "/Users/tianhaozhang/Desktop/XCountryOIRPrediction/data/pca_feature_importance_weighted.csv"
    weighted_importance_sorted.to_csv(weighted_importance_path, header=["variance_weighted_importance"])
    print("Variance-weighted feature importance (descending):")
    print(weighted_importance_sorted)
    print(f"Saved variance-weighted feature importance to {weighted_importance_path}")

if __name__ == "__main__":
    main()