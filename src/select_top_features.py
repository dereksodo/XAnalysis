import pandas as pd

# 路径
MERGED_PATH = "/Users/tianhaozhang/Desktop/XCountryOIRPrediction/data/merged_features.csv"
TOP13_PATH = "/Users/tianhaozhang/Desktop/XCountryOIRPrediction/data/top13_features.csv"

# 读取总表
df = pd.read_csv(MERGED_PATH)

# 排除ID列，统计各feature的非空比例
id_cols = ["country_code", "year"]
feature_cols = [col for col in df.columns if col not in id_cols]
feature_coverage = df[feature_cols].notna().mean().sort_values(ascending=False)

# 取前13个feature
top13 = feature_coverage.head(13).index.tolist()

print("Top 13 features by data coverage:")
for i, feat in enumerate(top13, 1):
    print(f"{i}. {feat}: {feature_coverage[feat]:.2%}")

# 生成只含前13个feature的新表
df_top13 = df[id_cols + top13]
df_top13.to_csv(TOP13_PATH, index=False)
print(f"Saved new table with top 13 features to {TOP13_PATH}")