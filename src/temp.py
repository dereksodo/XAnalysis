import pandas as pd

data_path = "/Users/tianhaozhang/Desktop/XCountryOIRPrediction/data/interpolation_top13.csv"
df = pd.read_csv(data_path)

# 假定你的表格有 'year' 这一列（int 或 str 都可）
df["year"] = df["year"].astype(int)  # 如有必要，强制类型转换

samples_1960_2020 = df[(df["year"] >= 1960) & (df["year"] <= 2020)].shape[0]
samples_2010_2020 = df[(df["year"] >= 2010) & (df["year"] <= 2020)].shape[0]

print("Number of samples (1960-2020):", samples_1960_2020)
print("Number of samples (2010-2020):", samples_2010_2020)