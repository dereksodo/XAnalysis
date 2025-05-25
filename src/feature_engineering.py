import os
import pandas as pd

DATA_DIR = "/Users/tianhaozhang/Desktop/XCountryOIRPrediction/data/processed"
OUTPUT_PATH = "/Users/tianhaozhang/Desktop/XCountryOIRPrediction/data/merged_features.csv"

def main():
    # 字典，key: (country_code, year), value: {feature1: v1, ...}
    all_data = dict()
    feature_names = set()

    # 依次读取 processed 文件夹下所有 feature 文件夹
    feature_folders = sorted([f for f in os.listdir(DATA_DIR)])
    for feature_folder in feature_folders:
        csv_path = os.path.join(DATA_DIR, feature_folder, "processed.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        # 获取所有年份列（排除 country_code 和 indicator_code）
        year_cols = [col for col in df.columns if col not in ("country_code", "indicator_code")]
        # 遍历所有 indicator_code
        for indicator_code, group_df in df.groupby("indicator_code"):
            feature_names.add(indicator_code)
            # 遍历每行（每个 country_code）
            for _, row in group_df.iterrows():
                country_code = row['country_code']
                # 遍历每个年份列，将数据加入 all_data
                for year in year_cols:
                    key = (country_code, int(year))
                    if key not in all_data:
                        all_data[key] = dict()
                    # 判断如果该数据为 0，则将其设为 float('nan')，否则保持原值
                    value = row[year]
                    if value == 0:
                        value = float('nan')
                    all_data[key][indicator_code] = value

    # 汇总所有 (country_code, year)，确保每个样本都有所有feature（无则为NaN）
    keys = sorted(all_data.keys())
    feature_names = sorted(feature_names)
    rows = []
    for key in keys:
        country_code, year = key
        row = {
            "country_code": country_code,
            "year": year
        }
        # 填充所有feature，没有的自动补NaN
        for fname in feature_names:
            row[fname] = all_data[key].get(fname, float('nan'))
        rows.append(row)
    df_all = pd.DataFrame(rows)
    df_all = df_all.sort_values(['country_code', 'year']).reset_index(drop=True)
    df_all.to_csv(OUTPUT_PATH, index=False)
    print(f"Total merged feature table shape: {df_all.shape}")
    print(f"Saved merged features to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()