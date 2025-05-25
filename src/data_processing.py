import os
import pandas as pd
from utils import NamesAndCodes

# 路径设置
DATA_DIR = "/Users/tianhaozhang/Desktop/XCountryOIRPrediction/data/cleaned"
PROCESSED_DIR = "/Users/tianhaozhang/Desktop/XCountryOIRPrediction/data/processed"

def process_indicator_folder(indicator_folder, main_csv_filename):
    folder_path = os.path.join(DATA_DIR, indicator_folder)
    csv_path = os.path.join(folder_path, main_csv_filename)
    print(f"Processing: {csv_path}")
    df = pd.read_csv(csv_path, skiprows = 3)  # 不用表头，直接处理行

    # 第五行为年份，第1~4行为说明/其他
    # 截断 years 到2020（包含1960-2020）
    years = [y for y in df.columns[4:] if y.isdigit() and int(y) <= 2020]
    data_list = []
    typ = int(indicator_folder) - 1

    for idx in range(5, len(df)):
        country_code = df.iloc[idx, 1]  # B列为国家代码
        # 只处理你关注的国家/联盟
        if country_code in NamesAndCodes.country_codes:
            indicator_code = df.iloc[idx,3] # D列为指标代码
            #print(f"Processing Country: {country_code}")
            #print(indicator_code)
            if str(indicator_code) in NamesAndCodes.feature_codes[typ]:
                row_data = df.iloc[idx, 4:].tolist()  # E列及往后为数据
                # if empty, set to None
                row_data = [None if pd.isna(x) else x for x in row_data]
                print(f"Processing {country_code} - {indicator_code}")
                record = {
                    "country_code": country_code,
                    "indicator_code": indicator_code,
                }
                for i, year in enumerate(years):
                    record[str(year)] = row_data[i] if i < len(row_data) else None
                data_list.append(record)

    # 保存为新CSV
    output_df = pd.DataFrame(data_list)
    output_dir = os.path.join(PROCESSED_DIR, indicator_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "processed.csv")
    output_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

def main():
    # 指定每个文件夹下主csv文件名（你可根据需要修改）
    indicator_folders = [str(i) for i in range(1, 21)] # 假设文件夹名为1到20

    for indicator in indicator_folders:
            process_indicator_folder(indicator, "raw.csv")

if __name__ == "__main__":
    main()