import os
import pandas as pd
from utils import NamesAndCodes, Paths

DATA_DIR = os.path.join(Paths.DATA_DIR, "cleaned")
PROCESSED_DIR = os.path.join(Paths.DATA_DIR, "processed")

def process_indicator_folder(indicator_folder, main_csv_filename):
    folder_path = os.path.join(DATA_DIR, indicator_folder)
    csv_path = os.path.join(folder_path, main_csv_filename)
    print(f"Processing: {csv_path}")
    df = pd.read_csv(csv_path, skiprows = 3)

    years = [y for y in df.columns[4:] if y.isdigit() and int(y) <= 2020]
    data_list = []
    typ = int(indicator_folder) - 1

    for idx in range(5, len(df)):
        country_code = df.iloc[idx, 1]

        if country_code in NamesAndCodes.country_codes:
            indicator_code = df.iloc[idx,3]
            #print(f"Processing Country: {country_code}")
            #print(indicator_code)
            if str(indicator_code) in NamesAndCodes.feature_codes[typ]:
                row_data = df.iloc[idx, 4:].tolist()
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

    output_df = pd.DataFrame(data_list)
    output_dir = os.path.join(PROCESSED_DIR, indicator_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "processed.csv")
    output_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

def main():
    indicator_folders = [str(i) for i in range(1, 21)]

    for indicator in indicator_folders:
            process_indicator_folder(indicator, "raw.csv")

if __name__ == "__main__":
    main()