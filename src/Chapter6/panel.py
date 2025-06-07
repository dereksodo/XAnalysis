"""
make_panel_1988_2019.py
-------------------------------------------------
将 interpolation_top13.csv（包含 13 个指标的宽表）
转换为 Chapter 6 所需的长表 panel_1988_2019.csv。


    ['year', 'country_code', 'IMP', 'EXP', 'GDP', 'GDP_GR', 'LEX']
"""

import pandas as pd
from pathlib import Path
import os
# --------------------------------------------------
# Imports and Paths
# --------------------------------------------------
from src.DataProcessing.utils import Paths, get_scores, FeatureNames, NamesAndCodes
from src.DataProcessing.evaluate import compute_metrics
from src.DataProcessing.utils import selected_features
# --------------------------------------------------
# 1. 路径与配置
# --------------------------------------------------
DATA_DIR   = Paths.DATA_DIR              # 可按需修改
INPUT_CSV  = os.path.join(DATA_DIR,"interpolation_top13.csv")
OUTPUT_CSV = os.path.join(DATA_DIR,"panel_1958_2019.csv")

COUNTRIES = NamesAndCodes.country_codes
YEAR_START, YEAR_END = 1958, 2019

# Use SF from data.selected_features
features = selected_features.SF

INDICATOR_MAP = {
    "NE.IMP.GNFS.ZS" : "IMP",      # Imports (% of GDP)
    "NE.EXP.GNFS.ZS" : "EXP",      # Exports (% of GDP)
    "NY.GDP.MKTP.KD.ZG": "GDP_GR", # GDP growth (annual %)
}
COLS_NEED = ["country_code", "year"] + list(INDICATOR_MAP.keys())

df = pd.read_csv(INPUT_CSV)

# 识别特征列（排除 country_code 和 year）
feature_columns = selected_features.SF

df = df[(df['year'] >= 1988) & (df['year'] <= 2019)]
# 将宽格式转换为长格式
panel = df.melt(
    id_vars=['country_code', 'year'],
    value_vars=feature_columns,
    var_name='feature_code',
    value_name='value'
)

# 保存为 CSV
panel.to_csv(os.path.join(DATA_DIR, "panel_1988_2019.csv"), index=False)