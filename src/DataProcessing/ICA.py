import pandas as pd
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import os

# 路径配置，请按实际路径修改
from src.DataProcessing.utils import Paths  # 如果你项目中有 Paths.py 管理路径

def run_ica():
    # 加载数据
    data_path = os.path.join(Paths.DATA_DIR, "interpolation_top13.csv")  # 请替换为你的标准化数据文件名
    df = pd.read_csv(data_path, index_col=0)

    # ICA 要求无缺失值、均值为0的数据
    X = df.drop(columns=["country_code", "year"]).values
    feature_names = df.drop(columns=["country_code", "year"]).columns

    # 标准化：均值为0，方差为1
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 运行 ICA（默认提取与特征数相同数量的独立分量）
    n_components = min(X.shape[0], X.shape[1])
    ica = FastICA(n_components=n_components, random_state=42)
    S_ = ica.fit_transform(X)   # 独立成分
    A_ = ica.mixing_            # 混合矩阵

    # 保存独立成分
    S_df = pd.DataFrame(S_, columns=[f"IC{i+1}" for i in range(S_.shape[1])])
    S_path = os.path.join(Paths.DATA_DIR, "ica_independent_components.csv")
    S_df.to_csv(S_path, index=False)

    # 保存混合矩阵（变量在行）
    A_df = pd.DataFrame(A_, index=feature_names, columns=[f"IC{i+1}" for i in range(A_.shape[1])])
    A_path = os.path.join(Paths.DATA_DIR, "ica_mixing_matrix.csv")
    A_df.to_csv(A_path)

    print(f"ICA finished. Independent components saved to:\n{S_path}\nMixing matrix saved to:\n{A_path}")

if __name__ == "__main__":
    run_ica()