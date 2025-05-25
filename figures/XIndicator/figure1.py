import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 路径设置
results_dir = "/Users/tianhaozhang/Desktop/XCountryOIRPrediction/figures/Xindicator"
periods = ['1960_2020', '2010_2020']
model_names = [
    'LR', 'LWR', 'SVR', 'Ridge', 'Lasso', 'ElasticNet', 'RF', 'KNN', 'XGB'
]

# 结果收集
data = []
for period in periods:
    for model in model_names:
        result_file = os.path.join(results_dir, f"{period}_{model}_results.csv")
        if not os.path.exists(result_file):
            continue
        df = pd.read_csv(result_file)
        mean_rmse_std = df["rmse/std"].mean()
        mean_r2 = df["r2"].mean()
        data.append({
            "Period": period.replace("_", "-"),
            "Model": model,
            "Average RMSE/STD": mean_rmse_std,
            "Average R2": mean_r2
        })

df_all = pd.DataFrame(data)

# 只画一张RMSE/STD和一张R2，x轴模型名，柱分period
x = np.arange(len(model_names))  # 模型序号
width = 0.35  # 柱宽

fig, ax = plt.subplots(figsize=(10, 5))
bar1 = ax.bar(x - width/2, [df_all.loc[(df_all["Period"] == '1960-2020') & (df_all["Model"] == m), "Average RMSE/STD"].values[0] if not df_all.loc[(df_all["Period"] == '1960-2020') & (df_all["Model"] == m), "Average RMSE/STD"].empty else np.nan for m in model_names], width, label='1960-2020')
bar2 = ax.bar(x + width/2, [df_all.loc[(df_all["Period"] == '2010-2020') & (df_all["Model"] == m), "Average RMSE/STD"].values[0] if not df_all.loc[(df_all["Period"] == '2010-2020') & (df_all["Model"] == m), "Average RMSE/STD"].empty else np.nan for m in model_names], width, label='2010-2020')
ax.axhline(1.0, color='r', linestyle='--', label='Feasible region threshold')
ax.set_ylabel("Average RMSE/STD")
ax.set_title("Average RMSE/STD for Each Model")
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=0, fontsize=10)
ax.legend()
fig.tight_layout()
plt.savefig(os.path.join(results_dir, "Average_RMSE_STD_figure1.png"), dpi=200)
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
bar1 = ax.bar(x - width/2, [df_all.loc[(df_all["Period"] == '1960-2020') & (df_all["Model"] == m), "Average R2"].values[0] if not df_all.loc[(df_all["Period"] == '1960-2020') & (df_all["Model"] == m), "Average R2"].empty else np.nan for m in model_names], width, label='1960-2020')
bar2 = ax.bar(x + width/2, [df_all.loc[(df_all["Period"] == '2010-2020') & (df_all["Model"] == m), "Average R2"].values[0] if not df_all.loc[(df_all["Period"] == '2010-2020') & (df_all["Model"] == m), "Average R2"].empty else np.nan for m in model_names], width, label='2010-2020')
ax.axhline(0.6, color='g', linestyle='--', label='Feasible region threshold')
ax.set_ylabel("Average $R^2$")
ax.set_title("Average $R^2$ for Each Model")
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=0, fontsize=10)
ax.legend()
fig.tight_layout()
plt.savefig(os.path.join(results_dir, "Average_R2_figure1.png"), dpi=200)
plt.show()