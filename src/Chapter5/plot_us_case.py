import pandas as pd
import matplotlib.pyplot as plt
import os

# 路径设置
metrics_path = "/Users/tianhaozhang/Desktop/XCountryOIRPrediction/results/time_series_metrics.csv"
output_dir = "/Users/tianhaozhang/Desktop/XCountryOIRPrediction/figures/time_series"
os.makedirs(output_dir, exist_ok=True)

# 读取并筛选中国数据
df = pd.read_csv(metrics_path)
df_china = df[df["country"] == "USA"]

# 指标和模型设置
metrics = ["rmse/std", "r2", "mase", "da"]
models = ["XGB", "RF", "ARIMA"]
indicators = df_china["target"].unique()

# 画图
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
axs = axs.flatten()

for i, metric in enumerate(metrics):
    pivot_df = df_china.pivot(index="target", columns="model", values=metric)
    pivot_df = pivot_df.loc[indicators, models]  # 固定顺序
    pivot_df.plot(kind="bar", ax=axs[i])
    axs[i].set_title(f"{metric}")
    axs[i].set_ylabel(metric)
    axs[i].set_xlabel("Indicator")
    axs[i].legend(title="Model")
    axs[i].grid(True)
    axs[i].tick_params(axis='x', rotation=45)

plt.suptitle("Forecasting Performance for US (Time Series Models)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# 保存
save_path = os.path.join(output_dir, "us_case_comparison.png")
plt.savefig(save_path, dpi = 300)
print(f"Save to: {save_path}")