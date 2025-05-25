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
        ans = 0
        for j in range(len(df)):
            alpha1 = 1 if df.iloc[j,3] < 1 else 0
            alpha2 = 1 if df.iloc[j,4] > 0.6 else 0
            alpha3 = 1 if df.iloc[j,5] < 1 else 0
            alpha4 = 1 if df.iloc[j,6] > 0.7 else 0
            ans += 1 if alpha1 + alpha2 + alpha3 + alpha4 >= 3 else 0
        data.append({
            "Period": period.replace("_", "-"),
            "Model": model,
            "Feasibility Rate": ans / len(df)
        })

df_all = pd.DataFrame(data)

# --- Single Feasibility Rate Line Plot with Two Periods ---
fig, ax = plt.subplots(figsize=(10, 5))

for period in ['1960-2020', '2010-2020']:
    df_period = df_all[df_all["Period"] == period].sort_values(by="Feasibility Rate")
    ax.plot(df_period["Model"], df_period["Feasibility Rate"], marker='o', label=period)

ax.set_xticks(range(len(df_period)))
ax.set_xticklabels(df_period["Model"], rotation=45)
ax.set_ylim(0, 1.0)
ax.axhline(0.75, color='r', linestyle='--', label='Threshold = 0.75')
ax.set_title("Feasibility Rate Sorted by Model")
ax.set_ylabel("Feasibility Rate")
ax.set_xlabel("Model (sorted by rate)")
ax.legend()

fig.tight_layout()
plt.savefig(os.path.join(results_dir, "Feasibility_Rate_Line_Overlay_figure3.png"), dpi=200)
plt.show()