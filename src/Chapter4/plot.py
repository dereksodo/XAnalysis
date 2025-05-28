def plot_metrics(df_res, start_year, end_year, model_name, save_dir):
    import os
    import matplotlib.pyplot as plt
    from src.DataProcessing.utils import FeatureNames

    plt.figure(figsize=(16, 10))  # 创建一个大的图像

    # 1. RMSE/STD
    plt.subplot(2, 2, 1)
    labels = df_res["target"].map(FeatureNames.code2name)
    plt.bar(labels, df_res["rmse/std"])
    plt.xlabel('Features')
    plt.ylabel('RMSE/STD')
    plt.title(f'{start_year}-{end_year} {model_name}  RMSE/STD')
    plt.axhline(1.0, color='r', linestyle='--', label='Feasible region threshold')
    plt.xticks(rotation=45, ha='right')
    plt.legend()

    # 2. R^2
    plt.subplot(2, 2, 2)
    labels = df_res["target"].map(FeatureNames.code2name)
    plt.bar(labels, df_res["r2"])
    plt.xlabel('Features')
    plt.ylabel('R²')
    plt.title(f'{start_year}-{end_year} {model_name}  R²')
    plt.axhline(0.6, color='g', linestyle='--', label='Feasible region threshold')
    plt.ylim(-1, 1)
    plt.xticks(rotation=45, ha='right')
    plt.legend()

    # 3. MASE
    plt.subplot(2, 2, 3)
    labels = df_res["target"].map(FeatureNames.code2name)
    plt.bar(labels, df_res["mase"])
    plt.xlabel('Features')
    plt.ylabel('MASE')
    plt.title(f'{start_year}-{end_year} {model_name}  MASE')
    plt.axhline(1.0, color='r', linestyle='--', label='MASE = 1')
    plt.ylim(0, 4.0)
    plt.xticks(rotation=45, ha='right')
    plt.legend()

    # 4. Directional Accuracy (DA)
    plt.subplot(2, 2, 4)
    labels = df_res["target"].map(FeatureNames.code2name)
    plt.bar(labels, df_res["da"])
    plt.xlabel('Features')
    plt.ylabel('Directional Accuracy')
    plt.title(f'{start_year}-{end_year} {model_name}  Directional Accuracy')
    plt.axhline(0.7, color='g', linestyle='--', label='DA = 0.7')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.legend()

    plt.tight_layout()
    fname = f"{save_dir}/{start_year}_{end_year}_{model_name}_metrics.png"
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {fname}")