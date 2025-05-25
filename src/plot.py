import matplotlib.pyplot as plt

def plot_metrics(df_res, start_year, end_year, model_name, save_dir):
    plt.figure(figsize=(14, 5))

    # 1. RMSE/STD
    plt.subplot(1, 2, 1)
    plt.bar(df_res["target"], df_res["rmse/std"])
    plt.xlabel('Features')
    plt.ylabel('RMSE/STD')
    plt.title(f'{start_year}-{end_year} {model_name}  RMSE/STD')
    plt.axhline(1.0, color='r', linestyle='--', label='Feasible region threshold')
    plt.xticks(rotation=45, ha='right')
    plt.legend()

    # 2. R^2
    plt.subplot(1, 2, 2)
    plt.bar(df_res["target"], df_res["r2"])
    plt.xlabel('Features')
    plt.ylabel('R²')
    plt.title(f'{start_year}-{end_year} {model_name}  R²')
    plt.axhline(0.6, color='g', linestyle='--', label='Feasible region threshold')
    plt.ylim(-1, 1)
    plt.xticks(rotation=45, ha='right')
    plt.legend()

    plt.tight_layout()
    fname = f"{save_dir}/{start_year}_{end_year}_{model_name}.png"
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {fname}")

    # 3. MASE
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.bar(df_res["target"], df_res["mase"])
    plt.xlabel('Features')
    plt.ylabel('MASE')
    plt.title(f'{start_year}-{end_year} {model_name}  MASE')
    plt.axhline(1.0, color='r', linestyle='--', label='MASE = 1')
    plt.ylim(0, 4.0)
    plt.xticks(rotation=45, ha='right')
    plt.legend()

    # 4. Directional Accuracy (DA)
    plt.subplot(1, 2, 2)
    plt.bar(df_res["target"], df_res["da"])
    plt.xlabel('Features')
    plt.ylabel('Directional Accuracy')
    plt.title(f'{start_year}-{end_year} {model_name}  Directional Accuracy')
    plt.axhline(0.7, color='g', linestyle='--', label='DA = 0.7')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.legend()

    plt.tight_layout()
    fname2 = f"{save_dir}/{start_year}_{end_year}_{model_name}_mase_da.png"
    plt.savefig(fname2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {fname2}")
