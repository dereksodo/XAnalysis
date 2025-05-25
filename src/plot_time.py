


import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_prediction_vs_actual(y_true, y_pred, years, title, save_path):
    """
    Plot predicted vs actual time series for a given indicator.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(years, y_true, label='Actual', marker='o')
    plt.plot(years, y_pred, label='Predicted', marker='x')
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_model_comparison_bar(df_results, metric, save_path):
    """
    Plot comparison bar chart of model performance on a specific metric.
    Assumes df_results has columns: ['target', 'model', metric]
    """
    plt.figure(figsize=(12, 6))
    pivoted = df_results.pivot(index='target', columns='model', values=metric)
    pivoted.plot(kind='bar', figsize=(12, 6))
    plt.ylabel(metric)
    plt.title(f'Model Comparison by {metric}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()