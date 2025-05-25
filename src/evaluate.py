import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def compute_metrics(y_true, y_pred, y_naive=None):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    std = np.std(y_true)
    normed_rmse = rmse / std if std > 0 else float('nan')

    # MASE
    if y_naive is None:
        y_naive = np.roll(y_true, 1)
        y_naive[0] = y_true[0]
    mae_model = np.mean(np.abs(y_pred - y_true))
    mae_naive = np.mean(np.abs(y_true[1:] - y_true[:-1])) if len(y_true) > 1 else float('nan')
    mase = mae_model / mae_naive if mae_naive != 0 else float('inf')

    # Directional Accuracy
    correct_direction = np.sum(
        np.sign(y_pred[1:] - y_pred[:-1]) == np.sign(y_true[1:] - y_true[:-1])
    )
    da = correct_direction / (len(y_true) - 1) if len(y_true) > 1 else float('nan')

    return {
        "rmse": rmse,
        "std": std,
        "rmse/std": normed_rmse,
        "r2": r2,
        "mase": mase,
        "da": da
    }
