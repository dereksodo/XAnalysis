import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error

def compute_metrics(y_true, y_pred, y_naive=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    if np.isclose(np.std(y_true), 0):
        if np.isclose(np.std(y_pred), 0):
            rmse_std, rmse, std, mase = 0.0, 0.0, 0.0, 0.0
            r2 = 1.0
            da = 0.5
        else:
            rmse_std, rmse, std, mase = 9.99, 9.99, 9.99, 9.99
            r2, da = -1.0, 0.5
        return {
                "rmse": rmse,
                "std": std,
                "rmse/std": rmse_std,
                "r2": r2,
                "mase": mase,
                "da": da
            }

    rmse = root_mean_squared_error(y_true, y_pred)
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
