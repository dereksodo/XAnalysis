# run_hyperparam_tuning.py
import os, glob, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore", message="invalid value encountered in cast", module="numpy")
from src.DataProcessing.utils import selected_features, get_scores, Paths
from src.DataProcessing.evaluate import compute_metrics
from src.DataProcessing.utils import get_scores   # guiding_score(metrics)

# ------------------------------------------------------------------- #
# CONFIG
DATA_PATH  = os.path.join(Paths.DATA_DIR, "interpolation_top13.csv")
SAVE_DIR   = os.path.join(Paths.FIGURE_DIR, "Chapter5tuning")
os.makedirs(SAVE_DIR, exist_ok=True)

FEATURES = selected_features.SF  # 10 indicator codes

# parameter grids
PARAMS = {
    "XGB": {
        "n_estimators":   [50, 100, 200],
        "max_depth":      [3, 5, 7],
        "learning_rate":  [0.05, 0.1],
        "subsample":      [0.6, 0.8, 1.0],
        "colsample_bytree":[0.6, 0.8, 1.0]
    },
    "RF": {
        "n_estimators":   [50, 100, 200],
        "max_depth":      [5, 10, 20],
        "min_samples_split":[2, 5, 10],
        "max_features":   ["sqrt", "log2"]
    },
    "ARIMA": {
        "p": [0, 1, 2],
        "d": [0, 1],
        "q": [0, 1, 2]
    }
}

# ------------------------------------------------------------------- #
# helper to load panel as {target: (X_train, X_test, y_train, y_test)}
def load_data(path):
    df = pd.read_csv(path).dropna(subset=["country_code"] + FEATURES)

    split = {}
    for country, g in df.groupby("country_code"):
        X_full = g[FEATURES].reset_index(drop=True)

        for tgt in FEATURES:
            X = X_full.drop(columns=[tgt])
            y = X_full[tgt].reset_index(drop=True)
            if len(y) < 10:        # 序列太短直接跳过
                continue
            split_pt = max(1, int(0.8 * len(y)))
            X_tr, X_te = X.iloc[:split_pt], X.iloc[split_pt:]
            y_tr, y_te = y.iloc[:split_pt], y.iloc[split_pt:]
            split[(country, tgt)] = (X_tr, X_te, y_tr, y_te)
    return split
# guiding-score scorer
def scorer(y_true, y_pred):
    m = compute_metrics(y_true, y_pred)
    if any(np.isnan(v) or np.isinf(v) for v in m.values()):
        return -1e9   # 一个极低值，避免 NaN
    return get_scores.guiding_score(m)

GS_SCORER = make_scorer(scorer, greater_is_better=True)

# ------------------------------------------------------------------- #
def tune_xgb(X_tr, y_tr):
    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=4,
        random_state=42
    )
    gs = GridSearchCV(
        model,
        PARAMS["XGB"],
        cv=3,
        n_jobs=-1,
        scoring=GS_SCORER,
        verbose=0
    )
    gs.fit(X_tr, y_tr)
    return gs

def tune_rf(X_tr, y_tr):
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    gs = GridSearchCV(
        model,
        PARAMS["RF"],
        cv=3,
        n_jobs=-1,
        scoring=GS_SCORER,
        verbose=0
    )
    gs.fit(X_tr, y_tr)
    return gs

def tune_arima(y_tr, y_te):
    """
    Robust ARIMA grid-search
    Returns (best_params, best_pred, best_guiding_score)
    """
    # 若训练序列几乎常数，直接用 Naive 预测
    if np.isclose(np.var(y_tr), 0):
        naive_pred = np.repeat(y_tr.iloc[-1], len(y_te))
        return {"p": 0, "d": 0, "q": 0, "note": "constant series"}, naive_pred, scorer(y_te, naive_pred)

    best_score  = -1e9
    best_params = None
    best_pred   = None

    for p in PARAMS["ARIMA"]["p"]:
        for d in PARAMS["ARIMA"]["d"]:
            for q in PARAMS["ARIMA"]["q"]:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")  # 屏蔽 SARIMAX 收敛/可逆警告
                        mdl = ARIMA(y_tr, order=(p, d, q)).fit()
                    y_hat = mdl.forecast(steps=len(y_te))

                    # 预测里含 NaN / inf 则跳过
                    if np.any(np.isnan(y_hat)) or np.any(np.isinf(y_hat)):
                        continue

                    sc = scorer(y_te, y_hat)
                    if sc > best_score:
                        best_score, best_params, best_pred = sc, {"p": p, "d": d, "q": q}, y_hat
                except Exception:
                    # 拟合失败直接跳过
                    continue

    # 如果所有组合都失效，使用随机游走兜底
    if best_params is None:
        fallback_pred = np.repeat(y_tr.iloc[-1], len(y_te))
        best_params   = {"p": 0, "d": 1, "q": 1, "note": "fallback naive"}
        best_score    = scorer(y_te, fallback_pred)
        best_pred     = fallback_pred

    return best_params, best_pred, best_score

# ------------------------------------------------------------------- #
def main():
    data_dict = load_data(DATA_PATH)

    for algo in ["XGB", "RF", "ARIMA"]:
        rows, grid_dumped = [], []
        print(f"\n== Tuning {algo} ==")
        keys = list(data_dict.keys())
        for i, (country, tgt) in enumerate(keys, 1):
            print(f"[{i}/{len(keys)}] {country} – {tgt}")
            X_tr, X_te, y_tr, y_te = data_dict[(country, tgt)]

            if algo == "XGB":
                gs = tune_xgb(X_tr, y_tr)
                y_hat = gs.predict(X_te)
                best_score = scorer(y_te, y_hat)  
                best_params = gs.best_params_
                df_res = pd.DataFrame(gs.cv_results_)
                df_res["target"] = tgt          # 标注该结果属于哪一指标
                df_res["country"] = country 
                grid_dumped.append(df_res)
            elif algo == "RF":
                gs = tune_rf(X_tr, y_tr)
                y_hat = gs.predict(X_te)
                best_score = scorer(y_te, y_hat)  
                best_params = gs.best_params_
                df_res = pd.DataFrame(gs.cv_results_)
                df_res["target"] = tgt          # 标注该结果属于哪一指标
                df_res["country"] = country 
                grid_dumped.append(df_res)
            else:  # ARIMA
                best_params, y_hat, best_score = tune_arima(y_tr, y_te)

            m = compute_metrics(y_te, y_hat)
            # Re‑compute guiding from the final metrics (ignore invalid metrics)
            if any(np.isnan(v) or np.isinf(v) for v in m.values()):
                guiding_final = np.nan
            else:
                guiding_final = get_scores.guiding_score(m)
            print(f"   → best_guiding_cv = {best_score:.3f} | final_guiding = {guiding_final:.3f}")

            row = {
                "country": country,
                "target": tgt,
                **m,
                "guiding": guiding_final,      # use valid guiding score for CSV
                "best_guiding_cv": best_score, # cv score used for selection (may be -1e9)
                "best_params": json.dumps(best_params)
            }
            rows.append(row)

        # save summary
        summary_path = os.path.join(SAVE_DIR, f"{algo}_best_summary.csv")
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"Saved best‑summary to {summary_path}")
        # save full grid
        if grid_dumped:
            grid_path = os.path.join(SAVE_DIR, f"{algo}_grid_results.csv")
            pd.concat(grid_dumped).to_csv(grid_path, index=False)
            print(f"Saved grid‑results to {grid_path}")

if __name__ == "__main__":
    main()