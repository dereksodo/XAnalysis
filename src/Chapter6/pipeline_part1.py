import numpy as np
import pandas as pd
import pymc as pm
import numpy as np
import os
from src.DataProcessing.utils import Paths, get_scores, selected_features, FeatureNames, NamesAndCodes
from src.DataProcessing.evaluate import compute_metrics

LOG_TRANSFORM_FEATURES = ["EG.USE.PCAP.KG.OE", "NY.GDP.MKTP.CD", "EN.ATM.GHGT.KT.CE"]

# --- Load and preprocess data ---
df_panel = pd.read_csv(os.path.join(Paths.DATA_DIR, 'panel_1988_2019.csv'))
df_panel_raw = df_panel.copy()
countries = NamesAndCodes.country_codes
features_full = selected_features.SF
features_cross = ["NY.GDP.MKTP.CD", "NE.EXP.GNFS.ZS", "NE.IMP.GNFS.ZS"]

feature_stats = {}
for f in features_full:
    mask = df_panel['feature_code'] == f
    values = df_panel.loc[mask, 'value'].copy()
    if f in LOG_TRANSFORM_FEATURES:
        values = np.log(values)  # 避免 log(0)
    mean_f = values.mean()
    std_f = values.std()
    feature_stats[f] = (mean_f, std_f)
    df_panel.loc[mask, 'value'] = (values - mean_f) / std_f


years = df_panel["year"].sort_values().unique()
df_wide = df_panel.pivot(index=["year", "country_code"], columns="feature_code", values="value")
df_wide = df_wide[features_full].dropna()

# 恢复 year, country_code 为列，方便后续 reshape
df_wide = df_wide.reset_index()

# 构造面板张量
X_full = df_wide.set_index(["year", "country_code"]).unstack("country_code").reindex(countries, axis=1, level=1).sort_index().to_numpy().reshape(len(df_wide["year"].unique()), len(countries), len(features_full))

X_cross = df_wide.set_index(["year", "country_code"])[features_cross].unstack("country_code").reindex(countries, axis=1, level=1).sort_index().to_numpy().reshape(len(df_wide["year"].unique()), len(countries), len(features_cross))

T, C, F_full = X_full.shape
F_cross = len(features_cross)
lags = 2

X_full_lagged = np.stack([
    X_full[lags - 1: -1],   # 对应 t-1
    X_full[lags - 2: -2],   # 对应 t-2
], axis=2)

X_cross_lagged = np.stack([
    X_cross[lags - 1: -1]
], axis=2)

Y = X_full[lags:]
T_eff = Y.shape[0]

# --- Define M matrix generator ---
def get_M_matrix(structure: str, countries: list, features_cross: list) -> np.ndarray:
    C = len(countries)
    F = len(features_cross)
    if structure == "uniform":
        M = np.ones((C, C, F)) / (C - 1)
        for f in range(F):
            np.fill_diagonal(M[:, :, f], 0)
        return M
    elif structure == "identity":
        return np.stack([np.eye(C) for _ in range(F)], axis=-1)
    elif structure == "zero":
        return np.zeros((C, C, F))
    else:
        raise ValueError(f"Unknown M structure: {structure}")

# --- Train Bayesian model ---
def train_bayesian_model(data, M_matrix, sigma_prior="Exponential", noise_scale=0.1, M_structure="custom"):
    
    
    """
    Optimized Bayesian GVAR model with clear logging and efficient cross-part calculation.
    """
    X_full_lagged, X_cross_lagged, Y, features_full, features_cross, countries, feature_stats = data
    #print(f"[INFO] Training Bayesian model with M_matrix shape: {M_matrix.shape}")
    #print(f"[INFO] Using features: {features_full}")
    T_eff, C, L, F_full = X_full_lagged.shape
    F_cross = len(features_cross)
    with pm.Model() as model:
        mu_a = pm.Normal("mu_a", mu=0, sigma=1, shape=F_full)
        sigma_a = pm.Exponential("sigma_a", lam=1.0, shape=F_full)
        a = pm.Normal("a", mu=mu_a, sigma=sigma_a, shape=(C, F_full))

        if sigma_prior == "Exponential":
            sigma = pm.Exponential("sigma", lam=1.0, shape=F_full)
        else:
            sigma = pm.HalfNormal("sigma", sigma=1.0, shape=F_full)

        x_tm1_self = X_full_lagged[:, :, 0, :]
        x_tm2_self = X_full_lagged[:, :, 1, :]
        x_tm1_cross = X_cross_lagged[:, :, 0, :]

        self_part = a * x_tm1_self + a**2 * x_tm2_self
        cross_part = np.zeros((T_eff, C, F_full))
        #print("[LOG] Starting cross-part calculation...")
        for f_idx, f in enumerate(features_cross):
            x_cross = x_tm1_cross[:, :, f_idx]  # shape: (T, C)
            M_f = M_matrix[:, :, f_idx]         # shape: (C, C)
            #print("x_corss.shape = ", x_cross.shape)
            #print("M_f.shape = ", M_f.shape)
            # Compute M_f.T @ x_cross.T then transpose back
            cross_sum = x_cross @ M_f.T  # shape: (T, C)
            #print("hi1")
            full_idx = features_full.index(f)
            #print("hi2")
            #print("cross_part.shape = ", cross_part.shape)
            #print("cross_sum.shape = ", cross_sum.shape)
            cross_part[:, :, full_idx] = cross_sum
            #print("hi3")
        #print("[LOG] Here1")
        mu = pm.Deterministic("mu", self_part)#delete cross_part
        #print("[LOG] Here2")
        sigma_broadcast = pm.Deterministic("sigma_broadcast", sigma * noise_scale)
        #print("[LOG] Here3")
        Y_tensor = pm.Data("Y_data", Y.astype("float32"))
        pm.Normal("Y_obs", mu=mu, sigma=sigma_broadcast.dimshuffle("x", "x", 0), observed=Y_tensor)

        #print("[INFO] Sampling started...")
        trace = pm.sample(draws=1000, tune=1000, chains=6, cores=6, target_accept=0.95)
        #print("[LOG] Sampling completed.")
        mu_post = trace.posterior["mu"].mean(("chain", "draw")).values
        #print("[INFO] Posterior mean calculated.")

        mu_orig = mu_post.copy()
        years_eff = df_wide["year"].unique()[lags:]
        a_mean = trace.posterior["a"].mean(("chain", "draw")).values
        print("[DEBUG] a_mean:", a_mean)
        # 保存预测与真实值为 CSV 文件（反标准化）
        '''
        result_rows = []
        for t in range(mu_orig.shape[0]):
            for c in range(mu_orig.shape[1]):
                for j in range(mu_orig.shape[2]):
                    feature = features_full[j]
                    mean, std = feature_stats[feature]
                    mask = (
                        (df_panel_raw["year"] == years_eff[t]) &
                        (df_panel_raw["country_code"] == countries[c]) &
                        (df_panel_raw["feature_code"] == feature)
                    )
                    y_true = df_panel_raw.loc[mask, "value"].values[0]
                    y_pred = mu_orig[t, c, j] * std + mean
                    if feature in LOG_TRANSFORM_FEATURES:
                        y_pred = np.exp(y_pred)
                    result_rows.append({
                        "year": years_eff[t],
                        "country": countries[c],
                        "feature": feature,
                        "y_true": y_true,
                        "y_pred": y_pred
                    })'''
        def inverse_standardize(y, mean, std):
            return y * std + mean

        result_rows = []
        for j, feature in enumerate(features_full):
            mean, std = feature_stats[feature]
            for c, country in enumerate(countries):
                for t in range(mu_orig.shape[0]):
                    mask = (
                        (df_panel_raw["year"] == years_eff[t]) &
                        (df_panel_raw["country_code"] == countries[c]) &
                        (df_panel_raw["feature_code"] == feature)
                    )
                    y_true = df_panel_raw.loc[mask, "value"].values[0]
                    y_pred = mu_orig[t, c, j] * std + mean
                    if feature in LOG_TRANSFORM_FEATURES:
                        y_pred = np.exp(y_pred)
                    if y_true >= 0 and y_pred < 0:
                        print(f"[ALERT] {feature} y_true and y_pred WRONG!!! year={years_eff[t]}, country={country}, Y_norm={Y[t, c, j]:.3f}, mean={mean:.3e}, std={std:.3e}")
                        print(f"{y_true:.3f} vs {y_pred:.3f}")
                    result_rows.append({
                        "year": years_eff[t],
                        "country": country,
                        "feature": feature,
                        "y_true": y_true,
                        "y_pred": y_pred
                    })
                    
        df_result = pd.DataFrame(result_rows)
        csv_path = os.path.join(Paths.FIGURE_DIR,"Chapter6", f"bayesian_prediction_{M_structure}.csv")
        df_result.to_csv(csv_path, index=False)
        #print(f"[INFO] Saved prediction results to {csv_path}")

        #print(f"[INFO] Posterior mean shape: {mu_post.shape}")
        return trace, (mu_orig, Y)

# --- Evaluate ---
def evaluate_model(predictions, panel_data):
    mu_orig, Y_orig = predictions
    # Unpack as X_full_lagged, X_cross_lagged, Y, features, features_cross, countries, feature_stats
    X_full_lagged, X_cross_lagged, Y, features, features_cross, countries, feature_stats = panel_data
    # Ensure LOG_TRANSFORM_FEATURES is available in scope
    global LOG_TRANSFORM_FEATURES
    print("[INFO] Starting model evaluation using guiding score...")
    gs_total, n = 0, 0
    for j in range(len(features)):
        feature = features[j]
        mean, std = feature_stats[feature]
        for i in range(len(countries)):
            y_true = Y_orig[:, i, j] * std + mean
            y_pred = mu_orig[:, i, j] * std + mean
            if feature in LOG_TRANSFORM_FEATURES:
                y_true = np.exp(y_true)
                y_pred = np.exp(y_pred)
            metrics = compute_metrics(y_true, y_pred)
            gscore = get_scores.guiding_score(metrics)
            #print(f"[DEBUG] Country: {countries[i]}, Feature: {features[j]}, Metrics: {metrics}, Guiding Score: {gscore:.4f}")
            gs_total += gscore
            n += 1
    #print(f"[INFO] Completed evaluation. Average guiding score: {gs_total / n:.4f}")
    return gs_total / n

# --- Run ---
panel_data = (X_full_lagged, X_cross_lagged, Y, features_full, features_cross, countries, feature_stats)
M_structures = ["uniform", "zero", "identity"]

for M_structure in M_structures:
    print(f"[INFO] ==========================")
    print(f"[INFO] M_structure = {M_structure}")
    print(f"[INFO] ==========================")
    try:
        M_matrix = get_M_matrix(M_structure, countries, features_cross)
        print(f"[INFO] M_matrix summary: mean={np.mean(M_matrix):.4f}, std={np.std(M_matrix):.4f}, min={np.min(M_matrix):.4f}, max={np.max(M_matrix):.4f}")
        trace, predictions = train_bayesian_model(
            data=panel_data,
            M_matrix=M_matrix,
            sigma_prior="Exponential",
            noise_scale=0.1,
            M_structure=M_structure
        )
        guiding_score = evaluate_model(predictions, panel_data)
        print(f">>> Guiding score: {guiding_score:.3f}")
    except Exception as e:
        print(f"[Error] Skipped structure {M_structure}: {e}")