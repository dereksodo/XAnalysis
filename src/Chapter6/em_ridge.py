import numpy as np
from sklearn.linear_model import Ridge
from scipy.stats import multivariate_normal

def em_gmm_ridge(X_self, X_cross, Y, lambda_A=1.0, lambda_M=1.0, max_iter=100, tol=1e-4, verbose=True):
    """
    Perform EM-Ridge with K=2 (latent clusters: traditional vs export-oriented).
    
    Inputs:
        X_self: (T, p) self-lagged features
        X_cross: (T, q) cross-country features
        Y: (T, d) target variables
    Returns:
        A_list, B_list: list of A^k and B^k for each of the 2 clusters
        pi: prior probabilities
        gamma: soft assignment (T, 2)
    """
    T, d = Y.shape
    K = 2  # Two latent clusters

    # Initialization
    A = [np.random.randn(d, X_self.shape[1]) * 0.1 for _ in range(K)]
    B = [np.random.randn(d, X_cross.shape[1]) * 0.1 for _ in range(K)]
    pi = np.full(K, 1 / K)
    sigma2 = 1.0  # shared isotropic noise
    gamma = np.full((T, K), 1 / K)

    for it in range(max_iter):
        # E-step: compute gamma
        for k in range(K):
            mu = X_self @ A[k].T + X_cross @ B[k].T
            gamma[:, k] = pi[k] * multivariate_normal.pdf(Y, mean=mu, cov=sigma2 * np.eye(d))
        gamma /= gamma.sum(axis=1, keepdims=True)

        Nk = gamma.sum(axis=0)
        pi = Nk / T

        # Save previous for convergence check
        A_prev = [Ak.copy() for Ak in A]
        B_prev = [Bk.copy() for Bk in B]

        # M-step: weighted ridge regression
        for k in range(K):
            # Update A^k
            yA = Y - X_cross @ B[k].T
            model_A = Ridge(alpha=lambda_A, fit_intercept=False)
            model_A.fit(X_self, yA, sample_weight=gamma[:, k])
            A[k] = model_A.coef_

            # Update B^k
            yB = Y - X_self @ A[k].T
            model_B = Ridge(alpha=lambda_M, fit_intercept=False)
            model_B.fit(X_cross, yB, sample_weight=gamma[:, k])
            B[k] = model_B.coef_

        max_change = max(
            max(np.linalg.norm(A[k] - A_prev[k]), np.linalg.norm(B[k] - B_prev[k])) for k in range(K)
        )
        if verbose:
            print(f"[EM iter {it}] max param change: {max_change:.6f}")
        if max_change < tol:
            break

    return A, B, pi, gamma