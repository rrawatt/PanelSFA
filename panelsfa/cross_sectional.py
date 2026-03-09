"""
panelsfa/cross_sectional.py
============================
Aigner, Lovell & Schmidt (1977) – Cross-Sectional SFA

Model
-----
    y_i = x_i β + v_i − s·u_i
    v_i ~ N(0, σ_v²)
    u_i ~ |N(0, σ_u²)|   (Half-Normal)

where s = +1 (production) or s = −1 (cost).

Parameterisation
----------------
    σ² = σ_v² + σ_u²
    γ  = σ_u² / σ²

Optimiser vector θ = [β₁…βₖ,  ln(σ²),  logit(γ)]

Log-likelihood (per observation, ALS eq. 4)
--------------------------------------------
    ℓ_i = ln(2) − ln(σ) + ln φ(s·ε_i/σ) + ln Φ(s·ε_i·λ/σ)

where λ = σ_u / σ_v.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from sklearn.utils.validation import check_X_y, check_is_fitted

from .base import (
    _BaseSFA,
    log_to_sigma_sq, logit_to_gamma,
    sigma_sq_to_log, gamma_to_logit,
)


# ---------------------------------------------------------------------------
# Log-likelihood kernel – pure NumPy/SciPy, no JIT
# ---------------------------------------------------------------------------

def _nll_als(theta, X, y, s):
    """
    Negative log-likelihood for ALS 1977 (Half-Normal).

    Parameters
    ----------
    theta : 1-D array  [β (k), ln σ², logit γ]
    X     : (n, k)
    y     : (n,)
    s     : +1 (production) / −1 (cost)

    Returns
    -------
    nll : float
    """
    k        = X.shape[1]
    beta     = theta[:k]
    sigma_sq = log_to_sigma_sq(theta[k])
    gamma    = logit_to_gamma(theta[k + 1])

    sigma_u_sq = gamma * sigma_sq
    sigma_v_sq = (1.0 - gamma) * sigma_sq
    sigma      = np.sqrt(sigma_sq)
    lam        = np.sqrt(sigma_u_sq / sigma_v_sq)   # λ = σ_u / σ_v

    eps = y - X @ beta          
    
    # We must invert the sign for the CDF: production (s=1) needs negative skew
    ll = (
        np.log(2.0) - np.log(sigma) 
        + norm.logpdf(eps / sigma) 
        + norm.logcdf(-s * eps * lam / sigma)
    )
    return -np.sum(ll)


# ---------------------------------------------------------------------------
# Estimator class
# ---------------------------------------------------------------------------

class CrossSectionalSFA(_BaseSFA):
    """
    Aigner, Lovell & Schmidt (1977) Cross-Sectional SFA.

    Parameters
    ----------
    model_type : {'production', 'cost'}, default 'production'
        's = +1' for production (y − Xβ = v − u),
        's = −1' for cost       (y − Xβ = v + u).
    max_iter : int, default 2000
        Maximum iterations for L-BFGS-B.
    tol : float, default 1e-8
        Function-value tolerance for convergence.

    Fitted attributes
    -----------------
    coef_           : ndarray (k,)  frontier coefficients β
    sigma_sq_       : float         total variance σ² = σ_v² + σ_u²
    gamma_          : float         variance ratio γ = σ_u²/σ²
    log_likelihood_ : float         maximised log-likelihood
    """

    def __init__(self, model_type: str = "production",
                 max_iter: int = 2000, tol: float = 1e-8):
        super().__init__(model_type=model_type, max_iter=max_iter, tol=tol)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Estimate ALS 1977 model via Maximum Likelihood.

        Parameters
        ----------
        X : array-like of shape (n, k)
            Frontier regressors. Include an intercept column explicitly.
        y : array-like of shape (n,)
            Output (production) or cost variable.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y, y_numeric=True)
        self._n_obs_ = len(y)
        s = self._get_s()

        # --- OLS warm-start -------------------------------------------
        beta0, sig2_0 = self._ols_init(X, y)
        theta0 = np.concatenate([
            beta0,
            [sigma_sq_to_log(sig2_0),
             gamma_to_logit(0.5)],
        ])
        self._n_params_ = len(theta0)

        # --- MLE via L-BFGS-B ----------------------------------------
        result = minimize(
            _nll_als,
            theta0,
            args=(X, y, s),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )
        if not result.success:
            import warnings
            warnings.warn(f"L-BFGS-B did not converge: {result.message}")

        # --- Store fitted attributes (trailing-underscore convention) --
        k = X.shape[1]
        params = self._unpack_theta(result.x, k)
        self.coef_           = params["beta"]
        self.sigma_sq_       = params["sigma_sq"]
        self.gamma_          = params["gamma"]
        self.log_likelihood_ = -result.fun
        self._theta_opt_     = result.x

        return self

    # ------------------------------------------------------------------
    # Parameter unpacking
    # ------------------------------------------------------------------

    def _unpack_theta(self, theta, k):
        """Transform optimiser vector → natural-scale parameter dict."""
        return {
            "beta"     : theta[:k],
            "sigma_sq" : log_to_sigma_sq(theta[k]),
            "gamma"    : logit_to_gamma(theta[k + 1]),
        }

    def _log_likelihood(self, theta, X, y):
        return _nll_als(theta, X, y, self._get_s())

    # ------------------------------------------------------------------
    # JLMS estimator  E[u_i | ε_i]   (Jondrow et al. 1982)
    # ------------------------------------------------------------------

    def _jlms(self, X, y):
        """
        Compute E[u_i | ε_i] via the JLMS formula for Half-Normal.

        JLMS (1982) eq. (9):
            μ*_i  = −s·ε_i · σ_u² / σ²
            σ*²   = σ_v² · σ_u² / σ²
            E[u|ε] = σ* · [ φ(μ*/σ*) / Φ(μ*/σ*) + μ*/σ* ]

        The `s` sign correctly orients the conditional mean for both
        production (s = +1) and cost (s = −1) frontiers.

        Returns
        -------
        e_u : ndarray (n,)  non-negative firm-level inefficiency estimates
        """
        check_is_fitted(self)
        k      = X.shape[1]
        params = self._unpack_theta(self._theta_opt_, k)

        beta       = params["beta"]
        sigma_sq   = params["sigma_sq"]
        gamma      = params["gamma"]
        s          = self._get_s()

        sigma_u_sq = gamma * sigma_sq
        sigma_v_sq = (1.0 - gamma) * sigma_sq
        sigma_star = np.sqrt(sigma_v_sq * sigma_u_sq / sigma_sq)

        eps     = y - X @ beta
        mu_star = -s * eps * sigma_u_sq / sigma_sq

        ratio = mu_star / sigma_star
        e_u   = sigma_star * (norm.pdf(ratio) / norm.cdf(ratio) + ratio)
        return np.clip(e_u, 0.0, None)
