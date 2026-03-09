"""
panelsfa/effects_panel.py
==========================
Battese & Coelli (1995) – Inefficiency Effects Panel SFA

Model
-----
    y_{it} = x_{it} β + v_{it} − s·u_{it}
    u_{it} ~ N(μ_{it}, σ_u²)⁺   (Truncated-Normal, observation-level mean)
    μ_{it} = δ Z_{it}            (linear inefficiency effects equation)

This is a *single-step* simultaneous MLE — no two-step bias.

Parameterisation
----------------
    σ²   = σ_v² + σ_u²,   γ = σ_u² / σ²

Optimiser vector θ = [β₁…βₖ,  ln(σ²),  logit(γ),  δ₁…δₘ]

BC95 Log-Likelihood Kernel (CORRECTION 1)
------------------------------------------
The exact observation-level marginal log-likelihood is:

    ℓ_{it} = −½ ln(2π)
             −½ ln(σ²)
             −½ · (ε_{it} + s·μ_{it})² / σ²
             + ln Φ(μ*_{it} / σ*)
             − ln Φ(μ_{it} / σ_u)

where:
    μ*_{it} = (μ_{it}·σ_v² − s·σ_u²·ε_{it}) / σ²
    σ*²     = σ_u²·σ_v² / σ²

This avoids the double-counting of the variance penalty that arises
from naively mixing norm.logpdf with an explicit −ε²/σ_v² term.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .base import (
    _BaseSFA,
    log_to_sigma_sq, logit_to_gamma,
    sigma_sq_to_log, gamma_to_logit,
)


# ---------------------------------------------------------------------------
# Log-likelihood kernel
# ---------------------------------------------------------------------------

def _nll_bc95(theta, X, y, Z, groups_mapped, s):
    """
    Negative log-likelihood for BC95 (observation-level Truncated-Normal).

    Parameters
    ----------
    theta         : [β (k), ln σ², logit γ, δ (m)]
    X             : (N, k)
    y             : (N,)
    Z             : (N, m)   inefficiency-effects regressors
    groups_mapped : (N,) int  contiguous 0-based entity codes (not used in
                              the LL sum but passed for API symmetry; BC95
                              likelihood sums independently at obs level)
    s             : +1 / −1

    Returns
    -------
    nll : float
    """
    k = X.shape[1]
    m = Z.shape[1]

    beta     = theta[:k]
    sigma_sq = log_to_sigma_sq(theta[k])
    gamma    = logit_to_gamma(theta[k + 1])
    delta    = theta[k + 2: k + 2 + m]

    sigma_u_sq = gamma * sigma_sq
    sigma_v_sq = (1.0 - gamma) * sigma_sq
    sigma_u    = np.sqrt(sigma_u_sq)
    sigma_star = np.sqrt(sigma_u_sq * sigma_v_sq / sigma_sq)  # scalar

    mu_it  = Z @ delta             # (N,)  observation-level inefficiency means
    eps_it = y - X @ beta          # (N,)  composite residuals

    # Posterior conditional mean of u_{it} | ε_{it}
    mu_star_it = (mu_it * sigma_v_sq - s * sigma_u_sq * eps_it) / sigma_sq  # (N,)

    # BC95 (1995) eq. (4) — CORRECTION 1: exact formulation, no double-counting
    ll = (
        - 0.5 * np.log(2.0 * np.pi)
        - 0.5 * np.log(sigma_sq)
        - 0.5 * ((eps_it + s * mu_it) ** 2) / sigma_sq
        + norm.logcdf(mu_star_it / sigma_star)
        - norm.logcdf(mu_it / sigma_u)
    )
    return -np.sum(ll)


# ---------------------------------------------------------------------------
# Estimator class
# ---------------------------------------------------------------------------

class EffectsPanelSFA(_BaseSFA):
    """
    Battese & Coelli (1995) Inefficiency Effects Panel SFA.

    One-step MLE: frontier parameters β and effects parameters δ are
    estimated simultaneously, eliminating two-step bias.

    Parameters
    ----------
    model_type : {'production', 'cost'}, default 'production'
    max_iter   : int, default 2000
    tol        : float, default 1e-8

    Fitted attributes
    -----------------
    coef_           : ndarray (k,)  frontier coefficients β
    delta_          : ndarray (m,)  inefficiency-effects coefficients δ
    sigma_sq_       : float         total variance σ²
    gamma_          : float         variance ratio σ_u²/σ²
    log_likelihood_ : float         maximised log-likelihood
    """

    def __init__(self, model_type: str = "production",
                 max_iter: int = 2000, tol: float = 1e-8):
        super().__init__(model_type=model_type, max_iter=max_iter, tol=tol)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y, groups, time=None, Z=None):
        """
        Estimate BC95 model via Maximum Likelihood.

        Parameters
        ----------
        X      : array-like (N, k)  frontier regressors (with intercept)
        y      : array-like (N,)    output / cost variable
        groups : array-like (N,)    entity identifiers
        time   : array-like (N,)    time indices (optional; not used in LL)
        Z      : array-like (N, m)  inefficiency-effects regressors (required)
                                    Include an intercept column in Z if desired.

        Returns
        -------
        self
        """
        if Z is None:
            raise ValueError(
                "EffectsPanelSFA requires Z (the inefficiency-effects "
                "regressors). Pass Z as a keyword argument to fit()."
            )

        X, y   = check_X_y(X, y, y_numeric=True)
        Z      = check_array(Z)
        groups = np.asarray(groups)
        self._n_obs_ = len(y)
        s = self._get_s()

        # Encode groups (not used in LL but cached for score_efficiency API)
        _, groups_mapped = np.unique(groups, return_inverse=True)

        k = X.shape[1]
        m = Z.shape[1]

        # --- OLS warm-start -------------------------------------------
        beta0, sig2_0 = self._ols_init(X, y)
        theta0 = np.concatenate([
            beta0,
            [sigma_sq_to_log(sig2_0),
             gamma_to_logit(0.5)],
            np.zeros(m),   # δ initial = 0
        ])
        self._n_params_ = len(theta0)

        # --- MLE via L-BFGS-B ----------------------------------------
        result = minimize(
            _nll_bc95,
            theta0,
            args=(X, y, Z, groups_mapped, s),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )
        if not result.success:
            import warnings
            warnings.warn(f"L-BFGS-B did not converge: {result.message}")

        # --- Store fitted attributes -----------------------------------
        params = self._unpack_theta(result.x, k, m)
        self.coef_           = params["beta"]
        self.sigma_sq_       = params["sigma_sq"]
        self.gamma_          = params["gamma"]
        self.delta_          = params["delta"]
        self.log_likelihood_ = -result.fun
        self._theta_opt_     = result.x
        self._k_             = k
        self._m_             = m
        self._groups_mapped_ = groups_mapped

        return self

    # ------------------------------------------------------------------
    # Parameter unpacking
    # ------------------------------------------------------------------

    def _unpack_theta(self, theta, k, m):
        return {
            "beta"     : theta[:k],
            "sigma_sq" : log_to_sigma_sq(theta[k]),
            "gamma"    : logit_to_gamma(theta[k + 1]),
            "delta"    : theta[k + 2: k + 2 + m],
        }

    def _log_likelihood(self, theta, X, y, Z=None, groups=None):
        return _nll_bc95(theta, X, y, Z, self._groups_mapped_, self._get_s())

    # ------------------------------------------------------------------
    # JLMS estimator (BC95)
    # ------------------------------------------------------------------

    def _jlms(self, X, y, Z=None, groups=None):
        """
        Compute E[u_{it} | ε_{it}] for every observation (N,).

        JLMS adaptation for observation-level Truncated-Normal (BC95):
            μ*_{it} = (μ_{it}·σ_v² − s·σ_u²·ε_{it}) / σ²
            σ*      = sqrt(σ_u²·σ_v²/σ²)
            E[u|ε]  = μ*_{it} + σ* · φ(μ*_{it}/σ*) / Φ(μ*_{it}/σ*)

        The `s` sign correctly routes cost vs. production frontiers.

        Parameters
        ----------
        X      : array-like (N, k)
        y      : array-like (N,)
        Z      : array-like (N, m)   inefficiency regressors (required)
        groups : ignored; accepted for API symmetry

        Returns
        -------
        e_u : ndarray (N,)  non-negative observation-level inefficiency
        """
        check_is_fitted(self)
        if Z is None:
            raise ValueError("Z is required for EffectsPanelSFA._jlms().")

        Z = check_array(Z)
        k, m = self._k_, self._m_
        params = self._unpack_theta(self._theta_opt_, k, m)

        beta     = params["beta"]
        sigma_sq = params["sigma_sq"]
        gamma    = params["gamma"]
        delta    = params["delta"]
        s        = self._get_s()

        sigma_u_sq = gamma * sigma_sq
        sigma_v_sq = (1.0 - gamma) * sigma_sq
        sigma_u    = np.sqrt(sigma_u_sq)
        sigma_star = np.sqrt(sigma_u_sq * sigma_v_sq / sigma_sq)

        eps_it    = y - X @ beta
        mu_it     = Z @ delta
        mu_star   = (mu_it * sigma_v_sq - s * sigma_u_sq * eps_it) / sigma_sq

        ratio = mu_star / sigma_star
        e_u   = mu_star + sigma_star * norm.pdf(ratio) / norm.cdf(ratio)
        return np.clip(e_u, 0.0, None)
