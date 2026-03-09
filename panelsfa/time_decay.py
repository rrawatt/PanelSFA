"""
panelsfa/time_decay.py
=======================
Battese & Coelli (1992) – Time-Varying Decay Panel SFA

Model
-----
    y_{it} = x_{it} β + v_{it} − s·u_{it}
    u_i    ~ N(μ, σ_u²)⁺                       (Truncated-Normal)
    u_{it} = u_i · exp(−η (t − T_i))           (time-decay structure)

where T_i is the last observed period for entity i.
η > 0 → inefficiency decreases over time; η < 0 → increases.

Parameterisation
----------------
    σ² = σ_v² + σ_u²,   γ = σ_u² / σ²

Optimiser vector θ = [β₁…βₖ,  ln(σ²),  logit(γ),  μ,  η]

Derivation of A_i (CORRECTION 2)
----------------------------------
The entity-level marginal likelihood integrates out u_i.  Because u_i is a
*single* draw shared across all T_i periods, the compound residual integrates
to a scalar Gaussian whose precision is:

    1/σ_v² · Σ_t h_{it}²  +  1/σ_u²
    ⟹  A_i = σ_u² · H_sq + σ_v²               (NOT + σ_v² · T_counts)

where H_sq = Σ_t h_{it}².  The extra T_counts factor would only arise if
the v_{it} residuals were pooled without weighting, which misaligns with
the BC92 integration path.

BC92 Log-Likelihood Kernel (CORRECTION 3)
------------------------------------------
The correct completion-of-the-square for the marginal likelihood is:

    ℓ_i = −T_i/2 · ln(2π)
          −T_i/2 · ln(σ_v²)
          −1/(2σ_v²) · Σ_t ε_{it}²
          +1/2 · ln(σ*_i²)
          −1/2 · ln(σ_u²)
          −1/2 · μ²/σ_u²
          +1/2 · (μ*_i/σ*_i)²
          + ln Φ(μ*_i / σ*_i)
          − ln Φ(μ / σ_u)

where:
    μ*_i   = (−s · σ_u² · ε*_i  +  μ · σ_v²) / A_i
    σ*_i²  = σ_u² · σ_v² / A_i
    ε*_i   = Σ_t h_{it} · ε_{it}

All panel aggregations use np.bincount — no Python for-loops.
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
# Vectorized log-likelihood kernel
# ---------------------------------------------------------------------------

def _nll_bc92(theta, X, y, groups_mapped, time, T_i_mapped, s):
    """
    Negative log-likelihood for BC92 (Truncated-Normal, time-decay).

    All entity-level aggregations use np.bincount — zero Python for-loops.

    Parameters
    ----------
    theta         : [β (k), ln σ², logit γ, μ, η]
    X             : (N, k)
    y             : (N,)
    groups_mapped : (N,) int   contiguous 0-based entity codes
    time          : (N,)       time indices (numeric)
    T_i_mapped    : (N,)       max-time of each obs's entity, broadcast back
    s             : +1 / −1

    Returns
    -------
    nll : float
    """
    k        = X.shape[1]
    beta     = theta[:k]
    sigma_sq = log_to_sigma_sq(theta[k])
    gamma    = logit_to_gamma(theta[k + 1])
    mu       = theta[k + 2]
    eta      = theta[k + 3]

    sigma_u_sq = gamma * sigma_sq
    sigma_v_sq = (1.0 - gamma) * sigma_sq
    sigma_u    = np.sqrt(sigma_u_sq)

    eps  = y - X @ beta                               # (N,)
    h_it = np.exp(-eta * (time - T_i_mapped))         # (N,)  decay weights

    # --- Vectorized entity-level aggregations via bincount --------------
    H_sq       = np.bincount(groups_mapped, weights=h_it ** 2)       # (G,)
    eps_star   = np.bincount(groups_mapped, weights=h_it * eps)       # (G,)  ε*_i
    sum_eps_sq = np.bincount(groups_mapped, weights=eps ** 2)         # (G,)
    T_counts   = np.bincount(groups_mapped).astype(float)             # (G,)

    # --- Entity-level posterior moments (CORRECTION 2: A_i has no T_counts) --
    A_i          = sigma_u_sq * H_sq + sigma_v_sq                     # (G,)
    mu_star_i    = (-s * sigma_u_sq * eps_star + mu * sigma_v_sq) / A_i  # (G,)
    sigma_star_i = np.sqrt(sigma_u_sq * sigma_v_sq / A_i)             # (G,)

    # --- Entity-level log-likelihood contributions (CORRECTION 3) ------
    ll_i = (
        - 0.5 * T_counts * np.log(2.0 * np.pi)
        - 0.5 * T_counts * np.log(sigma_v_sq)
        - 0.5 * sum_eps_sq / sigma_v_sq
        + 0.5 * np.log(sigma_star_i ** 2)
        - 0.5 * np.log(sigma_u_sq)
        - 0.5 * (mu ** 2) / sigma_u_sq
        + 0.5 * (mu_star_i / sigma_star_i) ** 2
        + norm.logcdf(mu_star_i / sigma_star_i)
        - norm.logcdf(mu / sigma_u)
    )
    return -np.sum(ll_i)


# ---------------------------------------------------------------------------
# Estimator class
# ---------------------------------------------------------------------------

class TimeDecayPanelSFA(_BaseSFA):
    """
    Battese & Coelli (1992) Time-Varying Decay Panel SFA.

    Parameters
    ----------
    model_type : {'production', 'cost'}, default 'production'
    max_iter   : int, default 2000
    tol        : float, default 1e-8

    Fitted attributes
    -----------------
    coef_           : ndarray (k,)  frontier coefficients β
    sigma_sq_       : float         total variance σ²
    gamma_          : float         variance ratio σ_u²/σ²
    mu_             : float         mean of truncated-normal u_i
    eta_            : float         time-decay parameter η
    log_likelihood_ : float         maximised log-likelihood
    """

    def __init__(self, model_type: str = "production",
                 max_iter: int = 2000, tol: float = 1e-8):
        super().__init__(model_type=model_type, max_iter=max_iter, tol=tol)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y, groups, time):
        """
        Estimate BC92 model via Maximum Likelihood.

        Parameters
        ----------
        X      : array-like (N, k)  frontier regressors (with intercept)
        y      : array-like (N,)    output / cost variable
        groups : array-like (N,)    entity identifiers (any comparable type)
        time   : array-like (N,)    time indices (numeric, 1-based within entity)

        Returns
        -------
        self
        """
        X, y   = check_X_y(X, y, y_numeric=True)
        groups = np.asarray(groups)
        time   = np.asarray(time, dtype=float)
        self._n_obs_ = len(y)
        s = self._get_s()

        # --- Encode groups to contiguous 0-based integers --------------
        unique_ids, groups_mapped = np.unique(groups, return_inverse=True)

        # --- Broadcast T_i (entity max-time) back to observation level -
        # A list comprehension here is a one-time O(N) pre-computation
        # in fit(), not a repeated call inside the LL hot path.
        T_max_per_group = np.array([
            time[groups_mapped == i].max()
            for i in range(len(unique_ids))
        ])
        T_i_mapped = T_max_per_group[groups_mapped]   # (N,)

        # --- OLS warm-start -------------------------------------------
        beta0, sig2_0 = self._ols_init(X, y)
        theta0 = np.concatenate([
            beta0,
            [sigma_sq_to_log(sig2_0),
             gamma_to_logit(0.5),
             0.0,    # μ initial
             0.0],   # η initial
        ])
        self._n_params_ = len(theta0)

        # --- MLE via L-BFGS-B ----------------------------------------
        result = minimize(
            _nll_bc92,
            theta0,
            args=(X, y, groups_mapped, time, T_i_mapped, s),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )
        if not result.success:
            import warnings
            warnings.warn(f"L-BFGS-B did not converge: {result.message}")

        # --- Store fitted attributes -----------------------------------
        k = X.shape[1]
        params = self._unpack_theta(result.x, k)
        self.coef_           = params["beta"]
        self.sigma_sq_       = params["sigma_sq"]
        self.gamma_          = params["gamma"]
        self.mu_             = params["mu"]
        self.eta_            = params["eta"]
        self.log_likelihood_ = -result.fun
        self._theta_opt_     = result.x
        # Cache pre-computed panel structure for JLMS reuse
        self._groups_mapped_ = groups_mapped
        self._T_i_mapped_    = T_i_mapped
        self._unique_ids_    = unique_ids

        return self

    # ------------------------------------------------------------------
    # Parameter unpacking
    # ------------------------------------------------------------------

    def _unpack_theta(self, theta, k):
        return {
            "beta"     : theta[:k],
            "sigma_sq" : log_to_sigma_sq(theta[k]),
            "gamma"    : logit_to_gamma(theta[k + 1]),
            "mu"       : theta[k + 2],
            "eta"      : theta[k + 3],
        }

    def _log_likelihood(self, theta, X, y, groups=None, time=None):
        return _nll_bc92(
            theta, X, y,
            self._groups_mapped_, time, self._T_i_mapped_,
            self._get_s(),
        )

    # ------------------------------------------------------------------
    # JLMS estimator (BC92) – fully vectorized
    # ------------------------------------------------------------------

    def _jlms(self, X, y, groups=None, time=None):
        """
        Compute E[u_{it} | ε_i] for every observation (N,).

        JLMS adaptation for BC92:
            E[u_i | ε_i]    = μ*_i + σ*_i · φ(μ*_i/σ*_i) / Φ(μ*_i/σ*_i)
            E[u_{it} | ε_i] = h_{it} · E[u_i | ε_i]

        Uses cached `_groups_mapped_` and `_T_i_mapped_` from fit().
        The `time` argument is accepted for API consistency; if provided
        it is used, otherwise the cached time from fit() is applied.

        The `s` parameter routes cost vs. production sign in mu_star_i.

        Returns
        -------
        e_u : ndarray (N,)  non-negative observation-level inefficiency
        """
        check_is_fitted(self)
        k      = X.shape[1]
        params = self._unpack_theta(self._theta_opt_, k)

        beta       = params["beta"]
        sigma_sq   = params["sigma_sq"]
        gamma      = params["gamma"]
        mu         = params["mu"]
        eta        = params["eta"]
        s          = self._get_s()

        sigma_u_sq = gamma * sigma_sq
        sigma_v_sq = (1.0 - gamma) * sigma_sq

        groups_mapped = self._groups_mapped_
        T_i_mapped    = self._T_i_mapped_
        time_arr      = (np.asarray(time, dtype=float)
                         if time is not None else T_i_mapped)

        eps  = y - X @ beta
        h_it = np.exp(-eta * (time_arr - T_i_mapped))   # (N,)

        # --- Vectorized entity-level aggregation (no for-loop) ---------
        H_sq     = np.bincount(groups_mapped, weights=h_it ** 2)
        eps_star = np.bincount(groups_mapped, weights=h_it * eps)
        T_counts = np.bincount(groups_mapped).astype(float)

        # CORRECTION 2: A_i = σ_u² · H_sq + σ_v²  (no T_counts multiplier)
        A_i          = sigma_u_sq * H_sq + sigma_v_sq
        mu_star_i    = (-s * sigma_u_sq * eps_star + mu * sigma_v_sq) / A_i
        sigma_star_i = np.sqrt(sigma_u_sq * sigma_v_sq / A_i)

        ratio  = mu_star_i / sigma_star_i
        E_u_i  = mu_star_i + sigma_star_i * norm.pdf(ratio) / norm.cdf(ratio)
        E_u_i  = np.clip(E_u_i, 0.0, None)              # (G,) entity-level

        # --- Broadcast back to obs level and scale by h_{it} -----------
        e_u = h_it * E_u_i[groups_mapped]               # (N,)
        return e_u
