"""
panelsfa/base.py
================
Shared infrastructure for all PanelSFA estimators.

Provides:
  - Unconstrained <-> natural parameter transforms
  - _BaseSFA: sklearn BaseEstimator + RegressorMixin with shared
    predict(), score_efficiency(), AIC/BIC, and OLS warm-start.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted


# ---------------------------------------------------------------------------
# Unconstrained <-> natural parameter transforms
# ---------------------------------------------------------------------------

def log_to_sigma_sq(log_sigma_sq: float) -> float:
    """exp(ln σ²) → σ²  (guarantees positivity)."""
    return np.exp(log_sigma_sq)


def logit_to_gamma(logit_gamma: float) -> float:
    """sigmoid(logit γ) → γ ∈ (0, 1)."""
    return 1.0 / (1.0 + np.exp(-logit_gamma))


def sigma_sq_to_log(sigma_sq: float) -> float:
    """σ² → ln σ²."""
    return np.log(sigma_sq)


def gamma_to_logit(gamma: float) -> float:
    """γ → logit γ."""
    return np.log(gamma / (1.0 - gamma))


# ---------------------------------------------------------------------------
# Abstract base estimator
# ---------------------------------------------------------------------------

class _BaseSFA(BaseEstimator, RegressorMixin):
    """
    Abstract base class for all PanelSFA estimators.

    Subclasses must implement:
        _log_likelihood(theta, X, y, **kw) -> float   returns NEGATIVE LL
        _unpack_theta(theta, ...)          -> dict     natural-scale params
        _jlms(X, y, **kw)                 -> ndarray  E[u|ε] per observation
    """

    _MODEL_TYPE_MAP = {"production": 1, "cost": -1}

    def __init__(self, model_type: str = "production",
                 max_iter: int = 2000, tol: float = 1e-8):
        self.model_type = model_type
        self.max_iter = max_iter
        self.tol = tol

    # ------------------------------------------------------------------
    # Public sklearn API
    # ------------------------------------------------------------------

    def predict(self, X):
        """
        Return deterministic frontier prediction X @ coef_.

        Parameters
        ----------
        X : array-like (n, k)

        Returns
        -------
        y_pred : ndarray (n,)
        """
        check_is_fitted(self)
        X = check_array(X)
        return X @ self.coef_

    def score_efficiency(self, X, y, **kw):
        """
        Compute Technical Efficiency via the JLMS estimator.

        Returns TE = exp(−E[u|ε]) ∈ (0, 1] for every observation.
        TE = 1 denotes a fully efficient unit.

        The sign of the inefficiency term (`s`) is handled internally
        inside each subclass's `_jlms` method, so this method is
        identical for production and cost frontiers.

        Parameters
        ----------
        X    : array-like (N, k)
        y    : array-like (N,)
        **kw : forwarded to subclass _jlms (groups, time, Z as needed)

        Returns
        -------
        te : ndarray (N,)
        """
        check_is_fitted(self)
        u_hat = self._jlms(X, y, **kw)
        return np.exp(-u_hat)

    # ------------------------------------------------------------------
    # Information criteria (populated after fit())
    # ------------------------------------------------------------------

    @property
    def aic_(self):
        check_is_fitted(self)
        return -2.0 * self.log_likelihood_ + 2.0 * self._n_params_

    @property
    def bic_(self):
        check_is_fitted(self)
        return -2.0 * self.log_likelihood_ + self._n_params_ * np.log(self._n_obs_)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_s(self) -> int:
        """Return s = +1 (production) or −1 (cost)."""
        if self.model_type not in self._MODEL_TYPE_MAP:
            raise ValueError(
                f"model_type must be 'production' or 'cost', "
                f"got '{self.model_type}'"
            )
        return self._MODEL_TYPE_MAP[self.model_type]

    def _ols_init(self, X, y):
        """
        OLS warm-start for the MLE optimizer.
        """
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid    = y - X @ beta
        sigma_sq = float(np.var(resid))
        
        # Shift the intercept to account for E[u] to prevent local minima.
        # Assuming starting gamma ≈ 0.5, E[u] ≈ sqrt(sigma_sq / pi)
        s = self._get_s()
        beta += s * np.sqrt(sigma_sq / np.pi)

        return beta, max(sigma_sq, 1e-6)

    # ------------------------------------------------------------------
    # Stubs – must be overridden by subclasses
    # ------------------------------------------------------------------

    def _log_likelihood(self, theta, X, y, **kw):  # pragma: no cover
        raise NotImplementedError

    def _unpack_theta(self, theta, *args):  # pragma: no cover
        raise NotImplementedError

    def _jlms(self, X, y, **kw):  # pragma: no cover
        raise NotImplementedError
