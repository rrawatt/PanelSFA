"""
panelsfa/four_component.py
==========================
Kumbhakar, Lien, and Hardaker (2014) – Four-Component Model

Model
-----
    y_{it} = β_0 + x_{it} β + μ_i + v_{it} − s·η_i − s·u_{it}
    
Error Decomposition:
    μ_i    ~ N(0, σ_μ²)        (Firm Heterogeneity / Unobserved Time-Invariant)
    v_{it} ~ N(0, σ_v²)        (Transient Noise / Idiosyncratic Shock)
    η_i    ~ |N(0, σ_η²)|      (Persistent / Structural Inefficiency)
    u_{it} ~ |N(0, σ_u²)|      (Transient / Time-Varying Inefficiency)

Estimation Strategy
-------------------
Three-Step Pseudo-Likelihood:
    1. Panel Within-Transformation (Fixed Effects) to consistently estimate β.
    2. Cross-Sectional SFA on the time-varying residuals to isolate σ_v² and σ_u².
    3. Cross-Sectional SFA on the time-invariant intercepts to isolate β_0, σ_μ², and σ_η².
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted

from .base import _BaseSFA
from .cross_sectional import CrossSectionalSFA

# ===========================================================================
# Estimator class
# ===========================================================================

class FourComponentSFA(BaseEstimator, RegressorMixin):
    """
    Kumbhakar, Lien, and Hardaker (2014) Four-Component Panel SFA.

    Parameters
    ----------
    model_type : {'production', 'cost'}, default 'production'
    max_iter   : int, default 2000
    tol        : float, default 1e-8

    Fitted attributes
    -----------------
    coef_           : ndarray (k,)  frontier coefficients β (excluding intercept)
    intercept_      : float         overall frontier intercept β_0
    sigma_v_sq_     : float         variance of transient noise (v_{it})
    sigma_u_sq_     : float         variance of transient inefficiency (u_{it})
    sigma_mu_sq_    : float         variance of firm heterogeneity (μ_i)
    sigma_eta_sq_   : float         variance of persistent inefficiency (η_i)
    """

    def __init__(self, model_type: str = "production",
                 max_iter: int = 2000, tol: float = 1e-8):
        self.model_type = model_type
        self.max_iter = max_iter
        self.tol = tol

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y, groups):
        """
        Estimate the Four-Component Model via the Three-Step procedure.

        Parameters
        ----------
        X      : array-like (N, k)  frontier regressors (DO NOT include an intercept column)
        y      : array-like (N,)    output / cost variable
        groups : array-like (N,)    entity identifiers
        """
        X, y = check_X_y(X, y, y_numeric=True)
        groups = np.asarray(groups)
        self._n_obs_ = len(y)

        unique_ids, groups_mapped = np.unique(groups, return_inverse=True)
        n_groups = len(unique_ids)

        # --- STEP 1: The Within-Transformation (Fixed Effects) --------
        T_counts = np.bincount(groups_mapped).astype(float)
        
        y_mean_i = np.bincount(groups_mapped, weights=y) / T_counts
        y_demeaned = y - y_mean_i[groups_mapped]
        
        X_demeaned = np.zeros_like(X)
        for j in range(X.shape[1]):
            X_mean_ij = np.bincount(groups_mapped, weights=X[:, j]) / T_counts
            X_demeaned[:, j] = X[:, j] - X_mean_ij[groups_mapped]

        # OLS without intercept on demeaned data
        beta, _, _, _ = np.linalg.lstsq(X_demeaned, y_demeaned, rcond=None)
        
        eps_total = y - X @ beta
        alpha_i = np.bincount(groups_mapped, weights=eps_total) / T_counts
        eps_it = eps_total - alpha_i[groups_mapped]

        # --- STEP 2: Transient SFA ------------------------------------
        self._step2_model = CrossSectionalSFA(
            model_type=self.model_type, 
            max_iter=self.max_iter, 
            tol=self.tol
        )
        X_step2 = np.ones((self._n_obs_, 1))
        self._step2_model.fit(X_step2, eps_it)
        
        sigma_sq_trans = self._step2_model.sigma_sq_
        gamma_trans = self._step2_model.gamma_
        sigma_v_sq = (1.0 - gamma_trans) * sigma_sq_trans
        sigma_u_sq = gamma_trans * sigma_sq_trans

        # --- STEP 3: Persistent SFA -----------------------------------
        self._step3_model = CrossSectionalSFA(
            model_type=self.model_type, 
            max_iter=self.max_iter, 
            tol=self.tol
        )
        X_step3 = np.ones((n_groups, 1))
        self._step3_model.fit(X_step3, alpha_i)
        
        beta_0_step3 = self._step3_model.coef_[0]
        sigma_sq_pers = self._step3_model.sigma_sq_
        gamma_pers = self._step3_model.gamma_
        sigma_mu_sq = (1.0 - gamma_pers) * sigma_sq_pers
        sigma_eta_sq = gamma_pers * sigma_sq_pers

        s = 1 if self.model_type == 'production' else -1
        E_u = np.sqrt(sigma_u_sq * 2.0 / np.pi)
        beta_0 = beta_0_step3 + s * E_u

        # --- Store fitted attributes ----------------------------------
        self.coef_           = beta
        self.intercept_      = beta_0
        self.sigma_v_sq_     = sigma_v_sq
        self.sigma_u_sq_     = sigma_u_sq
        self.sigma_mu_sq_    = sigma_mu_sq
        self.sigma_eta_sq_   = sigma_eta_sq
        self._unique_ids_    = unique_ids
        self._groups_mapped_ = groups_mapped

        return self

    # ------------------------------------------------------------------
    # Technical Efficiency (Multi-Dimensional Scoring)
    # ------------------------------------------------------------------

    def predict(self, X):
        """
        Return deterministic frontier prediction: β_0 + Xβ.
        """
        check_is_fitted(self)
        return self.intercept_ + X @ self.coef_

    def score_efficiency(self, X, y, groups):
        """
        Compute the Transient, Persistent, and Overall Technical Efficiency.
        
        Returns
        -------
        eff_dict : dict
            Contains three ndarrays of shape (N,):
            - 'transient'  : Transient Technical Efficiency (TTE_it)
            - 'persistent' : Persistent Technical Efficiency (PTE_i)
            - 'overall'    : Overall Technical Efficiency (OTE_it = PTE_i * TTE_it)
        """
        check_is_fitted(self)
        X, y = check_X_y(X, y, y_numeric=True)
        groups = np.asarray(groups)
        
        # 1. Re-evaluate Step 1 to isolate residuals for scoring
        _, groups_mapped = np.unique(groups, return_inverse=True)
        T_counts = np.bincount(groups_mapped).astype(float)
        
        eps_total = y - X @ self.coef_
        alpha_i = np.bincount(groups_mapped, weights=eps_total) / T_counts
        eps_it = eps_total - alpha_i[groups_mapped]
        
        # 2. Score Transient TE (TTE) using Step 2 Model
        X_step2 = np.ones((len(y), 1))
        tte = self._step2_model.score_efficiency(X_step2, eps_it)
        
        # 3. Score Persistent TE (PTE) using Step 3 Model
        X_step3 = np.ones((len(T_counts), 1))
        pte_group = self._step3_model.score_efficiency(X_step3, alpha_i)
        
        # Broadcast PTE back to observation level
        pte = pte_group[groups_mapped]
        
        # 4. Overall Technical Efficiency
        ote = pte * tte
        
        return {
            "transient": tte,
            "persistent": pte,
            "overall": ote
        }