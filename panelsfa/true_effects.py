"""
panelsfa/true_effects.py
===================
Greene (2005) – True Fixed Effects (TFE) & True Random Effects (TRE)

Models
------
1. True Fixed Effects (TFE):
    y_{it} = α_i + x_{it} β + v_{it} − s·u_{it}
    (Solves the Incidental Parameters Problem via Concentrated Likelihood)

2. True Random Effects (TRE):
    y_{it} = (β_0 + w_i) + x_{it} β + v_{it} − s·u_{it}
    w_i ~ N(0, σ_w²)
    (Integrates out unobserved heterogeneity via Maximum Simulated Likelihood)

where s = +1 (production) or s = −1 (cost).

Parameterisation
----------------
    σ² = σ_v² + σ_u²,   γ = σ_u² / σ²
    σ_w² = Variance of the random effect (TRE only)

Optimiser vectors:
    TFE: θ = [β₁…βₖ, ln(σ²), logit(γ)]  (α_i are profiled out internally)
    TRE: θ = [β₁…βₖ, ln(σ²), logit(γ), ln(σ_w²)]
"""

import numpy as np
from scipy.optimize import minimize, newton
from scipy.stats import norm, qmc
from scipy.special import logsumexp

from sklearn.utils.validation import check_X_y, check_is_fitted

from .base import (
    _BaseSFA,
    log_to_sigma_sq, logit_to_gamma,
    sigma_sq_to_log, gamma_to_logit,
)

# ===========================================================================
# 1. True Fixed Effects (TFE) - Concentrated Likelihood
# ===========================================================================

def _foc_alpha(alpha, y_i, X_i_beta, sigma_sq, lam, s):
    """
    First-Order Condition for the profiled entity-specific intercept α_i.
    Uses asymptotic expansion for the Inverse Mills Ratio to prevent NaN.
    """
    eps = y_i - X_i_beta - alpha
    sigma = np.sqrt(sigma_sq)
    z = -s * lam * eps / sigma
    
    # Numerically Stable Inverse Mills Ratio
    imr = np.zeros_like(z)
    mask = z > -30  # Threshold where logcdf begins failing
    
    # Standard evaluation for normal domain
    imr[mask] = np.exp(norm.logpdf(z[mask]) - norm.logcdf(z[mask]))
    
    # Asymptotic approximation (IMR ≈ -z) for extreme negative domain
    imr[~mask] = -z[~mask]
    
    return np.sum(eps / sigma_sq + s * (lam / sigma) * imr)


def _profile_alphas(X_beta, y, groups_mapped, n_groups, sigma_sq, lam, s):
    """
    Solves for the optimal α_i vector conditional on the current parameters.
    Returns array of shape (n_groups,).
    """
    alphas = np.zeros(n_groups)
    
    for g in range(n_groups):
        mask = (groups_mapped == g)
        y_i = y[mask]
        X_i_beta = X_beta[mask]
        
        # OLS residual mean as starting guess for the root solver
        x0_init = np.mean(y_i - X_i_beta)
        
        try:
            # Newton-Raphson to find the root of the FOC
            alpha_opt = newton(
                func=_foc_alpha, 
                x0=x0_init,
                args=(y_i, X_i_beta, sigma_sq, lam, s), 
                maxiter=50,
                tol=1e-5
            )
        except (RuntimeError, ValueError):
            # Fallback to the mean if the local curvature stalls the solver
            alpha_opt = x0_init
            
        alphas[g] = alpha_opt
        
    return alphas


def _nll_tfe(theta, X, y, groups_mapped, n_groups, s):
    """
    Negative Concentrated Log-Likelihood for TFE.
    """
    k        = X.shape[1]
    beta     = theta[:k]
    sigma_sq = log_to_sigma_sq(theta[k])
    gamma    = logit_to_gamma(theta[k + 1])

    sigma_u_sq = gamma * sigma_sq
    sigma_v_sq = (1.0 - gamma) * sigma_sq
    sigma      = np.sqrt(sigma_sq)
    lam = np.sqrt(sigma_u_sq / max(sigma_v_sq, 1e-12))

    X_beta = X @ beta
    
    # Dynamically profile out the incidental parameters
    alphas = _profile_alphas(X_beta, y, groups_mapped, n_groups, sigma_sq, lam, s)
    
    # Broadcast optimized alphas back to the observation level
    alpha_full = alphas[groups_mapped]
    eps = y - alpha_full - X_beta
    
    # Evaluate cross-sectional likelihood using the profiled residuals
    ll = (
        np.log(2.0) - np.log(sigma) 
        + norm.logpdf(eps / sigma) 
        + norm.logcdf(-s * eps * lam / sigma)
    )
    
    # Objective value
    nll = -np.sum(ll)
    
    # Store the profiled alphas in a transient dictionary to extract post-convergence
    _nll_tfe.last_alphas = alphas  
    return nll


class TrueFixedEffectsSFA(_BaseSFA):
    """
    Greene (2005) True Fixed Effects SFA.
    """
    def __init__(self, model_type: str = "production",
                 max_iter: int = 1000, tol: float = 1e-6):
        super().__init__(model_type=model_type, max_iter=max_iter, tol=tol)

    def fit(self, X, y, groups):
        X, y   = check_X_y(X, y, y_numeric=True)
        groups = np.asarray(groups)
        self._n_obs_ = len(y)
        s = self._get_s()

        unique_ids, groups_mapped = np.unique(groups, return_inverse=True)
        n_groups = len(unique_ids)

        # OLS warm-start
        beta0, sig2_0 = self._ols_init(X, y)
        theta0 = np.concatenate([
            beta0,
            [sigma_sq_to_log(sig2_0), gamma_to_logit(0.5)],
        ])
        self._n_params_ = len(theta0)

        # Clear static cache for safety
        _nll_tfe.last_alphas = None 

        result = minimize(
            _nll_tfe,
            theta0,
            args=(X, y, groups_mapped, n_groups, s),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )
        if not result.success:
            import warnings
            warnings.warn(f"L-BFGS-B did not converge: {result.message}")

        k = X.shape[1]
        params = self._unpack_theta(result.x, k)
        
        self.coef_           = params["beta"]
        self.sigma_sq_       = params["sigma_sq"]
        self.gamma_          = params["gamma"]
        self.log_likelihood_ = -result.fun
        self.alphas_         = _nll_tfe.last_alphas
        self._theta_opt_     = result.x
        self._groups_mapped_ = groups_mapped
        self._unique_ids_    = unique_ids

        return self

    def _unpack_theta(self, theta, k):
        return {
            "beta"     : theta[:k],
            "sigma_sq" : log_to_sigma_sq(theta[k]),
            "gamma"    : logit_to_gamma(theta[k + 1]),
        }

    def _log_likelihood(self, theta, X, y, groups=None):
        return _nll_tfe(theta, X, y, self._groups_mapped_, len(self._unique_ids_), self._get_s())

    def _jlms(self, X, y, groups=None):
        """
        JLMS estimator applied strictly to the transient inefficiency.
        """
        check_is_fitted(self)
        k = X.shape[1]
        params = self._unpack_theta(self._theta_opt_, k)

        beta       = params["beta"]
        sigma_sq   = params["sigma_sq"]
        gamma      = params["gamma"]
        s          = self._get_s()

        sigma_u_sq = gamma * sigma_sq
        sigma_v_sq = (1.0 - gamma) * sigma_sq
        sigma_star = np.sqrt(sigma_v_sq * sigma_u_sq / sigma_sq)

        alpha_full = self.alphas_[self._groups_mapped_]
        
        # Residual isolated from firm heterogeneity
        eps = y - alpha_full - X @ beta
        mu_star = -s * eps * sigma_u_sq / sigma_sq

        ratio = mu_star / sigma_star
        e_u   = sigma_star * (norm.pdf(ratio) / norm.cdf(ratio) + ratio)
        return np.clip(e_u, 0.0, None)


# ===========================================================================
# 2. True Random Effects (TRE) - Maximum Simulated Likelihood
# ===========================================================================

def _nll_tre(theta, X, y, groups_mapped, n_groups, halton_matrix, s):
    """
    Negative Maximum Simulated Log-Likelihood for TRE.
    
    halton_matrix: shape (n_groups, R), standard normal Halton draws
    """
    k          = X.shape[1]
    beta       = theta[:k]
    sigma_sq   = log_to_sigma_sq(theta[k])
    gamma      = logit_to_gamma(theta[k + 1])
    sigma_w_sq = log_to_sigma_sq(theta[k + 2])
    
    sigma   = np.sqrt(sigma_sq)
    sigma_w = np.sqrt(sigma_w_sq)
    lam     = np.sqrt((gamma * sigma_sq) / ((1.0 - gamma) * sigma_sq))
    
    R = halton_matrix.shape[1]

    # Map the simulated random effects to the observation level
    # W_ir shape: (N, R)
    W_ir = sigma_w * halton_matrix[groups_mapped, :]
    
    # eps_ir shape: (N, R)
    X_beta = (X @ beta)[:, None]
    eps_ir = y[:, None] - X_beta - W_ir
    
    # Observation-level log-likelihoods for each simulation (N, R)
    ll_itr = (
        np.log(2.0) - np.log(sigma) 
        + norm.logpdf(eps_ir / sigma) 
        + norm.logcdf(-s * eps_ir * lam / sigma)
    )
    
    # Group by entity: Sum over time periods t, keeping R simulations
    # ll_gr shape: (n_groups, R)
    ll_gr = np.zeros((n_groups, R))
    np.add.at(ll_gr, groups_mapped, ll_itr)
    
    # Integrate out the random effect: average across simulations
    # ln( 1/R * sum(exp(ll_gr)) ) = logsumexp(ll_gr) - ln(R)
    ll_i = logsumexp(ll_gr, axis=1) - np.log(R)
    
    return -np.sum(ll_i)


class TrueRandomEffectsSFA(_BaseSFA):
    """
    Greene (2005) True Random Effects SFA via MSL.
    """
    def __init__(self, model_type: str = "production", n_simulations: int = 200,
                 max_iter: int = 1000, tol: float = 1e-6):
        super().__init__(model_type=model_type, max_iter=max_iter, tol=tol)
        self.n_simulations = n_simulations

    def fit(self, X, y, groups):
        X, y   = check_X_y(X, y, y_numeric=True)
        groups = np.asarray(groups)
        self._n_obs_ = len(y)
        s = self._get_s()

        unique_ids, groups_mapped = np.unique(groups, return_inverse=True)
        n_groups = len(unique_ids)

        # Generate Halton draws once
        sampler = qmc.Halton(d=1, scramble=True)
        uniform_draws = sampler.random(n=n_groups * self.n_simulations)
        uniform_draws = np.clip(uniform_draws, 1e-10, 1.0 - 1e-10) # Prevent inf
        self._halton_matrix_ = norm.ppf(uniform_draws).reshape((n_groups, self.n_simulations))

        # OLS warm-start
        beta0, sig2_0 = self._ols_init(X, y)
        theta0 = np.concatenate([
            beta0,
            [sigma_sq_to_log(sig2_0), 
             gamma_to_logit(0.5), 
             sigma_sq_to_log(sig2_0 * 0.1)] # Init σ_w² to 10% of total variance
        ])
        self._n_params_ = len(theta0)

        result = minimize(
            _nll_tre,
            theta0,
            args=(X, y, groups_mapped, n_groups, self._halton_matrix_, s),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )
        if not result.success:
            import warnings
            warnings.warn(f"L-BFGS-B did not converge: {result.message}")

        k = X.shape[1]
        params = self._unpack_theta(result.x, k)
        
        self.coef_           = params["beta"]
        self.sigma_sq_       = params["sigma_sq"]
        self.gamma_          = params["gamma"]
        self.sigma_w_sq_     = params["sigma_w_sq"]
        self.log_likelihood_ = -result.fun
        self._theta_opt_     = result.x
        self._groups_mapped_ = groups_mapped
        self._n_groups_      = n_groups

        return self

    def _unpack_theta(self, theta, k):
        return {
            "beta"       : theta[:k],
            "sigma_sq"   : log_to_sigma_sq(theta[k]),
            "gamma"      : logit_to_gamma(theta[k + 1]),
            "sigma_w_sq" : log_to_sigma_sq(theta[k + 2]),
        }

    def _log_likelihood(self, theta, X, y, groups=None):
        return _nll_tre(theta, X, y, self._groups_mapped_, self._n_groups_, self._halton_matrix_, self._get_s())

    def _jlms(self, X, y, groups=None):
        """
        Simulated JLMS estimator. 
        Computes E[exp(-u)] conditional on the posterior of the random effect.
        """
        check_is_fitted(self)
        k = X.shape[1]
        params = self._unpack_theta(self._theta_opt_, k)

        beta       = params["beta"]
        sigma_sq   = params["sigma_sq"]
        gamma      = params["gamma"]
        sigma_w_sq = params["sigma_w_sq"]
        s          = self._get_s()

        sigma_u_sq = gamma * sigma_sq
        sigma_v_sq = (1.0 - gamma) * sigma_sq
        sigma      = np.sqrt(sigma_sq)
        sigma_star = np.sqrt(sigma_v_sq * sigma_u_sq / sigma_sq)
        lam        = np.sqrt(sigma_u_sq / sigma_v_sq)
        
        R = self.n_simulations
        W_ir = np.sqrt(sigma_w_sq) * self._halton_matrix_[self._groups_mapped_, :]
        
        X_beta = (X @ beta)[:, None]
        eps_ir = y[:, None] - X_beta - W_ir
        
        # 1. Compute observation likelihoods
        ll_itr = (
            np.log(2.0) - np.log(sigma) 
            + norm.logpdf(eps_ir / sigma) 
            + norm.logcdf(-s * eps_ir * lam / sigma)
        )
        
        # 2. Compute posterior simulation weights
        ll_gr = np.zeros((self._n_groups_, R))
        np.add.at(ll_gr, self._groups_mapped_, ll_itr)
        
        # Softmax over the simulations axis to get probability weights summing to 1
        ll_gr_max = np.max(ll_gr, axis=1, keepdims=True)
        weights_gr = np.exp(ll_gr - ll_gr_max)
        weights_gr /= np.sum(weights_gr, axis=1, keepdims=True)
        
        # Map weights back to observation level
        weights_ir = weights_gr[self._groups_mapped_, :]
        
        # 3. Compute conditional efficiency E[u_itr | eps_itr] for each simulation
        mu_star_ir = -s * eps_ir * sigma_u_sq / sigma_sq
        ratio = mu_star_ir / sigma_star
        u_hat_ir = sigma_star * (norm.pdf(ratio) / norm.cdf(ratio) + ratio)
        te_ir = np.exp(-np.clip(u_hat_ir, 0.0, None))
        
        # 4. Integrate (weighted average)
        te_i = np.sum(weights_ir * te_ir, axis=1)
        
        # Base class `score_efficiency` expects u_hat and applies exp(-u),
        # so we revert TE to the u scale.
        return -np.log(np.clip(te_i, 1e-10, 1.0))