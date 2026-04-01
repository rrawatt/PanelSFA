import numpy as np
import pytest
from numpy.testing import assert_allclose

from panelsfa import TrueFixedEffectsSFA, TrueRandomEffectsSFA

RNG = np.random.default_rng(42)

def generate_greene_data(n_entities=100, t_periods=10, model_type="TFE"):
    """
    Complications:
    1. TFE: Firm heterogeneity (alpha_i) is correlated with X (Endogeneity).
    2. TRE: Heterogeneity is random (w_i ~ N(0, sigma_w^2)).
    """
    N = n_entities * t_periods
    groups = np.repeat(np.arange(n_entities), t_periods)
    
    # X1 is time-varying
    x1 = RNG.normal(1.0, 2.0, N)
    
    sigma_v, sigma_u = 0.3, 0.6
    v = RNG.normal(0, sigma_v, N)
    u = np.abs(RNG.normal(0, sigma_u, N))
    
    if model_type == "TFE":
        # DO NOT include a global intercept for TFE to avoid perfect collinearity with alpha_i
        X = x1.reshape(-1, 1)
        beta_true = np.array([1.5])
        
        # alpha_i is correlated with the firm's average X (classic Fixed Effects test)
        x_mean_i = np.bincount(groups, weights=x1) / t_periods
        alpha_i = 0.8 * x_mean_i + RNG.normal(0, 1.0, n_entities)
        
        y = alpha_i[groups] + X @ beta_true + v - u
        
        sigma_sq_true = sigma_v**2 + sigma_u**2
        gamma_true = sigma_u**2 / sigma_sq_true
        return X, y, groups, beta_true, sigma_sq_true, gamma_true

    elif model_type == "TRE":
        # Global intercept included
        X = np.column_stack([np.ones(N), x1])
        beta_true = np.array([2.0, 1.5])
        
        sigma_w = 0.5
        w_i = RNG.normal(0, sigma_w, n_entities)
        
        y = X @ beta_true + w_i[groups] + v - u
        
        sigma_sq_true = sigma_v**2 + sigma_u**2
        gamma_true = sigma_u**2 / sigma_sq_true
        return X, y, groups, beta_true, sigma_sq_true, gamma_true, sigma_w**2


# =====================================================================
# NEW RECOVERY TESTS
# =====================================================================

def test_tfe_recovery():
    """Test if TrueFixedEffectsSFA recovers parameters using Concentrated Likelihood."""
    from panelsfa import TrueFixedEffectsSFA
    
    X, y, groups, b_true, sig2_true, gam_true = generate_greene_data(model_type="TFE")
    
    model = TrueFixedEffectsSFA(model_type="production")
    model.fit(X, y, groups=groups)
    
    assert_allclose(model.coef_, b_true, atol=0.08)
    assert_allclose(model.sigma_sq_, sig2_true, atol=0.1)
    
    # TFE efficiency scores evaluate only transient inefficiency
    te = model.score_efficiency(X, y, groups=groups)
    assert len(te) == len(y)
    assert np.all((te > 0) & (te <= 1.0))
    # Ensure profile alphas were calculated
    assert len(model.alphas_) == len(np.unique(groups))


def test_tre_recovery():
    """Test if TrueRandomEffectsSFA recovers parameters using Maximum Simulated Likelihood."""
    from panelsfa import TrueRandomEffectsSFA
    
    X, y, groups, b_true, sig2_true, gam_true, sigw2_true = generate_greene_data(model_type="TRE")
    
    # TRE uses simulation, so tolerance is slightly looser
    model = TrueRandomEffectsSFA(model_type="production", n_simulations=200)
    model.fit(X, y, groups=groups)
    
    assert_allclose(model.coef_, b_true, atol=0.15)
    assert_allclose(model.sigma_w_sq_, sigw2_true, atol=0.2)
    
    te = model.score_efficiency(X, y, groups=groups)
    assert len(te) == len(y)
    assert np.all((te > 0) & (te <= 1.0))


# =====================================================================
# NEW STRESS TESTS
# =====================================================================

def test_tfe_incidental_parameters_stress():
    """
    STRESS TEST: Large N, Small T for True Fixed Effects.
    Tests if the profile Newton-Raphson root solver degrades gracefully 
    (falls back to OLS mean) when T is too small for reliable curvature, 
    preventing an algorithmic crash.
    """
    from panelsfa import TrueFixedEffectsSFA
    
    n_entities = 500
    t_periods = 2  # Very small T
    N = n_entities * t_periods
    groups = np.repeat(np.arange(n_entities), t_periods)
    
    X = RNG.standard_normal((N, 1))
    y = 5.0 + X[:, 0] * 1.5 + RNG.normal(0, 0.5, N) - np.abs(RNG.normal(0, 0.5, N))
    
    model = TrueFixedEffectsSFA()
    # If the root-finder throws a RuntimeError, the except block must catch it.
    model.fit(X, y, groups=groups)
    
    assert np.isfinite(model.log_likelihood_)
    assert len(model.alphas_) == n_entities


def generate_tfe_unbalanced_data(n_entities=200):
    periods = RNG.integers(1, 15, size=n_entities)
    periods[0] = 1 
    periods[1] = 2 
    
    N = periods.sum()
    groups = np.repeat(np.arange(n_entities), periods)
    
    X = RNG.standard_normal((N, 2))
    beta_true = np.array([1.4, -0.6])
    
    alpha_i = RNG.normal(2.0, 1.5, n_entities)
    v = RNG.normal(0, 0.4, N)
    u = np.abs(RNG.normal(0, 0.7, N))
    
    y = alpha_i[groups] + X @ beta_true + v - u
    
    return X, y, groups, beta_true, N

def generate_tre_variance_stress_data(n_entities=100, t_periods=6, variance_type="zero"):
    N = n_entities * t_periods
    groups = np.repeat(np.arange(n_entities), t_periods)
    
    X = np.column_stack([np.ones(N), RNG.standard_normal(N)])
    beta_true = np.array([1.5, 0.8])
    
    v = RNG.normal(0, 0.3, N)
    u = np.abs(RNG.normal(0, 0.5, N))
    
    if variance_type == "zero":
        sigma_w = 0.0
    else:
        sigma_w = 1.5 
        
    w_i = RNG.normal(0, sigma_w, n_entities)
    y = X @ beta_true + w_i[groups] + v - u
    
    return X, y, groups, beta_true, sigma_w**2

def generate_tfe_no_inefficiency_data(n_entities=300, t_periods=15):
    N = n_entities * t_periods
    groups = np.repeat(np.arange(n_entities), t_periods)
    
    X = RNG.standard_normal((N, 1))
    beta_true = np.array([2.5])
    
    alpha_i = RNG.normal(0, 1.0, n_entities)
    v = RNG.normal(0, 0.4, N) 
    
    y = alpha_i[groups] + X @ beta_true + v
    
    return X, y, groups, beta_true


def test_tfe_unbalanced_panel_and_singletons():
    """
    STRESS TEST: Highly unbalanced panel with T=1 and T=2 singletons.
    Ensures the Newton-Raphson profile solver does not crash when there is 
    insufficient temporal variation to estimate a firm's alpha.
    """
    from panelsfa import TrueFixedEffectsSFA
    
    X, y, groups, b_true, N = generate_tfe_unbalanced_data()
    
    model = TrueFixedEffectsSFA(model_type="production", max_iter=1500)
    model.fit(X, y, groups=groups)
    
    assert np.isfinite(model.log_likelihood_)
    assert_allclose(model.coef_, b_true, atol=0.1)
    
    assert len(model.alphas_) == len(np.unique(groups))
    
    te = model.score_efficiency(X, y, groups=groups)
    assert len(te) == N
    assert np.all(np.isfinite(te))


def test_tre_zero_random_effect_variance():
    """
    STRESS TEST: DGP with exactly zero unobserved heterogeneity.
    Ensures the unconstrained log(sigma_w_sq) parameter correctly bounds 
    itself near zero without throwing -inf math domain errors during Halton integration.
    """
    from panelsfa import TrueRandomEffectsSFA
    
    X, y, groups, b_true, sigw2_true = generate_tre_variance_stress_data(variance_type="zero")
    
    model = TrueRandomEffectsSFA(model_type="production", n_simulations=150)
    model.fit(X, y, groups=groups)
    
    assert model.sigma_w_sq_ < 0.05
    assert_allclose(model.coef_, b_true, atol=0.1)


def test_tre_massive_random_effect_variance():
    """
    STRESS TEST: DGP where unobserved heterogeneity dominates the noise components.
    Ensures the logsumexp MSL approximation does not underflow when the 
    Halton draws are stretched across a massive variance scale.
    """
    from panelsfa import TrueRandomEffectsSFA
    
    X, y, groups, b_true, sigw2_true = generate_tre_variance_stress_data(variance_type="massive")
    
    model = TrueRandomEffectsSFA(model_type="production", n_simulations=250)
    model.fit(X, y, groups=groups)
    
    assert np.isfinite(model.log_likelihood_)
    assert_allclose(model.sigma_w_sq_, sigw2_true, atol=0.8) 
    assert_allclose(model.coef_[1:], b_true[1:], atol=0.2)


def test_tfe_no_inefficiency_collapse():
    """
    STRESS TEST: Pure Fixed Effects DGP with no actual inefficiency (u=0).
    Ensures the model correctly identifies the absence of the half-normal 
    component, pushing gamma toward 0 and efficiency scores toward 1.
    """
    from panelsfa import TrueFixedEffectsSFA
    
    X, y, groups, b_true = generate_tfe_no_inefficiency_data()
    
    model = TrueFixedEffectsSFA(model_type="production")
    model.fit(X, y, groups=groups)
    
    assert model.gamma_ < 0.1
    
    te = model.score_efficiency(X, y, groups=groups)
    assert np.mean(te) > 0.95

