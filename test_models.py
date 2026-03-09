"""
test_models.py
Validation suite for PanelSFA.
Run with: pytest test_models.py -v
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from panelsfa import CrossSectionalSFA, TimeDecayPanelSFA, EffectsPanelSFA

# Set seed for reproducible synthetic data
RNG = np.random.default_rng(42)

# =====================================================================
# DGP 1: Cross-Sectional (ALS 1977)
# =====================================================================
from scipy.stats import truncnorm

# =====================================================================
# DGP 1: Cross-Sectional (ALS 1977) - The Multicollinearity Stress
# =====================================================================
def generate_als_data(n=500, model_type="production"):
    """
    Complications:
    1. X1 is Log-Normal (skewed, strictly positive like Capital/Labor).
    2. X2 is highly collinear with X1 (correlation ~ 0.85).
    3. Variances are scaled up to lower the Signal-to-Noise ratio.
    """
    # Skewed, realistically scaled variable
    x1 = RNG.lognormal(mean=1.0, sigma=0.6, size=n)
    # X2 is heavily correlated with X1
    x2 = 0.8 * x1 + RNG.normal(loc=0, scale=1.5, size=n)
    
    X = np.column_stack([np.ones(n), x1, x2])
    beta_true = np.array([2.5, 0.85, -0.42])  # Less trivial betas
    
    sigma_v, sigma_u = 0.45, 0.75  # Much noisier environment
    v = RNG.normal(0, sigma_v, n)
    u = np.abs(RNG.normal(0, sigma_u, n))  # True Half-Normal
    
    s = 1 if model_type == "production" else -1
    y = X @ beta_true + v - s * u
    
    sigma_sq_true = sigma_v**2 + sigma_u**2
    gamma_true = sigma_u**2 / sigma_sq_true
    
    return X, y, beta_true, sigma_sq_true, gamma_true


# =====================================================================
# DGP 2: Time Decay Panel (BC 1992) - The "Sticky" Panel Stress
# =====================================================================
def generate_bc92_data(n_entities=500, t_periods=5):
    """
    Complications:
    1. X has "Entity Fixed Effects" (sticky over time) + AR(1) style noise.
    2. Uses true scipy.stats.truncnorm instead of a hacky np.clip().
    3. Time decay parameter (eta) is very small, making it harder to detect.
    """
    N = n_entities * t_periods
    groups = np.repeat(np.arange(n_entities), t_periods)
    time = np.tile(np.arange(1, t_periods + 1), n_entities)
    
    # Generate X with entity-level "stickiness"
    x_base = RNG.normal(loc=10.0, scale=4.0, size=n_entities)
    x_it = np.repeat(x_base, t_periods) + RNG.normal(loc=0, scale=1.0, size=N)
    
    X = np.column_stack([np.ones(N), x_it])
    beta_true = np.array([3.14, 0.65])
    
    sigma_v, sigma_u = 0.35, 0.60
    mu_true, eta_true = 0.8, 0.05  # Very subtle time decay
    
    # Generate true Truncated Normal (bounded below at 0)
    a = (0 - mu_true) / sigma_u
    u_i = truncnorm.rvs(a, np.inf, loc=mu_true, scale=sigma_u, size=n_entities, random_state=RNG)
    
    # Apply time decay
    T_i = np.repeat(t_periods, N)
    h_it = np.exp(-eta_true * (time - T_i))
    u_it = np.repeat(u_i, t_periods) * h_it
    
    v_it = RNG.normal(0, sigma_v, N)
    y = X @ beta_true + v_it - u_it
    
    sigma_sq_true = sigma_v**2 + sigma_u**2
    gamma_true = sigma_u**2 / sigma_sq_true
    
    return X, y, groups, time, beta_true, sigma_sq_true, gamma_true, mu_true, eta_true


# =====================================================================
# DGP 3: Inefficiency Effects (BC 1995) - The Endogeneity-Lite Stress
# =====================================================================
def generate_bc95_data(N=500):
    """
    Complications:
    1. Z includes a binary dummy variable AND a continuous variable.
    2. X is correlated with Z (e.g., larger firms have different ownership).
    3. Row-by-row true Truncated Normal generation.
    """
    # Z1: Binary Dummy (e.g., Public vs Private)
    z1 = RNG.binomial(1, p=0.4, size=N)
    # Z2: Continuous (e.g., Age of firm)
    z2 = RNG.uniform(1.0, 20.0, size=N)
    Z = np.column_stack([np.ones(N), z1, z2])
    
    # X correlates with Z2 (Older firms are larger)
    x1 = 0.6 * z2 + RNG.normal(0, 3.0, N)
    X = np.column_stack([np.ones(N), x1])
    
    beta_true = np.array([1.5, 0.73])
    delta_true = np.array([-0.2, 0.5, -0.04]) # Z2 reduces inefficiency slightly
    
    sigma_v, sigma_u = 0.25, 0.55
    
    mu_it = Z @ delta_true
    
    # Generate observation-level true Truncated Normal
    a_it = (0 - mu_it) / sigma_u
    u_it = truncnorm.rvs(a_it, np.inf, loc=mu_it, scale=sigma_u, random_state=RNG)
    
    v_it = RNG.normal(0, sigma_v, N)
    y = X @ beta_true + v_it - u_it
    
    groups = np.arange(N)
    return X, y, Z, groups, beta_true, delta_true


# =====================================================================
# TEST SUITE
# =====================================================================

def test_als_production_recovery():
    """Test if CrossSectionalSFA recovers true parameters for Production."""
    X, y, b_true, sig2_true, gam_true = generate_als_data(model_type="production")
    
    model = CrossSectionalSFA(model_type="production")
    model.fit(X, y)
    
    # Assert Betas (Tolerance 0.05 is standard for MLE with N=2000)
    assert_allclose(model.coef_, b_true, atol=0.05)
    
    # Assert Variances
    assert_allclose(model.sigma_sq_, sig2_true, atol=0.05)
    assert_allclose(model.gamma_, gam_true, atol=0.1)
    
    # Assert Efficiency Scores are bounded (0, 1]
    te = model.score_efficiency(X, y)
    assert np.all((te > 0) & (te <= 1.0))


def test_als_cost_recovery():
    """Test if CrossSectionalSFA handles the Cost sign flip correctly."""
    X, y, b_true, sig2_true, gam_true = generate_als_data(model_type="cost")
    
    model = CrossSectionalSFA(model_type="cost")
    model.fit(X, y)
    
    assert_allclose(model.coef_, b_true, atol=0.05)
    
    te = model.score_efficiency(X, y)
    assert np.all((te > 0) & (te <= 1.0))


def test_bc92_panel_recovery():
    """Test if TimeDecayPanelSFA recovers time decay and panel parameters."""
    X, y, groups, time, b_true, sig2_true, gam_true, mu_true, eta_true = generate_bc92_data()
    
    model = TimeDecayPanelSFA()
    model.fit(X, y, groups=groups, time=time)
    
    # Slopes must be strict. Intercept gets wider tolerance due to mu-collinearity.
    assert_allclose(model.coef_[1:], b_true[1:], atol=0.05)
    assert_allclose(model.coef_, b_true, atol=0.15)
    assert_allclose(model.eta_, eta_true, atol=0.05)
    assert_allclose(model.sigma_sq_, sig2_true, atol=0.1)
    
    te = model.score_efficiency(X, y)
    assert len(te) == len(y)
    assert np.all((te > 0) & (te <= 1.0))


def test_bc95_effects_recovery():
    """Test if EffectsPanelSFA recovers one-step simultaneous effects."""
    X, y, Z, groups, b_true, d_true = generate_bc95_data()
    
    model = EffectsPanelSFA()
    model.fit(X, y, groups=groups, Z=Z)
    
    # Slopes must be strict. Intercept gets wider tolerance due to delta_0-collinearity.
    assert_allclose(model.coef_[1:], b_true[1:], atol=0.05)
    assert_allclose(model.coef_, b_true, atol=0.15)
    assert_allclose(model.delta_, d_true, atol=0.15)
    
    te = model.score_efficiency(X, y, Z=Z)
    assert len(te) == len(y)
    assert np.all((te > 0) & (te <= 1.0))


def test_edge_case_no_inefficiency():
    """STRESS TEST: If data is pure OLS (no u), gamma should trend toward 0."""
    n = 500
    X = np.column_stack([np.ones(n), RNG.standard_normal((n, 1))])
    y = X @ np.array([1.0, 0.5]) + RNG.normal(0, 0.2, n)  # Pure noise, no u
    
    model = CrossSectionalSFA()
    model.fit(X, y)
    
    # Gamma represents proportion of variance from inefficiency.
    # It should be very close to 0.
    assert model.gamma_ < 0.05
    
    # Efficiency scores should all be very close to 1.0
    te = model.score_efficiency(X, y)
    assert np.mean(te) > 0.95

# =====================================================================
# STRESS TESTS & EDGE CASES
# =====================================================================

def test_edge_case_deterministic_frontier():
    """
    STRESS TEST: Gamma approaches 1. 
    Almost all error is inefficiency (sig_v = 0.01, sig_u = 1.0).
    This pushes lambda (sig_u/sig_v) to 100, which can blow up the CDF 
    if the math isn't stable.
    """
    n = 1000
    X = np.column_stack([np.ones(n), RNG.standard_normal((n, 1))])
    b_true = np.array([2.0, 1.5])
    
    # Tiny noise, massive inefficiency
    v = RNG.normal(0, 0.01, n)
    u = np.abs(RNG.normal(0, 1.0, n))
    y = X @ b_true + v - u
    
    model = CrossSectionalSFA()
    model.fit(X, y)
    
    # The optimizer should realize that noise is basically zero
    assert model.gamma_ > 0.90
    assert_allclose(model.coef_, b_true, atol=0.1)


def test_bc92_unbalanced_panel():
    """
    STRESS TEST: Unbalanced Panel Data.
    Firms have different numbers of observations (from 1 to 10 periods).
    This proves that the `np.bincount` and `T_i_mapped` vectorization 
    doesn't hard-crash when array sizes differ per entity.
    """
    n_entities = 200
    # Randomly assign between 1 and 10 periods per entity
    periods = RNG.integers(1, 11, size=n_entities)
    N = periods.sum()
    
    groups = np.repeat(np.arange(n_entities), periods)
    time = np.concatenate([np.arange(1, p + 1) for p in periods])
    
    X = np.column_stack([np.ones(N), RNG.standard_normal((N, 1))])
    b_true = np.array([1.0, 0.5])
    
    # Generate True Inefficiency
    u_i = np.clip(RNG.normal(0.5, 0.3, n_entities), 0.001, None)
    
    # Broadcast and apply decay manually
    T_i = np.repeat(periods, periods)
    h_it = np.exp(-0.1 * (time - T_i))
    u_it = np.repeat(u_i, periods) * h_it
    
    v_it = RNG.normal(0, 0.1, N)
    y = X @ b_true + v_it - u_it
    
    model = TimeDecayPanelSFA()
    model.fit(X, y, groups=groups, time=time)
    
    # It should still recover the beta despite the unbalanced structure
    assert_allclose(model.coef_, b_true, atol=0.08)
    assert len(model.score_efficiency(X, y, groups=groups, time=time)) == N


def test_severe_multicollinearity():
    """
    STRESS TEST: Highly correlated regressors.
    x1 and x2 are 99% correlated. The MLE should still converge without 
    throwing a LinAlgError, even if the individual betas get wobbly.
    """
    n = 1000
    x1 = RNG.standard_normal(n)
    x2 = x1 * 0.99 + RNG.normal(0, 0.01, n)  # 99% correlation
    X = np.column_stack([np.ones(n), x1, x2])
    
    b_true = np.array([1.0, 2.0, -1.0])
    y = X @ b_true + RNG.normal(0, 0.1, n) - np.abs(RNG.normal(0, 0.3, n))
    
    model = CrossSectionalSFA()
    model.fit(X, y)
    
    # Check that it actually fitted and generated a valid log-likelihood
    assert np.isfinite(model.log_likelihood_)
    assert len(model.coef_) == 3


def test_api_input_validation():
    """
    STRESS TEST: Scikit-learn API compliance.
    Ensures the library catches bad data shapes before the math explodes.
    """
    model = CrossSectionalSFA()
    
    # 1. y is a 2D matrix instead of a 1D vector
    X = np.random.rand(100, 2)
    y_bad = np.random.rand(100, 2)
    with pytest.raises(ValueError):
        model.fit(X, y_bad)
        
    # 2. X and y have different row counts
    y_bad_len = np.random.rand(99)
    with pytest.raises(ValueError):
        model.fit(X, y_bad_len)
        
    # 3. BC95 called without Z
    model_effects = EffectsPanelSFA()
    y_good = np.random.rand(100)
    groups = np.arange(100)
    with pytest.raises(ValueError, match="requires Z"):
        model_effects.fit(X, y_good, groups=groups)
