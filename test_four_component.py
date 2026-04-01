import numpy as np
import pytest
from numpy.testing import assert_allclose
from panelsfa import FourComponentSFA

RNG = np.random.default_rng(42)

def generate_four_component_data(n_entities=150, t_periods=10):
    """
    Generates all four stochastic components.
    X deliberately excludes a global intercept as required by the library.
    """
    N = n_entities * t_periods
    groups = np.repeat(np.arange(n_entities), t_periods)
    
    # X without intercept
    X = RNG.standard_normal((N, 2))
    beta_true = np.array([1.2, -0.5])
    beta_0_true = 2.5
    
    sigma_mu, sigma_v = 0.4, 0.3
    sigma_eta, sigma_u = 0.5, 0.6
    
    # Time-Invariant Components
    mu_i = RNG.normal(0, sigma_mu, n_entities)
    eta_i = np.abs(RNG.normal(0, sigma_eta, n_entities))
    
    # Time-Varying Components
    v_it = RNG.normal(0, sigma_v, N)
    u_it = np.abs(RNG.normal(0, sigma_u, N))
    
    y = beta_0_true + X @ beta_true + mu_i[groups] + v_it - eta_i[groups] - u_it
    
    return X, y, groups, beta_true, beta_0_true, sigma_v**2, sigma_u**2, sigma_mu**2, sigma_eta**2


def test_four_component_recovery():
    """Test if FourComponentSFA cleanly isolates all four variances via the Three-Step method."""
    from panelsfa import FourComponentSFA
    
    X, y, groups, b_true, b0_true, sig_v2, sig_u2, sig_mu2, sig_eta2 = generate_four_component_data()
    
    model = FourComponentSFA(model_type="production")
    model.fit(X, y, groups=groups)
    
    # Step 1 (Within-estimator) should be extremely precise
    assert_allclose(model.coef_, b_true, atol=0.05)
    
    # Overall intercept
    assert_allclose(model.intercept_, b0_true, atol=0.2)
    
    # Check that variances are populated and strictly positive
    assert model.sigma_v_sq_ > 0
    assert model.sigma_u_sq_ > 0
    assert model.sigma_mu_sq_ > 0
    assert model.sigma_eta_sq_ > 0
    
    # Ensure the multi-dimensional TE scoring dictionary works
    eff_dict = model.score_efficiency(X, y, groups=groups)
    assert "transient" in eff_dict
    assert "persistent" in eff_dict
    assert "overall" in eff_dict
    
    assert len(eff_dict["transient"]) == len(y)
    assert len(eff_dict["persistent"]) == len(y)  # Broadcasted to obs level
    
    # OTE must strictly be the product of PTE and TTE
    assert_allclose(eff_dict["overall"], eff_dict["persistent"] * eff_dict["transient"], rtol=1e-5)
    assert np.all((eff_dict["overall"] > 0) & (eff_dict["overall"] <= 1.0))

def generate_four_comp_variance_stress_data(n_entities=1000, t_periods=8, scenario="no_transient"):
    """
    Generates extreme edge cases for the Four-Component Model.
    n_entities increased to 1000, and signal-to-noise ratio boosted 
    to guarantee the estimator detects the structural skewness.
    """
    N = n_entities * t_periods
    groups = np.repeat(np.arange(n_entities), t_periods)
    
    X = RNG.standard_normal((N, 1))
    beta_true = np.array([2.0])
    beta_0_true = 1.5
    
    # Base variances: Boost eta and u, lower mu and v to ensure clear skewness
    sigma_mu, sigma_eta = 0.2, 0.8
    sigma_v, sigma_u = 0.2, 0.6
    
    if scenario == "no_transient":
        sigma_u = 0.0
    elif scenario == "no_heterogeneity":
        sigma_mu = 0.0
    elif scenario == "perfect_efficiency":
        sigma_u = 0.0
        sigma_eta = 0.0

    mu_i = RNG.normal(0, sigma_mu, n_entities)
    eta_i = np.abs(RNG.normal(0, sigma_eta, n_entities))
    
    v_it = RNG.normal(0, sigma_v, N)
    u_it = np.abs(RNG.normal(0, sigma_u, N)) if sigma_u > 0 else np.zeros(N)
    
    y = beta_0_true + X @ beta_true + mu_i[groups] + v_it - eta_i[groups] - u_it
    
    return X, y, groups, beta_true


def test_four_component_missing_inefficiency():
    """
    STRESS TEST: Four-Component Model with zero persistent inefficiency.
    If eta_i is zero, the 3rd step (Persistent SFA) should estimate 
    gamma_pers near 0 without throwing Log-Likelihood NaN errors.
    """
    from panelsfa import FourComponentSFA
    
    # INCREASED to 1000 to absolutely crush any finite-sample wrong skewness
    n_entities = 1000 
    t_periods = 8
    N = n_entities * t_periods
    groups = np.repeat(np.arange(n_entities), t_periods)
    
    X = RNG.standard_normal((N, 1))
    
    # Tiny mu, zero eta, small v, large u
    mu_i = RNG.normal(0, 0.2, n_entities)
    v_it = RNG.normal(0, 0.2, N)
    u_it = np.abs(RNG.normal(0, 0.6, N))
    
    y = 2.0 + X[:, 0] * 1.5 + mu_i[groups] + v_it - u_it
    
    model = FourComponentSFA()
    model.fit(X, y, groups=groups)
    
    # The variance of the persistent inefficiency should be mathematically tiny
    assert model.sigma_eta_sq_ < 0.05
    
    eff_dict = model.score_efficiency(X, y, groups=groups)
    
    # Because there is no persistent inefficiency, PTE should be highly efficient.
    assert np.mean(eff_dict["persistent"]) > 0.85


def generate_four_comp_unbalanced_data(n_entities=150):
    """
    Generates unbalanced panel data for the Four-Component model.
    Firms have a random number of observed periods (between 2 and 15).
    """
    periods = RNG.integers(2, 16, size=n_entities)
    N = periods.sum()
    groups = np.repeat(np.arange(n_entities), periods)
    
    X = RNG.standard_normal((N, 2))
    beta_true = np.array([1.8, -0.4])
    beta_0_true = 3.0
    
    mu_i = RNG.normal(0, 0.4, n_entities)
    eta_i = np.abs(RNG.normal(0, 0.6, n_entities))
    
    v_it = RNG.normal(0, 0.3, N)
    u_it = np.abs(RNG.normal(0, 0.5, N))
    
    y = beta_0_true + X @ beta_true + mu_i[groups] + v_it - eta_i[groups] - u_it
    
    return X, y, groups, beta_true, N


def generate_four_comp_variance_stress_data(n_entities=1000, t_periods=8, scenario="no_transient"):
    """
    Generates extreme edge cases for the Four-Component Model.
    n_entities increased to 1000, and signal-to-noise ratio boosted 
    to guarantee the estimator detects the structural skewness.
    """
    N = n_entities * t_periods
    groups = np.repeat(np.arange(n_entities), t_periods)
    
    X = RNG.standard_normal((N, 1))
    beta_true = np.array([2.0])
    beta_0_true = 1.5
    
    # Base variances: Boost eta and u, lower mu and v to ensure clear skewness
    sigma_mu, sigma_eta = 0.2, 0.8
    sigma_v, sigma_u = 0.2, 0.6
    
    if scenario == "no_transient":
        sigma_u = 0.0
    elif scenario == "no_heterogeneity":
        sigma_mu = 0.0
    elif scenario == "perfect_efficiency":
        sigma_u = 0.0
        sigma_eta = 0.0

    mu_i = RNG.normal(0, sigma_mu, n_entities)
    eta_i = np.abs(RNG.normal(0, sigma_eta, n_entities))
    
    v_it = RNG.normal(0, sigma_v, N)
    u_it = np.abs(RNG.normal(0, sigma_u, N)) if sigma_u > 0 else np.zeros(N)
    
    y = beta_0_true + X @ beta_true + mu_i[groups] + v_it - eta_i[groups] - u_it
    
    return X, y, groups, beta_true


def test_four_component_unbalanced_panel():
    """
    STRESS TEST: Unbalanced Panel in the Four-Component Model.
    Ensures that the Step 1 Within-Transformation (which heavily relies on 
    `np.bincount` and array divisions) correctly handles varying $T_i$ 
    without throwing shape mismatch errors.
    """
    from panelsfa import FourComponentSFA
    
    X, y, groups, b_true, N = generate_four_comp_unbalanced_data()
    
    model = FourComponentSFA(model_type="production")
    model.fit(X, y, groups=groups)
    
    # Step 1 slopes should still be highly accurate
    assert_allclose(model.coef_, b_true, atol=0.08)
    
    # Check that scoring respects the unbalanced observation count
    eff_dict = model.score_efficiency(X, y, groups=groups)
    assert len(eff_dict["transient"]) == N
    assert len(eff_dict["overall"]) == N
    assert np.all(np.isfinite(eff_dict["overall"]))


def test_four_component_zero_transient_inefficiency():
    """
    STRESS TEST: Firms have zero day-to-day transient inefficiency (u=0),
    but suffer from systemic persistent inefficiency (eta>0).
    Ensures Step 2 correctly identifies gamma_trans -> 0, while Step 3 
    still successfully isolates the structural inefficiency.
    """
    from panelsfa import FourComponentSFA
    
    X, y, groups, b_true = generate_four_comp_variance_stress_data(scenario="no_transient")
    
    model = FourComponentSFA(model_type="production")
    model.fit(X, y, groups=groups)
    
    # Transient inefficiency variance should collapse to near zero
    assert model.sigma_u_sq_ < 0.05
    
    # Persistent inefficiency variance should remain robustly positive
    assert model.sigma_eta_sq_ > 0.1
    
    eff_dict = model.score_efficiency(X, y, groups=groups)
    
    # Transient TE should be extremely close to 1.0 (perfect day-to-day)
    assert np.mean(eff_dict["transient"]) > 0.95
    
    # Overall TE should be entirely driven by the Persistent TE
    assert_allclose(eff_dict["overall"], eff_dict["persistent"], atol=0.05)


def test_four_component_pure_structural_inefficiency():
    """
    STRESS TEST: Firms have zero unobserved heterogeneity (mu=0).
    The Step 3 SFA receives an intercept composed entirely of deterministic 
    constant and half-normal persistent inefficiency. Ensures Step 3 
    pushes gamma_pers -> 1 without crashing the L-BFGS-B optimizer.
    """
    from panelsfa import FourComponentSFA
    
    X, y, groups, b_true = generate_four_comp_variance_stress_data(scenario="no_heterogeneity")
    
    model = FourComponentSFA(model_type="production")
    model.fit(X, y, groups=groups)
    
    # Firm heterogeneity variance should collapse to near zero
    assert model.sigma_mu_sq_ < 0.05
    
    # Persistent inefficiency variance should capture the remaining signal
    assert model.sigma_eta_sq_ > 0.1
    
    eff_dict = model.score_efficiency(X, y, groups=groups)
    assert np.all((eff_dict["persistent"] > 0) & (eff_dict["persistent"] <= 1.0))


def test_four_component_perfect_rational_agents():
    """
    STRESS TEST: Both transient and persistent inefficiency are exactly zero.
    This simulates perfect rational agents facing only random noise.
    Both internal SFA models must gracefully degrade to pure OLS without 
    inverse-mills ratio division errors.
    """
    from panelsfa import FourComponentSFA
    
    X, y, groups, b_true = generate_four_comp_variance_stress_data(scenario="perfect_efficiency")
    
    model = FourComponentSFA(model_type="production")
    model.fit(X, y, groups=groups)
    
    # Both inefficiency variances should be mathematically tiny
    assert model.sigma_u_sq_ < 0.05
    assert model.sigma_eta_sq_ < 0.05
    
    eff_dict = model.score_efficiency(X, y, groups=groups)
    
    # All efficiencies should be virtually 1.0
    assert np.mean(eff_dict["transient"]) > 0.95
    assert np.mean(eff_dict["persistent"]) > 0.95
    assert np.mean(eff_dict["overall"]) > 0.95