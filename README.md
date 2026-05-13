# PanelSFA

Parametric Stochastic Frontier Analysis with a **scikit-learn** compatible API. This library implements foundational and state-of-the-art frontier models, enabling the estimation of technical efficiency in both cross-sectional and panel data settings.

## Installation

```bash
pip install -e .

```

**Dependencies:** `numpy>=1.23`, `scipy>=1.9`, `scikit-learn>=1.2`

---

## Models

| Class | Paper | Inefficiency Distribution | Key Extra Parameters |
| --- | --- | --- | --- |
| `CrossSectionalSFA` | ALS 1977 | Half-Normal | — |
| `TimeDecayPanelSFA` | BC 1992 | Truncated-Normal + time decay | `groups`, `time` |
| `EffectsPanelSFA` | BC 1995 | Truncated-Normal (effects) | `groups`, `Z` |
| `TrueFixedEffectsSFA` | Greene 2005 | Half-Normal (Intercepts profiled) | `groups` |
| `TrueRandomEffectsSFA` | Greene 2005 | MSL with Halton Draws | `groups`, `n_simulations` |
| `FourComponentSFA` | Kumbhakar et al. 2014 | Transient & Persistent Inefficiency | `groups` |

---

## Quick-start

### ALS 1977 – Cross-Sectional

The foundational stochastic frontier model using a half-normal distribution for inefficiency.

```python
from panelsfa import CrossSectionalSFA

model = CrossSectionalSFA(model_type="production")
model.fit(X, y)

print(model.coef_)      # Frontier coefficients (β)
print(model.gamma_)     # Signal-to-noise ratio (γ)
te = model.score_efficiency(X, y)  # Technical Efficiency ∈ (0, 1]

```

### BC92 – Time-Varying Decay Panel

Extends ALS to panel data, allowing inefficiency to evolve deterministically over time via the $\eta$ parameter.

```python
from panelsfa import TimeDecayPanelSFA

model = TimeDecayPanelSFA(model_type="production")
model.fit(X, y, groups=entity_ids, time=time_index)

print(model.eta_)  # η > 0 implies inefficiency decreases over time
te = model.score_efficiency(X, y, groups=entity_ids, time=time_index)

```

### BC95 – Inefficiency Effects Panel

Allows environmental covariates ($Z$) to directly explain the mean of the inefficiency distribution in a single-step estimation.

```python
from panelsfa import EffectsPanelSFA

# Z contains covariates like farmer age or education level
model = EffectsPanelSFA(model_type="production")
model.fit(X, y, groups=entity_ids, Z=Z_covariates)

print(model.delta_)  # Coefficients quantifying inefficiency drivers
te = model.score_efficiency(X, y, Z=Z_covariates)

```

### Greene (2005) – True Fixed Effects (TFE)

Handles unobserved, time-invariant heterogeneity by profiling out firm-specific intercepts $\alpha_i$, preventing them from being captured as inefficiency.

```python
from panelsfa import TrueFixedEffectsSFA

model = TrueFixedEffectsSFA(model_type="production")
model.fit(X, y, groups=entity_ids)

print(model.alphas_)  # Recovered firm-specific intercepts
te = model.score_efficiency(X, y)

```

### Greene (2005) – True Random Effects (TRE)

Uses **Maximum Simulated Likelihood** with Halton draws to integrate out a randomly distributed firm effect $w_i$.

```python
from panelsfa import TrueRandomEffectsSFA

model = TrueRandomEffectsSFA(n_simulations=500)
model.fit(X, y, groups=entity_ids)

print(model.sigma_w_sq_) # Variance of the random effect (heterogeneity)
te = model.score_efficiency(X, y)

```

### Kumbhakar et al. (2014) – Four-Component Model

Disentangles four distinct components: firm heterogeneity, random noise, persistent inefficiency, and transient inefficiency.

```python
from panelsfa import FourComponentSFA

model = FourComponentSFA(model_type="production")
model.fit(X, y, groups=entity_ids)

# Multi-dimensional efficiency scoring
eff = model.score_efficiency(X, y, groups=entity_ids)
print(eff['transient'])   # Short-term deviations
print(eff['persistent'])  # Structural failures
print(eff['overall'])     # Unified efficiency score

```

---

## Numerical Engine & Parameterization

To ensure convergence on non-convex likelihood surfaces, `PanelSFA` utilizes an **OLS Warm-Start** with an intercept shift and reparameterizes constrained variables to an unconstrained space.

| Natural Parameter | Optimizer Parameter | Transform Back |
| --- | --- | --- |
| $\sigma^2 > 0$ | $\ln(\sigma^2)$ | `exp(·)` |
| $\gamma \in (0, 1)$ | $\text{logit}(\gamma)$ | `sigmoid(·)` |
| $\beta, \mu, \eta, \delta, \alpha$ | identity | — |

* 
**Optimizer:** L-BFGS-B (Limited-memory BFGS) for high-dimensional stability.


* **Efficiency Scoring:** Jondrow-Lovell-Materov-Schmidt (JLMS) conditional expectation $E[u|\varepsilon]$.

---
