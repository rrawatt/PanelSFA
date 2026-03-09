# PanelSFA

Parametric Stochastic Frontier Analysis with a scikit-learn API.  
Implements ALS (1977), Battese & Coelli (1992), and Battese & Coelli (1995).

## Installation

```bash
pip install -r requirements.txt .
```

**Dependencies:** `numpy>=1.23`, `scipy>=1.9`, `scikit-learn>=1.2`

---

## Models

| Class | Paper | Inefficiency Distribution | Key Extra Parameters |
|---|---|---|---|
| `CrossSectionalSFA` | ALS 1977 | Half-Normal | — |
| `TimeDecayPanelSFA` | BC 1992 | Truncated-Normal + time decay | `groups`, `time`, `mu_`, `eta_` |
| `EffectsPanelSFA` | BC 1995 | Truncated-Normal (effects) | `groups`, `Z`, `delta_` |

---

## Quick-start

### ALS 1977 – Cross-Sectional

```python
import numpy as np
from panelsfa import CrossSectionalSFA

rng = np.random.default_rng(0)
n   = 200
X   = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
y   = X @ [2.0, 0.5, -0.3] + rng.standard_normal(n) * 0.3

model = CrossSectionalSFA(model_type="production")
model.fit(X, y)

print(model.coef_)          # β
print(model.gamma_)         # σ_u² / σ²
print(model.sigma_sq_)      # σ²
print(model.aic_, model.bic_)

te = model.score_efficiency(X, y)   # TE ∈ (0, 1]
```

### BC92 – Time-Varying Decay Panel

```python
from panelsfa import TimeDecayPanelSFA

# groups and time must be aligned with rows of X, y
model = TimeDecayPanelSFA(model_type="production")
model.fit(X, y, groups=entity_ids, time=time_index)

print(model.eta_)   # >0 → inefficiency falls over time

te = model.score_efficiency(X, y, groups=entity_ids, time=time_index)
```

### BC95 – Inefficiency Effects Panel

```python
from panelsfa import EffectsPanelSFA

# Z: (N, m) matrix of firm-level inefficiency drivers
Z = np.column_stack([np.ones(N), firm_age, ownership_dummy])

model = EffectsPanelSFA(model_type="production")
model.fit(X, y, groups=entity_ids, Z=Z)

print(model.delta_)   # δ coefficients on Z

te = model.score_efficiency(X, y, Z=Z)
```

---

## Parameterisation (unconstrained optimisation)

All variance/ratio parameters are reparameterised so that `scipy.optimize.minimize`
operates over an unconstrained space. The class attributes always report
natural-scale values.

| Natural param | Optimiser param | Transform back |
|---|---|---|
| σ² > 0 | ln σ² | `exp(·)` |
| γ ∈ (0, 1) | logit γ | `sigmoid(·)` |
| β, μ, η, δ | identity | — |

## Cost vs. Production frontier

Set `model_type="cost"` to flip the sign of the inefficiency term (s = −1).
Technical Efficiency is always returned as `exp(−E[u|ε]) ∈ (0, 1]`.

---

## Directory layout

```
panelsfa/
├── __init__.py
├── base.py              # _BaseSFA: transforms, AIC/BIC, score_efficiency
├── cross_sectional.py   # ALS 1977
├── time_decay.py        # BC 1992
└── effects_panel.py     # BC 1995
setup.py
README.md
```
