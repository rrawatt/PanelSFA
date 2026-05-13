# Quickstart Guide

**PanelSFA** models inherit from `sklearn.base.BaseEstimator`, meaning they follow the standard `fit()` and `predict()` design pattern.

## Cross-Sectional Data (ALS 1977)

Here is a minimum working example of estimating a production frontier using cross-sectional data.

```python
import numpy as np
from panelsfa import CrossSectionalSFA

# 1. Generate Synthetic Data
rng = np.random.default_rng(42)
N = 500
X = np.column_stack([np.ones(N), rng.standard_normal((N, 2))])
true_beta = [1.5, 0.4, 0.8]

# y = Xβ + v - u (Production Frontier)
v = rng.normal(0, 0.2, N)
u = np.abs(rng.normal(0, 0.5, N))
y = X @ true_beta + v - u

# 2. Fit the Model
model = CrossSectionalSFA(model_type="production")
model.fit(X, y)

# 3. View Results
print(f"Estimated Coefficients: {model.coef_}")
print(f"Variance Ratio (Gamma): {model.gamma_}")

# 4. Score Efficiency (JLMS)
te = model.score_efficiency(X, y)
print(f"Average Technical Efficiency: {te.mean():.4f}")
```

## Panel Data (True Fixed Effects)
For panel data, models like Greene's True Fixed Effects require a groups array to identify the entity (e.g., firm, farm, or country) for each observation.

```Python
from panelsfa import TrueFixedEffectsSFA

# X, y, and entity_ids must be aligned row-by-row
model = TrueFixedEffectsSFA(model_type="cost") # Note: Cost frontier
model.fit(X, y, groups=entity_ids)

# Extract the profiled firm-specific intercepts
print(model.alphas_)
```