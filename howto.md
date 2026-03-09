

# PanelSFA: Parametric Stochastic Frontier Analysis in Python

`PanelSFA` is a high-performance, purely parametric **Stochastic Frontier Analysis (SFA)** engine built natively in Python. It strictly follows standard econometric parameterizations and utilizes a `scikit-learn` compatible API.

It is designed to replicate the robustness of **Stata** and **Frontier 4.1**, but with modern, vectorized Python optimizations, filling the gap for native longitudinal (panel data) efficiency benchmarking.

---

## 1. Core Architecture & Design Philosophy

`PanelSFA` intentionally abandons "formula-style" strings (like R) in favor of **explicit matrix inputs**. All models inherit from the standard `scikit-learn` framework.

* **No Magic Intercepts:** You must manually provide an intercept column (a column of 1.0s) in your X and Z matrices.
* **Unconstrained Optimization:** Internally, variances are optimized using logarithms, and ratios using logit transforms, avoiding the boundaries that crash other SFA libraries. The user-facing attributes are automatically transformed back to their natural economic scale.
* **Trailing Underscores:** Following the `scikit-learn` convention, all estimated parameters end with an underscore (e.g., `coef_`, `gamma_`).

---

## 2. Global Model Initialization

All three SFA models share the same base initialization arguments.

```python
from panelsfa import TimeDecayPanelSFA

model = TimeDecayPanelSFA(model_type="production", max_iter=2000, tol=1e-8)

```

### Parameters:

| Parameter | Type | Description |
| --- | --- | --- |
| **model_type** | `str` | **"production"** (Default): The frontier is a ceiling. Inefficiency subtracts from the frontier (y = X * beta + v - u). <br>

<br> **"cost"**: The frontier is a floor. Inefficiency adds to the frontier (y = X * beta + v + u). |
| **max_iter** | `int` | Maximum iterations for the L-BFGS-B optimizer. Default is 2000. |
| **tol** | `float` | Tolerance for log-likelihood convergence. Default is 1e-8. |

---

## 3. The Models

### Model 1: CrossSectionalSFA (ALS 1977)

Estimates a standard cross-sectional SFA using a **Half-Normal** inefficiency distribution.

**When to use:** You have a single snapshot of data (no time element).

```python
from panelsfa import CrossSectionalSFA
import numpy as np

# X must be an (N, k) matrix, y must be an (N,) vector
model = CrossSectionalSFA(model_type="production")
model.fit(X, y)

```

**Fitted Attributes:**

* `model.coef_`: The beta coefficients for the frontier variables (array).
* `model.sigma_sq_`: Total variance, sigma^2 = sigma_v^2 + sigma_u^2 (float).
* `model.gamma_`: The variance ratio, gamma = sigma_u^2 / sigma^2 (float). As gamma approaches 1, inefficiency dominates the noise.

---

### Model 2: TimeDecayPanelSFA (Battese & Coelli 1992)

Estimates a panel SFA model using a **Truncated-Normal** distribution with a built-in time-decay mechanism.

* **When to use:** You have longitudinal data and want to know if firms are becoming more or less efficient over time.

```python
from panelsfa import TimeDecayPanelSFA

# groups: (N,) array of firm IDs
# time: (N,) array of time periods (e.g., 2010, 2011...)
model = TimeDecayPanelSFA(model_type="production")
model.fit(X, y, groups=bank_ids, time=years)

```

**Fitted Attributes (in addition to ALS attributes):**

* `model.mu_`: The mean (mu) of the underlying Truncated Normal distribution.
* `model.eta_`: The time-decay parameter (eta).
* If **eta_ > 0**: Inefficiency is decreasing (firms are improving).
* If **eta_ < 0**: Inefficiency is increasing (firms are worsening).



---

### Model 3: EffectsPanelSFA (Battese & Coelli 1995)

A simultaneous one-step Maximum Likelihood estimator. Instead of a uniform time decay, it models the mean of the inefficiency distribution as a direct function of external policy variables (Z).

* **When to use:** You want to test *why* firms are inefficient (e.g., does private ownership reduce inefficiency?).

```python
from panelsfa import EffectsPanelSFA

# Z: (N, m) matrix of inefficiency drivers (must include intercept if desired)
model = EffectsPanelSFA(model_type="production")
model.fit(X, y, groups=bank_ids, Z=Z_matrix)

```

**Fitted Attributes (in addition to ALS attributes):**

* `model.delta_`: The delta coefficients mapping the Z variables to inefficiency (array). A negative delta implies the variable reduces inefficiency.

---

## 4. Diagnostics and Post-Estimation

Once any model is fitted, the following universal methods and properties become available.

### Technical Efficiency Scoring (JLMS)

You can calculate the firm-specific technical efficiency score using the Jondrow, Lovell, Materov, and Schmidt (1982) estimator.

```python
# Returns an array of scores between (0, 1]
# 1.0 = Perfectly efficient
te_scores = model.score_efficiency(X, y)

# For panel models, pass the structural arrays:
te_scores_panel = model.score_efficiency(X, y, groups=bank_ids, time=years)
te_scores_effects = model.score_efficiency(X, y, Z=Z_matrix)

```

### Information Criteria

For model selection and hypothesis testing (e.g., comparing OLS to SFA, or comparing Cobb-Douglas to Translog).

```python
print(f"Log-Likelihood: {model.log_likelihood_}")
print(f"AIC: {model.aic_}")
print(f"BIC: {model.bic_}")

```

### Prediction

To get the deterministic frontier (the optimal theoretical output without noise or inefficiency):

```python
optimal_output = model.predict(X) # Returns X @ beta

```

---

## 5. Typical Workflow Example

Here is how a researcher prepares data and runs an analysis:

```python
import numpy as np
import pandas as pd
from panelsfa import TimeDecayPanelSFA

# 1. Load Data
df = pd.read_csv("indian_banks_panel.csv")

# 2. Extract Vectors and Matrices
y = df['log_loans'].values
groups = df['bank_id'].values
time = df['year'].values

# 3. Build X Matrix (Manually add intercept!)
X = np.column_stack([
    np.ones(len(df)),           # Intercept
    df['log_labor'].values,
    df['log_capital'].values
])

# 4. Fit Model
model = TimeDecayPanelSFA(model_type="production")
model.fit(X, y, groups=groups, time=time)

# 5. Review Results
print(f"Frontier Coefs (Beta): {model.coef_}")
print(f"Time Trend (Eta):      {model.eta_}")

# 6. Score Firms
df['efficiency'] = model.score_efficiency(X, y, groups=groups, time=time)
print(df.groupby('bank_type')['efficiency'].mean())

```

---

