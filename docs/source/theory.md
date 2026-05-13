# Theory & Methodology

PanelSFA is built on a rigorous maximum likelihood architecture designed to handle the notoriously non-convex likelihood surfaces associated with Stochastic Frontier Analysis.

## The Error Decomposition

A generic stochastic frontier is defined as:
$$y_{it} = x_{it}^{\prime}\beta + v_{it} - s \cdot u_{it}$$

Where:
* $v_{it} \sim \mathcal{N}(0, \sigma_v^2)$ is the symmetric, random noise (idiosyncratic shock).
* $u_{it} \geq 0$ is the asymmetric, non-negative inefficiency term.
* $s = +1$ for a **production** frontier (inefficiency reduces output).
* $s = -1$ for a **cost** frontier (inefficiency increases cost).

## Numerical Stability Engineering

To ensure the L-BFGS-B optimizer converges reliably, the library utilizes several internal safeguards.

### 1. OLS Warm-Start and Intercept Shift
Optimizing directly from random initialization often leads to local minima. PanelSFA first runs an OLS regression to establish consistent slope parameters. Because OLS estimates the conditional mean, the intercept is then mathematically shifted to the edge of the data cloud:
$$\beta_{0, start} = \beta_{0, OLS} + s \cdot \sqrt{\frac{\hat{\sigma}^2}{\pi}}$$
This places the optimizer in the "basin of attraction" for the global maximum.

### 2. Unconstrained Reparameterization
Optimizers struggle with hard boundaries (e.g., $\sigma^2 > 0$). PanelSFA maps all constrained parameters to the unconstrained real number line ($\mathbb{R}$) using link functions:
* **Total Variance:** Optimized as $\zeta = \ln(\sigma^2)$. The derivative $\exp(\zeta)$ prevents the "vanishing gradient" problem near zero.
* **Variance Ratio:** Optimized as $\psi = \ln(\frac{\gamma}{1-\gamma})$. This ensures the recovered $\gamma = \sigma_u^2 / \sigma^2$ is strictly bound between 0 and 1.

### 3. The Waldman (1982) Diagnostic
Before the numerical engine engages, the controller evaluates the third moment ($\mu_3$) of the OLS residuals. If the residuals exhibit the "wrong skewness" (e.g., positive skew for a production function), the model halts, as Waldman (1982) proved that no local maximum exists for $\sigma_u > 0$ under these conditions.