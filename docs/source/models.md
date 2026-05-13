# Supported Models

PanelSFA implements models ranging from foundational cross-sectional frameworks to advanced multidimensional panel estimators.

## Cross-Sectional Models
* **Aigner, Lovell, and Schmidt (ALS 1977)**: The foundational SFA model. Assumes a Half-Normal distribution for the inefficiency term.

## Panel Models (Deterministic)
* **Battese and Coelli (BC 1992)**: Assumes inefficiency follows a Truncated-Normal distribution that decays (or grows) deterministically over time via an $\eta$ parameter.
* **Battese and Coelli (BC 1995)**: A single-step estimator where the mean of the inefficiency distribution is a linear function of environmental covariates ($Z$). Eliminates two-step bias.

## Panel Models (Unobserved Heterogeneity)
* **Greene (2005) True Fixed Effects (TFE)**: Profiles out time-invariant firm intercepts ($\alpha_i$) using an internal Newton-Raphson loop. Ensures structural differences between firms are not mistakenly scored as inefficiency.
* **Greene (2005) True Random Effects (TRE)**: Integrates out a firm-level random effect $w_i$ using Maximum Simulated Likelihood via Halton sequences.

## The Four-Component Model
* **Kumbhakar, Lien, and Hardaker (2014)**: A three-step pseudo-likelihood estimator that completely disentangles firm heterogeneity, random noise, transient (short-term) inefficiency, and persistent (structural) inefficiency.