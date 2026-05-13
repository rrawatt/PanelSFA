.. PanelSFA documentation master file

PanelSFA: High-Performance Stochastic Frontier Analysis in Python
================================================================

.. image:: https://img.shields.io/pypi/v/panelsfa.svg
   :target: https://pypi.org/project/panelsfa/
   :alt: PyPI Version

**PanelSFA** is a modern Python library designed to fill the "Python Gap" in econometric software[cite: 33, 40]. It brings production-grade Stochastic Frontier Analysis (SFA) to the scientific computing ecosystem with a focus on numerical stability, scikit-learn compatibility, and high-performance panel data estimation[cite: 34, 113, 179].

By replicating the capabilities of proprietary software like Stata, **PanelSFA** democratizes technical efficiency analysis for researchers and practitioners alike[cite: 41, 171, 178].

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

Key Features
------------

* **Comprehensive Model Suite**: Implements foundational models like ALS (1977) and BC95, alongside advanced panel estimators like Greene's True Fixed Effects and the Four-Component model[cite: 41, 187, 189].
* **Scikit-learn API**: Designed with an Estimator API for seamless integration into machine learning pipelines and hyperparameter searches[cite: 114, 115, 179].
* **Numerical Stability**: Utilizes OLS warm-starts, parameter reparameterization (log/logit links), and L-BFGS-B optimization for reliable convergence on complex likelihood surfaces[cite: 119, 145, 146, 148].
* **Performance**: Optimized for large panel datasets using vectorized operations and efficient memory management[cite: 113].

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Models & Methodology

   models
   theory
   benchmarks

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`