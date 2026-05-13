.. PanelSFA documentation master file

PanelSFA: High-Performance Stochastic Frontier Analysis in Python
================================================================

.. image:: https://img.shields.io/pypi/v/panelsfa.svg
   :target: https://pypi.org/project/panelsfa/
   :alt: PyPI Version

**PanelSFA** is a modern Python library designed to fill the "Python Gap" in econometric software. It brings production-grade Stochastic Frontier Analysis (SFA) to the scientific computing ecosystem with a focus on numerical stability, scikit-learn compatibility, and high-performance panel data estimation.

By replicating the capabilities of proprietary software like Stata, **PanelSFA** democratizes technical efficiency analysis for researchers and practitioners alike.

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

Key Features
------------

* **Comprehensive Model Suite**: Implements foundational models like ALS (1977) and BC95, alongside advanced panel estimators like Greene's True Fixed Effects and the Four-Component model.
* **Scikit-learn API**: Designed with an Estimator API for seamless integration into machine learning pipelines and hyperparameter searches.
* **Numerical Stability**: Utilizes OLS warm-starts, parameter reparameterization (log/logit links), and L-BFGS-B optimization for reliable convergence on complex likelihood surfaces.
* **Performance**: Optimized for large panel datasets using vectorized operations and efficient memory management.

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