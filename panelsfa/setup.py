from setuptools import setup, find_packages

setup(
    name="panelsfa",
    version="0.1.0",
    description="Parametric Stochastic Frontier Analysis with a scikit-learn API",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.23",
        "scipy>=1.9",
        "scikit-learn>=1.2",
    ],
    python_requires=">=3.9",
)
