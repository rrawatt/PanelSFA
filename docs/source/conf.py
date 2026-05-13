# Configuration file for the Sphinx documentation builder.

import os
import sys
# Tell Sphinx to look two folders up to find the panelsfa python package
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'PanelSFA'
copyright = '2026, Adrish Jana, Rohit Rawat, Sukin S'
author = 'Adrish Jana, Rohit Rawat, Sukin S'
release = '0.1.2'
master_doc = 'index'

# -- General configuration ---------------------------------------------------

# These are the critical extensions for your library
extensions = [
    'sphinx.ext.autodoc',       # Pulls docstrings from your Python code
    'sphinx.ext.napoleon',      # Parses Google/NumPy-style docstrings
    'sphinx.ext.mathjax',       # Renders LaTeX math (crucial for SFA)
    'sphinx.ext.viewcode',      # Adds "[source]" links to your docs
    'myst_parser'               # Allows you to write docs in Markdown (.md)
]

# Tell Sphinx to accept both .rst and .md files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# Use the professional Read the Docs theme instead of the default Alabaster
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']