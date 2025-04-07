import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# Project information
project = 'Folps'
author = 'Hernan Noriega'
release = '0.0.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
]

# HTML theme
html_theme = 'sphinx_rtd_theme'

# Paths
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
