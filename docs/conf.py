import os
import sys
from datetime import datetime

# Add project root so autodoc can import the package
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'FOLPSpipe'
author = 'Hernán E. Noriega'
release = '0.0.0'
copyright = f"{datetime.now().year}, {author}"

# Sphinx extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# Mock imports to avoid failures when optional heavy dependencies are not installed
autodoc_mock_imports = ['jax', 'jax.numpy', 'classy', 'interpax']

# Intersphinx mappings for common projects
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# HTML theme: prefer sphinx_rtd_theme if available, otherwise fall back to a builtin theme
try:
    import sphinx_rtd_theme
    extensions.append('sphinx_rtd_theme')
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
except Exception:
    html_theme = 'alabaster'
    html_theme_path = []

# Logo: point to the repository root image
# Use a path relative to the docs directory
html_logo = os.path.join('..', 'folps_logo.png')

# Documentation language
language = 'en'

# Paths and excludes
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Autodoc options
autodoc_member_order = 'groupwise'
autodoc_typehints = 'description'

# Additional HTML options
html_static_path = ['_static']

# Autosummary: generate API stubs automatically
autosummary_generate = True
