FOLPSpipe documentation
=======================

Short description
-----------------

FOLPSpipe is a Python package for fast, accurate computation of the redshift-space power spectrum (and bispectrum) in cosmologies with massive neutrinos. The code supports two backends:

- NumPy (default) — stable, widely available CPU implementation.
- JAX — optional accelerated implementation that can run on CPU, GPU or Apple Metal when available.

Important: the package is written to be backend-agnostic; set the environment variable `FOLPS_BACKEND` to `numpy` or `jax` before importing `folps` to select the desired backend.

Quick install
-------------

See `requirements.txt` for required Python packages. Typical installation sequence:

.. code-block:: bash

   git clone <your-repo>
   cd FOLPSpipe
   python -m pip install -r requirements.txt

Quick example
-------------

Minimal example (see the Examples page for a runnable notebook-style script):

.. code-block:: python

   import os
   os.environ['FOLPS_BACKEND'] = 'numpy'  # or 'jax'
   from folps import *

   # load precomputed linear pk or use CLASS via cosmo_class.run_class
   data = {'k': k_array, 'pk': pk_array}

   matrix = MatrixCalculator()
   mmatrices = matrix.get_mmatrices()

   nonlinear = NonLinearPowerSpectrumCalculator(mmatrices=mmatrices, kernels='fk', z=0.3)
   table, table_now = nonlinear.calculate_loop_table(k=data['k'], pklin=data['pk'], cosmo=None)

Main package layout
-------------------

- `folps/folps.py`: core high-level routines, backend manager and main calculators.
- `folps/cosmo_class.py`: helpers to compute or load linear power spectra (CLASS wrapper).
- `folps/tools.py`: NumPy-based utility functions (interpolation, integration, Legendre polynomials, extrapolation, etc.).
- `folps/tools_jax.py`: JAX equivalents of the utilities for accelerated execution.
- Example notebooks: `run_folps_numpy.ipynb` and `run_folps_jax.ipynb`.

API and examples
----------------

The API reference is generated from docstrings. See the "API Reference" for module-level details, and the "Examples" section for runnable examples extracted from the notebooks.

.. toctree::
   :maxdepth: 2

   Installation
   Requirements
   Quickstart
   usage
   examples
   api
   Classes
   Changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`