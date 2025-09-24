Requirements
============

Minimum prerequisites
---------------------

- Python 3.8+ recommended.
- pip installed.

Core Python packages (see `requirements.txt`) include:

- numpy
- scipy
- sphinx (for building docs)
- sphinx_rtd_theme

Optional / heavier dependencies (needed for full functionality):

- `classy` (CLASS python wrapper) — used to compute linear power spectra.
- `jax` and related packages — optional accelerated backend.
- `interpax` — used by the JAX interpolation utilities.

Installing everything (recommended for full features):

.. code-block:: bash

   python -m pip install -r requirements.txt

Using Google Colab or cloud environments
----------------------------------------

You can run notebooks in Google Colab, but you must install required packages in the first cell. Example:

.. code-block:: python

   %pip install numpy scipy matplotlib
   %pip install -e git+https://github.com/<your-username>/FOLPSpipe#egg=FOLPSpipe
