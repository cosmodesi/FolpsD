
Examples and quick start
========================

This page contains runnable examples adapted from `run_folps_numpy.ipynb`. The examples use the NumPy backend and a fallback linear power spectrum shipped with the repository (`folps/inputpkT.txt`).

Simple end-to-end example (NumPy backend)
----------------------------------------

This script demonstrates a minimal workflow: select backend, load or compute linear pk, compute matrices (backend-independent), compute loop contributions, and obtain RSD multipoles.

.. code-block:: python

   import os
   import numpy as np

   # Choose backend before importing the package
   os.environ['FOLPS_BACKEND'] = 'numpy'  # or 'jax'

   from folps import *

   # Load linear power spectrum fallback included in the repo
   k_arr, pk_arr = np.loadtxt('folps/inputpkT.txt', unpack=True)

   # 1) Precompute matrices (only once)
   matrix = MatrixCalculator()
   mmatrices = matrix.get_mmatrices()

   # 2) Non-linear loop table
   nonlinear = NonLinearPowerSpectrumCalculator(mmatrices=mmatrices, kernels='fk', z=0.3)
   table, table_now = nonlinear.calculate_loop_table(k=k_arr, pklin=pk_arr, cosmo=None, z=0.3)

   # 3) Set nuisance parameters and geometry
   b1 = 1.645
   b2 = -0.46
   bs2 = -4./7*(b1 - 1)
   b3nl = 32./315*(b1 - 1)
   alpha0 = 3.0
   alpha2 = -28.9
   alpha4 = 0.0
   ctilde = 0.0
   PshotP = 1. / 0.0002118763
   alphashot0 = 0.08
   alphashot2 = -8.1
   X_Fog_pk = 1
   pars = [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, PshotP, X_Fog_pk]

   qpar, qper = qpar_qperp(Omega_fid=0.31, Omega_m=0.3211636237981114, z_pk=0.3)

   # 4) Compute multipoles
   kout = np.logspace(np.log10(0.01), np.log10(0.3), num=100)
   multipoles = RSDMultipolesPowerSpectrumCalculator(model='FOLPSD')
   P0, P2, P4 = multipoles.get_rsd_pkell(kobs=kout, qpar=qpar, qper=qper, pars=pars, table=table, table_now=table_now, bias_scheme='folps', damping='lor')

   print('k:', kout[:5])
   print('P0:', P0[:5])

Notes and tips
--------------

- To use the JAX backend set `os.environ['FOLPS_BACKEND']='jax'` before importing `folps`. JAX and `interpax` must be installed.
- The `MatrixCalculator` output is backend-independent: once computed it can be reused for different cosmologies.
- For full end-to-end runs with CLASS, install `classy` and use `cosmo_class.run_class` (see `run_folps_numpy.ipynb` for an example). If CLASS is not available the notebook falls back to `inputpkT.txt`.

Running the example script
--------------------------

You can run the example script saved at `examples/run_simple.py`. From the repository root:

.. code-block:: bash

   python examples/run_simple.py

The script saves results to `examples/run_simple_output.npz` containing arrays `k`, `P0`, `P2`, `P4`.

Building the documentation locally
---------------------------------

Install dependencies and build HTML:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   cd docs
   sphinx-build -b html . _build/html

