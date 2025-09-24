Quickstart
==========

This quickstart gives the minimal steps to run an example using the bundled NumPy backend and the fallback linear power spectrum shipped in `folps/inputpkT.txt`.

1) Set the backend before importing the package

.. code-block:: python

   import os
   os.environ['FOLPS_BACKEND'] = 'numpy'

2) Run the example script

.. code-block:: bash

   python examples/run_simple.py

The script will save `examples/run_simple_output.npz` with the arrays `k`, `P0`, `P2`, and `P4`.
