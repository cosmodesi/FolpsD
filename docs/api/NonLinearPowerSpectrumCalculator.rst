NonLinearPowerSpectrumCalculator
================================

Description
-----------

Class that computes 1-loop corrections to the linear power spectrum. It accepts the precomputed matrices
(`mmatrices`) created by `MatrixCalculator` and returns loop tables used by the RSD multipoles calculator.

Usage
-----

.. code-block:: python

   from folps import NonLinearPowerSpectrumCalculator

   nonlinear = NonLinearPowerSpectrumCalculator(mmatrices=mmatrices, kernels='fk', z=0.3)
   table, table_now = nonlinear.calculate_loop_table(k=k_arr, pklin=pk_arr, cosmo=None, z=0.3)

Notes
-----

- The class supports kernels 'fk' (scale-dependent growth) and 'eds' (Einstein-de Sitter approximation).
- Provide either a `cosmo` object or the required growth parameters in kwargs (see docstrings in code).
