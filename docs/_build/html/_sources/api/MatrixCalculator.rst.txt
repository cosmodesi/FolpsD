MatrixCalculator
================

Description

`MatrixCalculator` computes the M matrices that are independent of cosmology and
can be reused in different runs. The matrices are stored on disk under the filename
constructed from the FFTLog settings and flags.

Usage

.. code-block:: python


   from folps import MatrixCalculator


   matrix = MatrixCalculator(nfftlog=128, A_full=True)

   mmatrices = matrix.get_mmatrices()

Notes

- The computation may take some time but is done once; results are cached in a .npy file.
- The default `b_nu` and k-range are tuned to the original implementation; changing them is not yet fully tested.
