RSDMultipolesPowerSpectrumCalculator
====================================

Description
-----------

Calculator to obtain redshift-space distortion (RSD) multipoles from the loop tables returned by
`NonLinearPowerSpectrumCalculator`.

Usage
-----

.. code-block:: python

   from folps import RSDMultipolesPowerSpectrumCalculator

   multipoles = RSDMultipolesPowerSpectrumCalculator(model='FOLPSD')
   P0, P2, P4 = multipoles.get_rsd_pkell(kobs=kout, qpar=qpar, qper=qper, pars=pars, table=table, table_now=table_now)

Notes
-----

- Multiple models are supported (e.g., 'FOLPSD', 'TNS', 'EFT'); consult the code for exact options.
