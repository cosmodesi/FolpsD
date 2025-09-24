BackendManager and helpers
==========================

Description
-----------

BackendManager selects between NumPy and JAX backends and exposes small backend-aware helpers.

Key functions
-------------

- `BackendManager`: select backend and get backend-specific modules (np, interp, simpson, etc.).
- `spherical_jn_backend`: backend-aware spherical Bessel helper.
- `get_cm`, `f_over_f0_EH`: cosmology helpers used by other classes.
