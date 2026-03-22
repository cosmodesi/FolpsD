API Reference
API Reference
=============

This page exposes the main modules and a short summary for each. The detailed API is generated from docstrings using Sphinx `autodoc`.

folps.folps
-----------

Core module. Responsibilities:

- Backend manager: select between NumPy and JAX implementations via the `FOLPS_BACKEND` environment variable or by creating a `BackendManager`.
- High-level classes and calculators: `MatrixCalculator`, `NonLinearPowerSpectrumCalculator`, `RSDMultipolesPowerSpectrumCalculator`, `BispectrumCalculator`, and helpers such as `get_cm`, `f_over_f0_EH`, and `run_class` wrapper.
- Small backend-aware helpers (e.g., `spherical_jn_backend`) so algorithms work identically with NumPy or JAX arrays.

The detailed per-class pages are available below. These are written as short
manual summaries so the docs build does not require importing heavy dependencies.

.. toctree::
   :maxdepth: 1

   api/MatrixCalculator
   api/NonLinearPowerSpectrumCalculator
   api/RSDMultipolesPowerSpectrumCalculator
   api/BispectrumCalculator
   api/Backend_and_helpers
   api/Tools

folps.cosmo_class
-----------------

Utilities to obtain or load the linear power spectrum. The main entry point is `run_class`, a thin wrapper around CLASS that returns a dictionary with keys `k`, `pk`, `fz`, `Dz`, and other useful quantities. If CLASS isn't available, the repository includes precomputed `inputpkT.txt` used as a fallback in examples.

Short description: helper to get or load linear power spectra (use `run_class` or fallback files).

folps.tools
-----------

NumPy-based utility functions used by the core algorithms. Highlights:

- `interp`: cubic interpolator wrapper (scipy.interpolate), returns a NumPy array.
- `simpson`: robust composite Simpson integrator adapted from SciPy.
- `legendre`: returns Legendre polynomial functions for ell=0,2,4.
- `extrapolate`, `extrapolate_linear_loglog`: helpers to extend pk to required k ranges.

Short description: NumPy utility functions for interpolation, integration and extrapolation.

See the `api/` pages for short manual descriptions of the main helpers.


folps.tools_jax
---------------

JAX-compatible replacements for `tools.py`. Function names and behavior mirror `tools.py` as closely as possible so the high-level code can run unchanged when the JAX backend is selected. Note: some functions call external JAX-friendly libraries (e.g., `interpax`) and require JAX to be installed.

Short description: JAX-compatible replacements for the helpers in `tools.py`.
Requires JAX and additional JAX-friendly libraries (e.g., interpax) to be installed.