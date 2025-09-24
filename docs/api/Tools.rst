Tools
=====

This page documents the utility functions implemented in the ``folps.tools``
module. Each function signature is shown together with a short description of
its purpose, arguments, return values, implementation notes, and a brief
example where appropriate.

All function names and signatures are kept in English exactly as they appear
in the source code.

Legendre
--------

Signature

    ``legendre(ell)``

Purpose

    Return a callable that evaluates the Legendre polynomial of order ``ell``
    over an array of points ``x``.

Parameters

- ``ell`` (int): polynomial order. The implementation currently supports
  orders 0, 2 and 4.

Returns

- callable: a function ``P(x)`` returning the Legendre polynomial evaluated
  at ``x`` for the specified order.

Errors

- ``NotImplementedError`` if ``ell`` is not one of the supported orders.

Notes

- This is an explicit, minimal implementation for the low-order polynomials
  commonly used in multipole statistics (``ell = 0, 2, 4``). For other
  orders prefer ``scipy.special.legendre`` or similar utilities.

Example

.. code-block:: python

   P2 = legendre(2)
   vals = P2(np.array([0.0, 0.5, 1.0]))

interp
------

Signature

    ``interp(k, x, y)``

Purpose

    One-dimensional cubic interpolator. Given nodes ``(x, y)`` evaluate the
    interpolated values at points ``k``. The implementation uses
    ``scipy.interpolate.interp1d`` with ``kind='cubic'`` and
    ``fill_value='extrapolate'``.

Parameters

- ``k`` (array-like): query points where the interpolation is required.
- ``x`` (array-like): monotonically increasing nodes in the x-axis.
- ``y`` (array-like): function values at the nodes ``x``.

Returns

- ``ndarray`` (float64): interpolated values at ``k`` (forced to float64 for
  numerical consistency).

Notes

- If you need JAX-compatible interpolation, use the corresponding routines in
  ``tools_jax.py``.

Example

.. code-block:: python

   kq = np.linspace(0.1, 1.0, 50)
   yq = interp(kq, x_nodes, y_nodes)

tupleset (helper)
------------------

Signature

  ``tupleset(t, i, value)``

Purpose

  Small helper that returns a new tuple based on ``t`` with the element at
  position ``i`` replaced by ``value``. It is used internally to construct
  multi-dimensional slice objects.

Parameters

- ``t`` (tuple): original tuple.
- ``i`` (int): index to replace.
- ``value``: new value to place at position ``i``.

Returns

- tuple: a new tuple with the specified replacement.

true_divide (helper)
---------------------

Signature

    ``true_divide(h0, h1, out=None, where=None)``

Purpose

    Element-wise division utility that supports a boolean ``where`` mask and
    optional ``out`` storage. The routine mimics safe division logic found in
    NumPy/SciPy/JAX code paths and helps avoiding division-by-zero issues while
    preserving broadcasting semantics.

Parameters

- ``h0``, ``h1``: arrays to be divided (must be broadcast-compatible).
- ``out``: optional output array to store the result.
- ``where``: optional boolean mask indicating where the division should be
  performed.

Returns

- ``ndarray``: the resulting array, shaped as ``out`` if provided, otherwise
  as the broadcast result of ``h0/h1``.

Notes

- If ``out`` is ``None`` the function allocates an array filled with zeros and
  writes the computed values into it according to ``where``.

simpson (integrator)
---------------------

Signature

    ``simpson(y, *, x=None, dx=1.0, axis=-1, even=_NoValue)``

Purpose

    Composite Simpson rule implementation for integrating sampled data. This
    code is adapted from SciPy and supports irregular spacing as well as the
    handling of even/odd numbers of samples.

Parameters

- ``y``: array-like of sampled function values (supports multi-dimensional
  arrays).
- ``x``: optional array-like of sample points. If omitted, the spacing
  specified by ``dx`` is used.
- ``dx``: scalar spacing used when ``x`` is None (default ``1.0``).
- ``axis``: axis along which to integrate.
- ``even``: controls behaviour for an even number of samples. This parameter
  mirrors the SciPy API; the default behaviour matches SciPy's precise
  selection.

Returns

- float or ``ndarray``: estimated integral along the requested axis.

Implementation notes

- A helper ``_basic_simpson`` is used to apply Simpson's rule to the odd
  number-of-samples case; the wrapper handles adjustments required when the
  number of points is even.
- The implementation aims to remain numerically stable and compatible with
  SciPy's reference behaviour.

extrapolate
-----------

Signature

  ``extrapolate(x, y, xq)``

Purpose

  Perform a linear regression on the data ``(x, y)`` and evaluate the
  fitted line at points ``xq``.

Parameters

- ``x``, ``y``: base data used for the linear fit.
- ``xq``: query points where the extrapolated values are required.

Returns

- tuple ``(xq, yq)``: the input query coordinates and the extrapolated
  values.

Notes

- The routine performs an ordinary least-squares linear fit and returns the
  corresponding straight line. It is suitable when the asymptotic behaviour
  is approximately linear in the chosen scale.

extrapolate_linear_loglog
-------------------------

Signature

    ``extrapolate_linear_loglog(k, pk, kcut, k_extrapolate, is_high_k=True)``

Purpose

    Extrapolate a power spectrum ``pk(k)`` toward higher or lower ``k`` using
    a log-log linear fit. The routine selects a small window of points near
    ``kcut`` and fits a straight line in log10-space to generate geometrically
    spaced extrapolated points.

Parameters

- ``k``, ``pk``: arrays of modes and corresponding power spectrum values.
- ``kcut``: cut point where extrapolation begins.
- ``k_extrapolate``: final ``k`` bound for extrapolation (max or min as
  required).
- ``is_high_k``: if ``True`` extrapolate to larger ``k``, otherwise toward
  smaller ``k``.

Returns

- ``(k_result, pk_result)``: arrays concatenating the original region with the
  extrapolated segment.

Notes

- The method operates on ``log10(k)`` and ``log10(abs(pk))`` to preserve
  apparent power-law behaviours and handles the sign of ``pk`` separately.

Convenience wrappers for extrapolation
--------------------------------------

- ``extrapolate_high_k(k, pk, kcutmax, kmax)``
- ``extrapolate_low_k(k, pk, kcutmin, kmin)``
- ``extrapolate_k(k, pk, kcutmin, kmin, kcutmax, kmax)``

These convenience wrappers call the general routine above with appropriate
arguments to extrapolate only at high- or low-``k``, or both ends.

extrapolate_pklin
-----------------

Signature

  ``extrapolate_pklin(k, pk)``

Purpose

  Check whether the supplied ``k`` range covers the typical working window
  (approximately ``kmin=1e-5`` to ``kmax=200``). If coverage is missing the
  input power spectrum is extended using ``extrapolate_k``.

Returns

- ``(k_out, pk_out)``: arrays that are guaranteed to cover roughly the range
  ``[1e-5, 200]``.

get_pknow
---------

Signature

    ``get_pknow(k, pk, h)``

Purpose

    Compute the 'no-wiggle' (smooth) component of the linear power spectrum
    by following the approach of Hamann et al. (2010). The method uses a
    discrete sine transform (DST) to isolate and remove BAO-related
    oscillatory modes, then reconstructs the smooth component with the
    inverse transform.

Parameters

- ``k`` (array): wavenumber vector where ``pk`` is defined.
- ``pk`` (array): linear power spectrum (contains BAO wiggles).
- ``h`` (float): dimensionless Hubble parameter (H0/100) used to define the
  internal sampling grid.

Returns

- ``(k, pknow)``: tuple containing ``k`` and the reconstructed smooth power
  spectrum evaluated on that grid.

Notes

- The routine densely samples ln(k P(k)), applies a type-I DST, removes the
  harmonic components in a BAO window (``mcutmin`` to ``mcutmax``) and
  reassembles the smooth spectrum.
- This implementation requires SciPy (FFT / DST). It is not JAX-compatible as
  implemented.

get_linear_ir
-------------

Signature

    ``get_linear_ir(k, pk, h, pknow=None, fullrange=False, kmin=0.01, kmax=0.5, rbao=104, saveout=False)``

Purpose

    Compute an infrared-resummed (IR-resummed) version of the linear power
    spectrum. The routine applies a damping factor computed from the smooth
    (no-wiggle) component to model the broadening of BAO oscillations by
    large-scale displacements.

Parameters

- ``k, pk``: input wavenumbers and linear power spectrum.
- ``h``: dimensionless Hubble parameter.
- ``pknow``: optional precomputed no-wiggle power spectrum. If ``None``, the
  function will call ``get_pknow`` internally.
- ``fullrange``: if ``False`` the function returns a reduced sampling; if
  ``True`` it returns the full internal grid used in the computation.
- ``kmin, kmax``: bounding wavenumbers used for some internal integrals.
- ``rbao``: BAO scale (default 104 Mpc/h).
- ``saveout``: boolean flag reserved for debugging or output saving.

Returns

- If ``fullrange`` is ``False`` the function returns ``(newkT, newpk)`` sampled
  on a subgrid; if ``True`` it returns ``(kT, pk_IRs)`` covering the complete
  internal range.

Notes

- When ``pknow`` is not provided the routine computes it via
  ``get_pknow``. The variance integral used to compute the damping factor is
  evaluated with a Simpson quadrature over a geometrical momentum grid.

get_linear_ir_ini
-----------------

Signature

  ``get_linear_ir_ini(k, pkl, pklnw, h=0.6711, k_BAO=1.0 / 104.)``

Purpose

  A compact variant to compute an IR-resummed linear spectrum using a fixed
  BAO scale (``k_BAO``). Returns ``(k, pkl_IR)``.

Hubble, DA, qpar_qperp
----------------------

- ``Hubble(Om, z_ev)``: returns ``H(z)/H0`` for a flat matter + cosmological
  constant model: ``sqrt(Om*(1+z)**3 + (1-Om))``.
- ``DA(Om, z_ev)``: comoving angular diameter distance computed via a
  numerical integral over ``1/H(z)``.
- ``qpar_qperp(Omega_fid, Omega_m, z_pk, cosmo=None)``: compute Alcock-Paczynski
  distortion factors ``(q_parallel, q_perp)`` by comparing a fiducial
  cosmology with the cosmology of interest. If a ``cosmo`` object (CLASS)
  is provided its distance and H(z) methods are used; otherwise the simple
  analytic expressions above are employed.

pknwJ (alternative routine)
----------------------------

Signature

  ``pknwJ(k, PSLk, h)``

Purpose

  An alternative / historical implementation for extracting the no-wiggle
  component of the power spectrum, conceptually similar to
  ``get_pknow``. It is preserved in the module for compatibility and
  debugging.


Final remarks
-------------

- The functions in ``folps.tools`` combine general numerical utilities
  (integration, interpolation) with cosmology-specific tools (power spectrum
  extrapolation, wiggle/no-wiggle separation, IR resummation) and small
  internal helpers.

- To keep API documentation fully synchronized with the codebase it is
  recommended to add or enrich docstrings in ``folps/tools.py`` and then
  enable ``autodoc`` for automated API pages. I can help with either adding
  docstrings or switching the docs to ``autodoc`` if you prefer.

References
----------

- Hamann, J., et al. (2010). Method for computing a smooth (no-wiggle)
  component of the matter power spectrum.
- SciPy documentation and source related to ``simpson`` and discrete sine
  transforms (DST/IDST).
