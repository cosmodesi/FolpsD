Tools
=====

This page provides concise, reference-style documentation for the public
utility functions defined in the ``folps.tools`` module. Each entry shows the
Python signature, a short description, parameter and return types, implementation
notes and a short usage example where helpful. The layout follows the Sphinx
Python domain conventions so it renders similarly to generated API pages.

Note: these pages are written as static reference documentation (no import is
required to build the docs). If you later enable ``autodoc`` you can replace
these manual entries with automatically generated ones from in-source
docstrings.

.. py:function:: legendre(ell)

  Return a callable that evaluates the Legendre polynomial of order
  ``ell``.

  :param int ell: polynomial order (supported: ``0``, ``2``, ``4``).
  :returns: callable P(x) evaluating the polynomial at ``x``.
  :rtype: callable
  :raises NotImplementedError: if ``ell`` is not supported.

  Notes:
  The implementation provides explicit expressions for the low-order
  polynomials commonly used in multipole statistics. For higher orders use
  ``scipy.special.legendre``.

  Example:

  .. code-block:: python

    P2 = legendre(2)
    vals = P2(np.array([0.0, 0.5, 1.0]))


.. py:function:: interp(k, x, y)

  One-dimensional cubic interpolation wrapper.

    :param k: query points where an interpolated value is required.
    :type k: array-like
    :param x: monotonic x nodes.
    :type x: array-like
    :param y: function values at the nodes ``x``.
    :type y: array-like
    :returns: interpolated values at ``k``.
    :rtype: numpy.ndarray

    Notes:
    The function uses ``scipy.interpolate.interp1d(kind='cubic',
    fill_value='extrapolate')`` and forces the output to ``float64`` for
    numerical consistency. For JAX-compatible behaviour use the matching
    functions in ``tools_jax.py``.

    Example:

    .. code-block:: python

      kq = np.linspace(0.1, 1.0, 50)
      yq = interp(kq, x_nodes, y_nodes)


.. py:function:: tupleset(t, i, value)

  Small helper returning a new tuple with element ``i`` replaced by
  ``value``.

  :param tuple t: original tuple
  :param int i: index to replace
  :param value: value to place at index ``i``
  :returns: new tuple with the replacement
  :rtype: tuple


.. py:function:: true_divide(h0, h1, out=None, where=None)

  Safe elementwise division with optional ``where`` mask and ``out``
  parameter.

  :param h0: numerator array
  :type h0: array-like
  :param h1: denominator array
  :type h1: array-like
  :param out: optional output array
  :param where: optional boolean mask
  :returns: elementwise division result
  :rtype: numpy.ndarray


.. py:function:: simpson(y, *, x=None, dx=1.0, axis=-1, even=_NoValue)

  Composite Simpson rule integrator adapted from SciPy.

  :param y: sampled function values
  :type y: array-like
  :param x: sample points (optional)
  :type x: array-like or None
  :param float dx: spacing used when ``x`` is ``None``
  :param int axis: axis along which to integrate
  :param even: controls behaviour for even number of samples
  :returns: estimated integral (float or array)
  :rtype: float or numpy.ndarray

  Notes
  -----
  The implementation includes an internal ``_basic_simpson`` helper and
  handles both regular and irregular spacing, as well as even/odd sample
  counts in a way compatible with SciPy's behaviour.


.. py:function:: extrapolate(x, y, xq)

  Fit a straight line to ``(x, y)`` by ordinary least squares and evaluate it
  at ``xq``.

  :param x: independent variable
  :type x: array-like
  :param y: dependent variable
  :type y: array-like
  :param xq: query points
  :type xq: array-like
  :returns: tuple ``(xq, yq)`` with extrapolated values
  :rtype: tuple


.. py:function:: extrapolate_linear_loglog(k, pk, kcut, k_extrapolate, is_high_k=True)

  Log–log linear extrapolation for power spectra.

  :param k: input wavenumbers
  :type k: array-like
  :param pk: input power spectrum
  :type pk: array-like
  :param float kcut: cutoff wavenumber where extrapolation begins
  :param float k_extrapolate: final bound for extrapolation
  :param bool is_high_k: if True extrapolate to higher ``k`` else to lower
  :returns: tuple ``(k_result, pk_result)`` including the extrapolated region
  :rtype: tuple

  Notes
  -----
  The routine works in ``log10(k)`` and ``log10(abs(pk))`` to capture
  power-law behaviour. The sign of ``pk`` is preserved and handled
  separately.


.. rubric:: Convenience wrappers

.. py:function:: extrapolate_high_k(k, pk, kcutmax, kmax)

.. py:function:: extrapolate_low_k(k, pk, kcutmin, kmin)

.. py:function:: extrapolate_k(k, pk, kcutmin, kmin, kcutmax, kmax)

  Convenience wrappers that call the general extrapolation routine for the
  appropriate k-end(s).


.. py:function:: extrapolate_pklin(k, pk)

  Ensure the power spectrum covers the working range (approx. 1e-5 to 200).

  :param k: input wavenumbers
  :type k: array-like
  :param pk: input power spectrum
  :type pk: array-like
  :returns: extended (k_out, pk_out)
  :rtype: tuple


.. py:function:: get_pknow(k, pk, h)

  Compute a smooth (no-wiggle) linear power spectrum using a DST-based
  filtering method (following Hamann et al. and related approaches).

  :param k: input wavenumbers
  :type k: array-like
  :param pk: linear power spectrum with BAO wiggles
  :type pk: array-like
  :param float h: H0/100 used to build the internal sampling grid
  :returns: tuple ``(k, pknow)`` with the reconstructed smooth spectrum
  :rtype: tuple

  Notes
  -----
  The algorithm densely samples ln(k*P(k)), applies a type-I discrete sine
  transform (DST), removes harmonic components in a BAO window, and
  reconstructs the smooth component via the inverse transform. SciPy
  (FFT/DST) is required; this routine is not JAX-compatible in its current
  form.


.. py:function:: get_linear_ir(k, pk, h, pknow=None, fullrange=False, kmin=0.01, kmax=0.5, rbao=104, saveout=False)

  Compute an IR-resummed linear power spectrum by damping BAO oscillations
  according to large-scale displacement variance derived from the no-wiggle
  component.

  :param k, pk: input wavenumbers and linear power spectrum
  :param float h: H0/100
  :param pknow: optional no-wiggle spectrum; if ``None`` it is computed
    internally via ``get_pknow``
  :param bool fullrange: when ``False`` returns a sub-sampled result; when
    ``True`` returns the full internal grid
  :returns: see notes
  :rtype: tuple

  Notes
  -----
  The damping integral used for the displacement variance is evaluated with
  a Simpson-like quadrature on a geometrical momentum grid. When ``pknow`` is
  not provided the function computes it via ``get_pknow``.


.. py:function:: get_linear_ir_ini(k, pkl, pklnw, h=0.6711, k_BAO=1.0 / 104.)

  Compact variant to compute an IR-resummed linear spectrum using a fixed
  BAO scale ``k_BAO``. Returns ``(k, pkl_IR)``.


.. py:function:: Hubble(Om, z_ev)

  Return H(z)/H0 for a flat matter + Lambda cosmology: ``sqrt(Om*(1+z)**3 + (1-Om))``.


.. py:function:: DA(Om, z_ev)

  Comoving angular diameter distance computed via numerical integration of
  ``1/H(z)``.


.. py:function:: qpar_qperp(Omega_fid, Omega_m, z_pk, cosmo=None)

  Compute Alcock-Paczynski distortion factors ``(q_parallel, q_perp)`` by
  comparing a fiducial cosmology with the cosmology of interest. If a
  ``cosmo`` object (CLASS) is provided its distance/H(z) methods are used;
  otherwise simple analytic expressions are used.


.. py:function:: pknwJ(k, PSLk, h)

  Historical/alternative implementation of the no-wiggle extraction. Kept
  primarily for debugging and backward compatibility; conceptually similar
  to ``get_pknow``.


Final remarks
-------------

The functions in ``folps.tools`` cover generic numerical utilities (integration,
interpolation) and cosmology-specific helpers (power spectrum extrapolation,
wiggle/no-wiggle separation, IR resummation). If you prefer an automatically
generated API reference, I can add docstrings to the source and enable
``autodoc`` so pages like this are produced directly from the code.

References
----------

- Hamann, J., et al. (2010). Method for computing a smooth (no-wiggle)
  component of the matter power spectrum.
- SciPy documentation and source related to ``simpson`` and discrete sine
  transforms (DST/IDST).

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
