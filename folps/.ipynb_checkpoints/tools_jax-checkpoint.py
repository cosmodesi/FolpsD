import warnings
from jax import numpy as jnp
import numpy as np
from scipy import special

import interpax
import hashlib

from jax import config; config.update('jax_enable_x64', True)


def roots_legendre(n):
    """
    Compute Gauss-Legendre quadrature roots and weights using JAX.
    
    This is a JAX-native implementation that computes the roots of the Legendre
    polynomial of degree n and their corresponding weights for Gauss-Legendre
    quadrature on the interval [-1, 1].
    
    Parameters
    ----------
    n : int
        The degree of the Legendre polynomial (number of quadrature points).
    
    Returns
    -------
    roots : ndarray, shape(n,)
        The roots of the Legendre polynomial of degree n.
    weights : ndarray, shape(n,)
        The weights for Gauss-Legendre quadrature.
    
    Notes
    -----
    This implementation uses Newton-Raphson iteration to find the roots of
    Legendre polynomials using JAX operations, making it compatible with
    JAX transformations (jit, grad, etc.).
    
    Reference
    ---------
    - Golub & Welsch (1969), "Calculation of Gauss Quadrature Rules"
    - Press et al., "Numerical Recipes"
    """
    # Initial guesses for roots using Chebyshev approximation
    i = jnp.arange(1, n + 1)
    theta = jnp.pi * (4 * i - 1) / (4 * n + 2)
    x = jnp.cos(theta)
    
    # Newton-Raphson iteration to refine roots
    for _ in range(10):  # Usually converges in 3-5 iterations
        # Compute Legendre polynomial and its derivative using recurrence relations
        P = jnp.ones_like(x)
        P_prev = jnp.zeros_like(x)
        
        for k in range(1, n + 1):
            P_next = ((2 * k - 1) * x * P - (k - 1) * P_prev) / k
            P_prev = P
            P = P_next
        
        # Derivative: P'_n(x) = n * (x * P_n(x) - P_{n-1}(x)) / (x^2 - 1)
        P_deriv = n * (x * P - P_prev) / (x**2 - 1)
        
        # Newton-Raphson update
        x = x - P / P_deriv
    
    # Compute weights: w_i = 2 / ((1 - x_i^2) * [P'_n(x_i)]^2)
    P_deriv_final = n * (x * P - P_prev) / (x**2 - 1)
    weights = 2.0 / ((1.0 - x**2) * P_deriv_final**2)
    
    return x, weights


def legendre(ell):
    """
    Return Legendre polynomial of given order.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Legendre_polynomials
    """
    if ell == 0:

        return lambda x: jnp.ones_like(x)

    if ell == 2:

        return lambda x: 1. / 2. * (3 * x**2 - 1)

    if ell == 4:

        return lambda x: 1. / 8. * (35 * x**4 - 30 * x**2 + 3)

    raise NotImplementedError('Legendre polynomial for ell = {:d} not implemented'.format(ell))


def interp(xq, x, f, method='cubic'):
    """
    Interpolate a 1d function.

    Note
    ----
    Using interpax: https://github.com/f0uriest/interpax

    Parameters
    ----------
    xq : ndarray, shape(Nq,)
        query points where interpolation is desired
    x : ndarray, shape(Nx,)
        coordinates of known function values ("knots")
    f : ndarray, shape(Nx,...)
        function values to interpolate
    method : str
        method of interpolation

        - ``'nearest'``: nearest neighbor interpolation
        - ``'linear'``: linear interpolation
        - ``'cubic'``: C1 cubic splines (aka local splines)
        - ``'cubic2'``: C2 cubic splines (aka natural splines)
        - ``'catmull-rom'``: C1 cubic centripetal "tension" splines
        - ``'cardinal'``: C1 cubic general tension splines. If used, can also pass
          keyword parameter ``c`` in float[0,1] to specify tension
        - ``'monotonic'``: C1 cubic splines that attempt to preserve monotonicity in the
          data, and will not introduce new extrema in the interpolated points
        - ``'monotonic-0'``: same as ``'monotonic'`` but with 0 first derivatives at
          both endpoints

    derivative : int >= 0
        derivative order to calculate
    extrap : bool, float, array-like
        whether to extrapolate values beyond knots (True) or return nan (False),
        or a specified value to return for query points outside the bounds. Can
        also be passed as a 2 element array or tuple to specify different conditions
        for xq<x[0] and x[-1]<xq
    period : float > 0, None
        periodicity of the function. If given, function is assumed to be periodic
        on the interval [0,period]. None denotes no periodicity

    Returns
    -------
    fq : ndarray, shape(Nq,...)
        function value at query points
    """
    method = {1: 'linear', 3: 'cubic'}.get(method, method)
    xq = jnp.asarray(xq)
    shape = xq.shape
    # Use extrap=True to match scipy.interpolate(..., fill_value='extrapolate') used in the numpy backend
    return interpax.interp1d(xq.reshape(-1), x, f, method=method, extrap=True).reshape(shape + f.shape[1:])


def interp_at_kmin(x, f):
    """
    Interpolate at k→0 using linear extrapolation from the first two points.
    This provides more stable and accurate results for very small k values
    compared to cubic extrapolation, matching the behavior of scipy.
    
    Parameters
    ----------
    x : ndarray, shape(Nx,)
        coordinates of known function values ("knots")
    f : ndarray, shape(Nx,...)
        function values to interpolate
    
    Returns
    -------
    f0 : float or ndarray
        extrapolated function value at k→0
    """
    # Use linear extrapolation from first two points: f(0) ≈ f[0] - x[0] * (f[1]-f[0])/(x[1]-x[0])
    # This is more stable than cubic extrapolation for k→0
    slope = (f[1] - f[0]) / (x[1] - x[0])
    return f[0] - x[0] * slope


_NoValue = None


def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)


def true_divide(h0, h1, out=None, where=None):
    if out is None:
        out = jnp.zeros_like(h1)
    if where is None:
        out = out.at[...].set(h0 / h1)
        return out
    return jnp.where(jnp.asarray(where), h0 / h1, out)


def _basic_simpson(y, start, stop, x, dx, axis):
    nd = len(y.shape)
    if start is None:
        start = 0
    step = 2
    slice_all = (slice(None),)*nd
    slice0 = tupleset(slice_all, axis, slice(start, stop, step))
    slice1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
    slice2 = tupleset(slice_all, axis, slice(start+2, stop+2, step))

    if x is None:  # Even-spaced Simpson's rule.
        result = jnp.sum(y[slice0] + 4.0*y[slice1] + y[slice2], axis=axis)
        result *= dx / 3.0
    else:
        # Account for possibly different spacings.
        #    Simpson's rule changes a bit.
        h = jnp.diff(x, axis=axis)
        sl0 = tupleset(slice_all, axis, slice(start, stop, step))
        sl1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
        h0 = h[sl0].astype(float)
        h1 = h[sl1].astype(float)
        hsum = h0 + h1
        hprod = h0 * h1
        h0divh1 = true_divide(h0, h1, out=jnp.zeros_like(h0), where=h1 != 0)
        tmp = hsum/6.0 * (y[slice0] *
                          (2.0 - true_divide(1.0, h0divh1,
                                                out=jnp.zeros_like(h0divh1),
                                                where=h0divh1 != 0)) +
                          y[slice1] * (hsum *
                                       true_divide(hsum, hprod,
                                                      out=jnp.zeros_like(hsum),
                                                      where=hprod != 0)) +
                          y[slice2] * (2.0 - h0divh1))
        result = jnp.sum(tmp, axis=axis)
    return result


def simpson(y, *, x=None, dx=1.0, axis=-1, even=_NoValue):
    """
    Integrate y(x) using samples along the given axis and the composite
    Simpson's rule. If x is None, spacing of dx is assumed.

    If there are an even number of samples, N, then there are an odd
    number of intervals (N-1), but Simpson's rule requires an even number
    of intervals. The parameter 'even' controls how this is handled.

    Parameters
    ----------
    y : array_like
        Array to be integrated.
    x : array_like, optional
        If given, the points at which `y` is sampled.
    dx : float, optional
        Spacing of integration points along axis of `x`. Only used when
        `x` is None. Default is 1.
    axis : int, optional
        Axis along which to integrate. Default is the last axis.
    even : {None, 'simpson', 'avg', 'first', 'last'}, optional
        'avg' : Average two results:
            1) use the first N-2 intervals with
               a trapezoidal rule on the last interval and
            2) use the last
               N-2 intervals with a trapezoidal rule on the first interval.

        'first' : Use Simpson's rule for the first N-2 intervals with
                a trapezoidal rule on the last interval.

        'last' : Use Simpson's rule for the last N-2 intervals with a
               trapezoidal rule on the first interval.

        None : equivalent to 'simpson' (default)

        'simpson' : Use Simpson's rule for the first N-2 intervals with the
                  addition of a 3-point parabolic segment for the last
                  interval using equations outlined by Cartwright [1]_.
                  If the axis to be integrated over only has two points then
                  the integration falls back to a trapezoidal integration.

                  .. versionadded:: 1.11.0

        .. versionchanged:: 1.11.0
            The newly added 'simpson' option is now the default as it is more
            accurate in most situations.

        .. deprecated:: 1.11.0
            Parameter `even` is deprecated and will be removed in SciPy
            1.14.0. After this time the behaviour for an even number of
            points will follow that of `even='simpson'`.

    Returns
    -------
    float
        The estimated integral computed with the composite Simpson's rule.

    See Also
    --------
    quad : adaptive quadrature using QUADPACK
    romberg : adaptive Romberg quadrature
    quadrature : adaptive Gaussian quadrature
    fixed_quad : fixed-order Gaussian quadrature
    dblquad : double integrals
    tplquad : triple integrals
    romb : integrators for sampled data
    cumulative_trapezoid : cumulative integration for sampled data
    cumulative_simpson : cumulative integration using Simpson's 1/3 rule
    ode : ODE integrators
    odeint : ODE integrators

    Notes
    -----
    For an odd number of samples that are equally spaced the result is
    exact if the function is a polynomial of order 3 or less. If
    the samples are not equally spaced, then the result is exact only
    if the function is a polynomial of order 2 or less.
    Copy-pasted from https://github.com/scipy/scipy/blob/v1.12.0/scipy/integrate/_quadrature.py

    References
    ----------
    .. [1] Cartwright, Kenneth V. Simpson's Rule Cumulative Integration with
           MS Excel and Irregularly-spaced Data. Journal of Mathematical
           Sciences and Mathematics Education. 12 (2): 1-9

    Examples
    --------
    >>> from scipy import integrate
    >>> import numpy as jnp
    >>> x = jnp.arange(0, 10)
    >>> y = jnp.arange(0, 10)

    >>> integrate.simpson(y, x)
    40.5

    >>> y = jnp.power(x, 3)
    >>> integrate.simpson(y, x)
    1640.5
    >>> integrate.quad(lambda x: x**3, 0, 9)[0]
    1640.25

    >>> integrate.simpson(y, x, even='first')
    1644.5

    """
    y = jnp.asarray(y)
    nd = len(y.shape)
    N = y.shape[axis]
    last_dx = dx
    first_dx = dx
    returnshape = 0
    if x is not None:
        x = jnp.asarray(x)
        if len(x.shape) == 1:
            shapex = [1] * nd
            shapex[axis] = x.shape[0]
            saveshape = x.shape
            returnshape = 1
            x = x.reshape(tuple(shapex))
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-D or the "
                             "same as y.")
        if x.shape[axis] != N:
            raise ValueError("If given, length of x along axis must be the "
                             "same as y.")

    # even keyword parameter is deprecated
    if even is not _NoValue:
        warnings.warn(
            "The 'even' keyword is deprecated as of SciPy 1.11.0 and will be "
            "removed in SciPy 1.14.0",
            DeprecationWarning, stacklevel=2
        )

    if N % 2 == 0:
        val = 0.0
        result = 0.0
        slice_all = (slice(None),) * nd

        # default is 'simpson'
        even = even if even not in (_NoValue, None) else "simpson"

        if even not in ['avg', 'last', 'first', 'simpson']:
            raise ValueError(
                "Parameter 'even' must be 'simpson', "
                "'avg', 'last', or 'first'."
            )

        if N == 2:
            # need at least 3 points in integration axis to form parabolic
            # segment. If there are two points then any of 'avg', 'first',
            # 'last' should give the same result.
            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            if x is not None:
                last_dx = x[slice1] - x[slice2]
            val += 0.5 * last_dx * (y[slice1] + y[slice2])

            # calculation is finished. Set `even` to None to skip other
            # scenarios
            even = None

        if even == 'simpson':
            # use Simpson's rule on first intervals
            result = _basic_simpson(y, 0, N-3, x, dx, axis)

            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            slice3 = tupleset(slice_all, axis, -3)

            h = jnp.asarray([dx, dx], dtype=jnp.float64)
            if x is not None:
                # grab the last two spacings from the appropriate axis
                hm2 = tupleset(slice_all, axis, slice(-2, -1, 1))
                hm1 = tupleset(slice_all, axis, slice(-1, None, 1))

                diffs = jnp.float64(jnp.diff(x, axis=axis))
                h = [jnp.squeeze(diffs[hm2], axis=axis),
                     jnp.squeeze(diffs[hm1], axis=axis)]

            # This is the correction for the last interval according to
            # Cartwright.
            # However, I used the equations given at
            # https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_rule_for_irregularly_spaced_data
            # A footnote on Wikipedia says:
            # Cartwright 2017, Equation 8. The equation in Cartwright is
            # calculating the first interval whereas the equations in the
            # Wikipedia article are adjusting for the last integral. If the
            # proper algebraic substitutions are made, the equation results in
            # the values shown.
            num = 2 * h[1] ** 2 + 3 * h[0] * h[1]
            den = 6 * (h[1] + h[0])
            alpha = true_divide(
                num,
                den,
                out=jnp.zeros_like(den),
                where=den != 0
            )

            num = h[1] ** 2 + 3.0 * h[0] * h[1]
            den = 6 * h[0]
            beta = true_divide(
                num,
                den,
                out=jnp.zeros_like(den),
                where=den != 0
            )

            num = 1 * h[1] ** 3
            den = 6 * h[0] * (h[0] + h[1])
            eta = true_divide(
                num,
                den,
                out=jnp.zeros_like(den),
                where=den != 0
            )

            result += alpha*y[slice1] + beta*y[slice2] - eta*y[slice3]

        # The following code (down to result=result+val) can be removed
        # once the 'even' keyword is removed.

        # Compute using Simpson's rule on first intervals
        if even in ['avg', 'first']:
            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            if x is not None:
                last_dx = x[slice1] - x[slice2]
            val += 0.5*last_dx*(y[slice1]+y[slice2])
            result = _basic_simpson(y, 0, N-3, x, dx, axis)
        # Compute using Simpson's rule on last set of intervals
        if even in ['avg', 'last']:
            slice1 = tupleset(slice_all, axis, 0)
            slice2 = tupleset(slice_all, axis, 1)
            if x is not None:
                first_dx = x[tuple(slice2)] - x[tuple(slice1)]
            val += 0.5*first_dx*(y[slice2]+y[slice1])
            result += _basic_simpson(y, 1, N-2, x, dx, axis)
        if even == 'avg':
            val /= 2.0
            result /= 2.0
        result = result + val
    else:
        result = _basic_simpson(y, 0, N-2, x, dx, axis)
    if returnshape:
        x = x.reshape(saveshape)
    return result


##### more tools ######


def extrapolate(x, y, xq):
    """
    Extrapolation.

    Args:
        x, y: data set with x- and y-coordinates.
        xq: x-coordinates of extrapolation.
    Returns:
        extrapolates the data set ‘x’, ‘y’  to the range given by ‘xq’.
    """
    def linear_regression(x, y):
        """
        Linear regression.

        Args:
            x, y: data set with x- and y-coordinates.
        Returns:
            slope ‘m’ and the intercept ‘b’.
        """
        xm = jnp.mean(x)
        ym = jnp.mean(y)
        npts = len(x)

        SS_xy = jnp.sum(x * y) - npts * xm * ym
        SS_xx = jnp.sum(x**2) - npts * xm**2
        m = SS_xy / SS_xx

        b = ym - m * xm
        return (m, b)

    m, b = linear_regression(x, y)
    return (xq, m * xq + b)


def extrapolate_pklin(k, pk, extrap=(10**(-7), 200), lim=None):
    """
    Extrapolation to the input linear power spectrum.

    JAX-jit friendly: avoids boolean masking (which creates data-dependent
    shapes) and data-dependent arange lengths. Uses fixed-size linspace grids
    for low/high-k extrapolation so array shapes remain static under JIT.

    Args:
        k, pk : k-coordinates and linear power spectrum.
        extrap : tuple(float, float)
            Min/Max k used for extrapolation.
        lim : unused (kept for API compatibility).
    Returns:
        (k_new, pk_new) with low/high extrapolated tails added.
    """
    # Use JAX arrays/ops so this function can be traced inside jax.jit
    k = jnp.asarray(k)
    pk = jnp.asarray(pk)

    # Do NOT slice with a boolean mask inside jit; keep static shapes
    kcut, pkcut = k, pk

    # Fixed number of points for low/high extrapolation to keep shapes static
    N_LOW = 16
    N_HIGH = 64

    # Compute approximate log-spacing from the endpoints
    logk0 = jnp.log10(kcut[0])
    logk1 = jnp.log10(kcut[1])
    dlogk_low = logk1 - logk0

    logkN = jnp.log10(kcut[-1])
    logkNm1 = jnp.log10(kcut[-2])
    dlogk_high = logkN - logkNm1

    # Low-k grid: from extrap[0] up to just below the first k, with fixed size
    logk_low_grid = jnp.linspace(jnp.log10(extrap[0]), logk0 - dlogk_low, N_LOW)
    sign_low = jnp.sign(pkcut[0])
    fitx_low = jnp.log10(jnp.abs(kcut[:5]))
    fity_low = jnp.log10(jnp.abs(pkcut[:5]))
    _, logpk_low = extrapolate(fitx_low, fity_low, logk_low_grid)

    # High-k grid: from just above the last k up to extrap[1], with fixed size
    logk_high_grid = jnp.linspace(logkN + dlogk_high, jnp.log10(extrap[1]), N_HIGH)
    sign_high = jnp.sign(pkcut[-1])
    fitx_high = jnp.log10(jnp.abs(kcut[-6:]))
    fity_high = jnp.log10(jnp.abs(pkcut[-6:]))
    _, logpk_high = extrapolate(fitx_high, fity_high, logk_high_grid)

    knew = jnp.concatenate([jnp.power(10.0, logk_low_grid), kcut, jnp.power(10.0, logk_high_grid)], axis=0)
    pknew = jnp.concatenate([sign_low * jnp.power(10.0, logpk_low), pkcut, sign_high * jnp.power(10.0, logpk_high)], axis=0)

    return knew, pknew

    
import jax
@jax.jit
def get_pknow(k, pk, h):
    """
    Routine (based on J. Hamann et. al. 2010, arXiv:1003.3999) to get the
    non-wiggle piece of the linear power spectrum.

    JIT-compatible version: uses JAX ops throughout and calls SciPy's
    DST/IDST via a host callback with static shapes to preserve numerical
    results while remaining jittable.

    Args:
        k: wave-number (1D array).
        pk: linear power spectrum (1D array, same shape as k).
        h: H0/100 (float).
    Returns:
        (k, PNWkTot): tuple of arrays with the non-wiggle spectrum evaluated
        at the input k.
    """
    import jax

    # Constants (static for JIT)
    kmin = 7.0e-5 / h
    kmax = 7.0 / h
    nk = 2**16  # must remain constant for static shapes
    mcutmin = 120
    mcutmax = 240

    # Sample ln(k P_L(k)) on an equidistant k-grid
    ksT = kmin + jnp.arange(nk, dtype=jnp.float64) * (kmax - kmin) / (nk - 1)
    # Interpolate using SciPy CubicSpline via host callback to match numpy version
    def _cubic_spline_eval(xq_np, x_np, y_np):
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(np.asarray(x_np), np.asarray(y_np), extrapolate=True)
        return cs(np.asarray(xq_np)).astype(np.float64)

    import jax
    PSL = jax.pure_callback(
        _cubic_spline_eval,
        jax.ShapeDtypeStruct(ksT.shape, jnp.float64),
        ksT, k, pk
    )
    logkpk = jnp.log(ksT * PSL)

    # Discrete sine transform type-I, ortho norm, via host callback (SciPy)
    def _dst1_ortho(x_np):
        from scipy.fft import dst
        return dst(np.asarray(x_np), type=1, norm="ortho").astype(np.float64)

    FST_shape = jax.ShapeDtypeStruct((nk,), jnp.float64)
    FSTlogkpkT = jax.pure_callback(_dst1_ortho, FST_shape, logkpk)

    # Split even/odd harmonics
    FSTlogkpkOddT = FSTlogkpkT[::2]
    FSTlogkpkEvenT = FSTlogkpkT[1::2]

    # Remove harmonics around the BAO peak (cut range)
    len_even = nk // 2  # static

    # Even branch indices and values (1-based indexing in original method)
    xEvenTcutmin = jnp.arange(1, mcutmin - 1, 1)
    xEvenTcutmax = jnp.arange(mcutmax + 2, len_even + 1, 1)
    EvenTcutmin = FSTlogkpkEvenT[0:mcutmin - 2]
    EvenTcutmax = FSTlogkpkEvenT[mcutmax + 1:len_even]
    xEvenTcuttedT = jnp.concatenate((xEvenTcutmin, xEvenTcutmax)).astype(jnp.float64)
    nFSTlogkpkEvenTcuttedT = jnp.concatenate((EvenTcutmin, EvenTcutmax))

    # Odd branch indices and values
    xOddTcutmin = jnp.arange(1, mcutmin, 1)
    xOddTcutmax = jnp.arange(mcutmax + 1, len_even + 1, 1)
    OddTcutmin = FSTlogkpkOddT[0:mcutmin - 1]
    OddTcutmax = FSTlogkpkOddT[mcutmax:len_even]
    xOddTcuttedT = jnp.concatenate((xOddTcutmin, xOddTcutmax)).astype(jnp.float64)
    nFSTlogkpkOddTcuttedT = jnp.concatenate((OddTcutmin, OddTcutmax))

    # Interpolate the FST harmonics inside the BAO range (JAX-friendly interp)
    qEven = jnp.arange(2, mcutmax + 1, 1.0)
    qOdd = jnp.arange(0, mcutmax - 1, 1.0)
    # These are small arrays; still use CubicSpline to match numpy
    PreEvenT = jax.pure_callback(
        _cubic_spline_eval,
        jax.ShapeDtypeStruct(qEven.shape, jnp.float64),
        qEven, xEvenTcuttedT, nFSTlogkpkEvenTcuttedT
    )
    PreOddT = jax.pure_callback(
        _cubic_spline_eval,
        jax.ShapeDtypeStruct(qOdd.shape, jnp.float64),
        qOdd, xOddTcuttedT, nFSTlogkpkOddTcuttedT
    )
    pre_middle = jnp.column_stack([
        PreOddT[mcutmin:mcutmax - 1],
        PreEvenT[mcutmin:mcutmax - 1]
    ]).ravel()
    preT = jnp.concatenate([
        FSTlogkpkT[:2 * mcutmin],
        pre_middle,
        FSTlogkpkT[2 * mcutmax - 2:]
    ])

    # Inverse DST-I via host callback (SciPy)
    def _idst1_ortho(x_np):
        from scipy.fft import idst
        return idst(np.asarray(x_np), type=1, norm="ortho").astype(np.float64)

    pre_shape = jax.ShapeDtypeStruct((nk,), jnp.float64)
    FSTofFSTlogkpkNWT = jax.pure_callback(_idst1_ortho, pre_shape, preT)
    PNWT = jnp.exp(FSTofFSTlogkpkNWT) / ksT

    # Evaluate non-wiggle spectrum back on input k-grid
    # Final interpolation back to input k-grid using CubicSpline
    PNWk = jax.pure_callback(
        _cubic_spline_eval,
        jax.ShapeDtypeStruct(k.shape, jnp.float64),
        k, ksT, PNWT
    )

    # Low-k correction (DeltaAppf) and final stitching using where (no masks/concat)
    DeltaAppf = k * (PSL[7] - PNWT[7]) / PNWT[7] / ksT[7]
    cond_low = k < 1.0e-3
    cond_high = k > ksT[-1]
    P_low = pk / (DeltaAppf + 1.0)
    P_mid = PNWk
    P_high = pk
    PNWkTot = jnp.where(cond_low, P_low, jnp.where(cond_high, P_high, P_mid))

    return (k, PNWkTot)


# Simple python-side cache for get_pknow results to avoid recomputing expensive
# host-callbacks (DST / CubicSpline) when the same input `pk` is requested
# multiple times. This wrapper is intentionally NOT jitted: it lives on the
# Python side, checks a small LRU-like dict keyed by a hash of `pk` and `h`,
# and calls the jitted `get_pknow` only when needed.
_PKNOW_CACHE_SIZE = 16
_pknow_cache = {}
_pknow_cache_keys = []

def _array_hash(arr):
    """Compute a short hash for a numeric array (numpy or jax array).

    The hash takes into account raw bytes, shape and dtype to reduce collisions.
    """
    a = np.asarray(arr)
    h = hashlib.sha1()
    h.update(a.tobytes())
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    return h.hexdigest()


def get_pknow_cached(k, pk, h, use_cache=True):
    """Cacheing wrapper around `get_pknow`.

    Parameters
    ----------
    k, pk, h : as in `get_pknow`.
    use_cache : bool
        If True (default) use the cache. If False, call `get_pknow` always.

    Returns
    -------
    (k_out, pknow_out)
    """
    if not use_cache:
        return get_pknow(k, pk, h)

    key = _array_hash(pk) + f"_h{float(h)}"
    res = _pknow_cache.get(key, None)
    if res is not None:
        return res

    # Not cached: compute (this will run the jitted function, possibly triggering
    # the host callbacks the first time).
    res = get_pknow(k, pk, h)

    # Maintain small LRU-like cache
    if len(_pknow_cache_keys) >= _PKNOW_CACHE_SIZE:
        old = _pknow_cache_keys.pop(0)
        try:
            del _pknow_cache[old]
        except KeyError:
            pass
    _pknow_cache[key] = res
    _pknow_cache_keys.append(key)
    return res



def get_linear_ir(k, pk, h, pknow=None, fullrange=False, kmin=0.01, kmax=0.5, rbao=104, saveout=False):
    """
    Calculates the infrared resummation of the linear power spectrum.

    Parameters:
    k, pk : array_like
        Wave numbers and power spectrum values.
    h : float
        Hubble parameter, H0/100.
    pknow : array_like, optional
        Pre-computed non-wiggle power spectrum.
    fullrange : bool, optional
        If True, returns the full range of k and pk_IRs.
    kmin, kmax : float, optional
        Minimum and maximum k values for filtering.
    rbao : float, optional
        BAO radius for damping.
    saveout : bool, optional
        If True, saves the output to a file.

    Returns:
    tuple
        Filtered or full arrays of k and pk_IRs.
    """
    if pknow is None:
        if h is None:
            raise ValueError("Argument 'h' is required when 'pknow' is None")
        kT, pk_nw = get_pknow(k, pk, h)
    else:
        pk_nw = pknow
    
    p = jnp.geomspace(10**(-6), 0.4, num=100)
    PSL_NW = interp(p, kT, pk_nw)
    sigma2_NW = 1 / (6 * jnp.pi**2) * simpson(PSL_NW * (1 - special.spherical_jn(0, p * rbao) + 2 * special.spherical_jn(2, p * rbao)), x=p)
    pk_IRs = pk_nw + jnp.exp(-kT**2 * sigma2_NW)*(pk - pk_nw)
    
    mask = (kT >= kmin) & (kT <= kmax) & (jnp.arange(len(kT)) % 2 == 0)
    newkT = kT[mask]
    newpk = pk_IRs[mask]
    
    output = (kT, pk_IRs) if fullrange else (newkT, newpk)
                             
    if saveout:
        jnp.savetxt('pk_IR.txt', jnp.array(output).T, delimiter=' ')

    return output



def get_linear_ir_ini(k, pkl, pklnw, h=0.6711, k_BAO=1.0 / 104.):
    """
    Computes the initial infrared-resummed linear power spectrum using a fixed BAO scale.

    Parameters
    ----------
    k : array_like
        Wavenumbers [h/Mpc].
    pkl : array_like
        Linear power spectrum with wiggles.
    pklnw : array_like
        Linear no-wiggle (smooth) power spectrum.
    h : float, optional
        Hubble parameter, H0/100. Default is 0.6711.
    k_BAO : float, optional
        Inverse of the BAO scale in [1/Mpc]. Default is 1.0 / 104.

    Returns
    -------
    tuple of ndarray
        Tuple containing:
            - k : Wavenumbers [h/Mpc].
            - pkl_IR : Infrared-resummed power spectrum.
    """
    # Integration range (geometric spacing)
    p = np.geomspace(1e-6, 0.4, num=100)

    # Interpolate no-wiggle spectrum on integration grid
    pk_nw_interp = interp(p, k, pklnw)

    # Compute damping factor Sigma^2
    j0 = special.spherical_jn(0, p / k_BAO)
    j2 = special.spherical_jn(2, p / k_BAO)
    integrand = pk_nw_interp * (1 - j0 + 2 * j2)
    sigma2 = 1 / (6 * jnp.pi**2) * simpson(integrand, x=p)

    # Apply IR resummation damping
    pkl_IR = pklnw + jnp.exp(-k**2 * sigma2) * (pkl - pklnw)

    return k, pkl_IR


#AP tools
def Hubble(Om, z_ev):
    return jnp.sqrt(Om * (1 + z_ev)**3 + (1 - Om))

def DA(Om, z_ev, nsteps=1000):
    z_grid = jnp.linspace(0.0, z_ev, nsteps)
    dz = z_ev / (nsteps - 1)
    integrand = 1.0 / Hubble(Om, z_grid)
    r = jnp.trapezoid(integrand, dx=dz)
    return r / (1.0 + z_ev)

def qpar_qperp(Omega_fid, Omega_m, z_pk, cosmo=None):
     #check this eqs for CLASS  (see script in external disk)
    if cosmo is not None:
        DA_fid = DA(Omega_fid, z_pk)
        H_fid = Hubble(Omega_fid, z_pk)
        DA_m = cosmo.angular_distance(z_pk)
        H_m = cosmo.Hubble(z_pk)
    else:
        DA_fid = DA(Omega_fid, z_pk)
        DA_m = DA(Omega_m, z_pk)
        H_fid = Hubble(Omega_fid, z_pk)
        H_m = Hubble(Omega_m, z_pk)

    qperp = DA_m / DA_fid
    qpar = H_fid / H_m
    return qpar, qperp


import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# ==================== CubicSpline (not-a-knot, extrapolate=True) ====================

def _solve_tridiag_not_a_knot_second_derivs(x, y):
    """
    Resuelve segundas derivadas s[0..n-1] para spline cúbico con condiciones
    not-a-knot en x[1] y x[n-2]. Implementado 100% JAX y sin ramas Python
    dependientes de trazas (usa lax.cond y fori_loop válidos).
    """
    x = jnp.asarray(x, jnp.float64)
    y = jnp.asarray(y, jnp.float64)
    n = x.size
    h = jnp.diff(x)  # (n-1,)

    main  = jnp.zeros(n,   dtype=jnp.float64)
    lower = jnp.zeros(n-1, dtype=jnp.float64)
    upper = jnp.zeros(n-1, dtype=jnp.float64)
    rhs   = jnp.zeros(n,   dtype=jnp.float64)

    # Filas interiores (i = 1..n-2)
    def fill_interior(i, carry):
        main, lower, upper, rhs = carry
        hi_1 = h[i-1]
        hi   = h[i]
        main = main.at[i].set(2.0*(hi_1 + hi))
        lower = lower.at[i-1].set(hi_1)
        upper = upper.at[i].set(hi)
        rhs_i = 6.0*((y[i+1]-y[i])/hi - (y[i]-y[i-1])/hi_1)
        rhs = rhs.at[i].set(rhs_i)
        return (main, lower, upper, rhs)

    (main, lower, upper, rhs) = jax.lax.fori_loop(
        1, n-1, fill_interior, (main, lower, upper, rhs)
    )

    # Not-a-knot usando forma equivalente estable:
    # s0 - s1 = 0  --> fila 0
    main  = main.at[0].set(1.0)
    upper = upper.at[0].set(-1.0)
    rhs   = rhs.at[0].set(0.0)
    # s_{n-2} - s_{n-1} = 0 --> fila n-1
    main  = main.at[n-1].set(1.0)
    lower = lower.at[n-2].set(-1.0)
    rhs   = rhs.at[n-1].set(0.0)

    # Thomas: forward sweep (i = 1..n-1)
    # cprime[i] definido para i = 0..n-2
    cprime = jnp.zeros(n-1, dtype=jnp.float64)
    dprime = jnp.zeros(n,   dtype=jnp.float64)
    cprime = cprime.at[0].set(upper[0]/main[0])
    dprime = dprime.at[0].set(rhs[0]/main[0])

    def forward(i, vals):
        cprime, dprime = vals
        denom = main[i] - lower[i-1]*cprime[i-1]
        # dprime[i]
        dprime_i = (rhs[i] - lower[i-1]*dprime[i-1]) / denom
        dprime = dprime.at[i].set(dprime_i)

        # cprime[i] si i < n-1; si i == n-1, no se escribe (evita OOB)
        def set_cprime(_):
            return cprime.at[i].set(upper[i]/denom)
        def keep(_):
            return cprime
        cprime = jax.lax.cond(i < (n-1), set_cprime, keep, operand=None)
        return (cprime, dprime)

    cprime, dprime = jax.lax.fori_loop(1, n, forward, (cprime, dprime))

    # Thomas: backward substitution
    # s[n-1] = dprime[n-1]
    s = jnp.zeros(n, dtype=jnp.float64).at[n-1].set(dprime[n-1])

    # Recorremos t = 0..n-2 y usamos idx = n-2 - t  => i = n-2, ..., 0
    def backward(t, s):
        i = (n - 2) - t
        # s[i] = dprime[i] - cprime[i] * s[i+1]
        return s.at[i].set(dprime[i] - cprime[i]*s[i+1])

    s = jax.lax.fori_loop(0, n-1, backward, s)
    return s

def _cubic_eval_not_a_knot(xq, x, y, s):
    """
    Evalúa el spline cúbico (con segundas derivadas s) en xq.
    Extrapola usando el primer/último tramo (extrapolate=True).
    """
    x = jnp.asarray(x, jnp.float64)
    y = jnp.asarray(y, jnp.float64)
    s = jnp.asarray(s, jnp.float64)
    # índice de tramo (clamp para extrapolación)
    i = jnp.clip(jnp.searchsorted(x, xq, side="right") - 1, 0, x.size - 2)

    xi   = x[i]
    xi1  = x[i+1]
    hi   = xi1 - xi
    si   = s[i]
    si1  = s[i+1]
    yi   = y[i]
    yi1  = y[i+1]

    dx1 = xi1 - xq
    dx  = xq - xi

    term1 = si  * (dx1**3) / (6.0*hi)
    term2 = si1 * (dx**3)  / (6.0*hi)
    term3 = (yi  - si *(hi**2)/6.0) * (dx1/hi)
    term4 = (yi1 - si1*(hi**2)/6.0) * (dx/hi)
    return term1 + term2 + term3 + term4

def cubic_spline_not_a_knot_eval(xq, x, y):
    """Interpolación cúbica not-a-knot con extrapolación, 100% JAX."""
    s = _solve_tridiag_not_a_knot_second_derivs(x, y)
    xq = jnp.asarray(xq, jnp.float64)
    return jax.vmap(lambda t: _cubic_eval_not_a_knot(t, x, y, s))(xq)


# ==================== DST-I / IDST-I (norm="ortho") 100% JAX ====================

def dst1_ortho(x):
    """
    DST-I con normalización ortonormal (equivalente a SciPy: norm='ortho').
    Implementación vía RFFT sobre la extensión impar:
      y = [0, x1..xN, 0, -xN..-x1],  len(y) = 2*(N+1)
    Identidad correcta (¡incluye 1/2!):
      DST-I(x) = -0.5 * Im( RFFT(y) )[1:N+1] * sqrt(2/(N+1))
    """
    x = jnp.asarray(x, jnp.float64)
    N = x.shape[0]
    y = jnp.concatenate([jnp.array([0.0], jnp.float64),
                         x,
                         jnp.array([0.0], jnp.float64),
                         -x[::-1]])
    Y = jnp.fft.rfft(y)  # longitud 2*(N+1) -> rfft devuelve N+2 puntos
    return (-0.5 * Y.imag[1:N+1]) * jnp.sqrt(2.0/(N+1))

def idst1_ortho(X):
    """
    Inversa ortonormal de DST-I. Para la versión 'ortho', la inversa es la misma
    operación (matriz ortonormal: U^{-1} = U^T = U).
    """
    return dst1_ortho(X)


# ==================== Algoritmo BAO non-wiggle (Hamann+ 2010) ====================

@jax.jit
def get_pknow_jax(k, pk, h,
                  nk=2**16, mcutmin=120, mcutmax=240):
    """
    100% JAX, misma lógica que tu referencia:
      - CubicSpline not-a-knot (extrapolate=True)
      - DST/IDST tipo-I 'ortho'
    """
    k = jnp.asarray(k, jnp.float64)
    pk = jnp.asarray(pk, jnp.float64)

    kmin = 7.0e-5 / h
    kmax = 7.0     / h

    ksT = kmin + jnp.arange(nk, dtype=jnp.float64) * (kmax - kmin) / (nk - 1)

    # Interpola P(k) a la malla interna con el MISMO esquema cúbico
    PSL = cubic_spline_not_a_knot_eval(ksT, k, pk)

    # DST-I ortho de log(k P(k))
    logkpk = jnp.log(ksT * PSL)
    FST = dst1_ortho(logkpk)  # tamaño nk

    # Separar impares/pares
    FST_odd  = FST[::2]
    FST_even = FST[1::2]
    len_even = nk // 2

    # Cortes (par)
    xEvenTcutmin = jnp.arange(1, mcutmin - 1, 1)
    xEvenTcutmax = jnp.arange(mcutmax + 2, len_even + 1, 1)
    EvenTcutmin  = FST_even[0:mcutmin - 2]
    EvenTcutmax  = FST_even[mcutmax + 1:len_even]
    xEvenTcutted = jnp.concatenate((xEvenTcutmin, xEvenTcutmax)).astype(jnp.float64)
    F_even_cut   = jnp.concatenate((EvenTcutmin, EvenTcutmax))

    # Cortes (impar)
    xOddTcutmin  = jnp.arange(1, mcutmin, 1)
    xOddTcutmax  = jnp.arange(mcutmax + 1, len_even + 1, 1)
    OddTcutmin   = FST_odd[0:mcutmin - 1]
    OddTcutmax   = FST_odd[mcutmax:len_even]
    xOddTcutted  = jnp.concatenate((xOddTcutmin, xOddTcutmax)).astype(jnp.float64)
    F_odd_cut    = jnp.concatenate((OddTcutmin, OddTcutmax))

    # Re-interpolar armónicos dentro de la ventana BAO con el mismo spline
    qEven = jnp.arange(2, mcutmax + 1, 1.0)
    qOdd  = jnp.arange(0, mcutmax - 1, 1.0)
    PreEven = cubic_spline_not_a_knot_eval(qEven, xEvenTcutted, F_even_cut)
    PreOdd  = cubic_spline_not_a_knot_eval(qOdd,  xOddTcutted,  F_odd_cut)

    pre_middle = jnp.column_stack([
        PreOdd[mcutmin:mcutmax - 1],
        PreEven[mcutmin:mcutmax - 1]
    ]).ravel()

    pre = jnp.concatenate([
        FST[:2 * mcutmin],
        pre_middle,
        FST[2 * mcutmax - 2:]
    ])

    # IDST-I ortho (autoinversa)
    logkpk_nw = idst1_ortho(pre)
    PNWT = jnp.exp(logkpk_nw) / ksT

    # Interpola P_nw a la malla original con el mismo spline
    PNWk = cubic_spline_not_a_knot_eval(k, ksT, PNWT)

    # Corrección low-k y stitching (idéntico a la referencia)
    DeltaAppf = k * (PSL[7] - PNWT[7]) / PNWT[7] / ksT[7]
    cond_low  = k < 1.0e-3
    cond_high = k > ksT[-1]
    P_low  = pk / (DeltaAppf + 1.0)
    P_mid  = PNWk
    P_high = pk
    PNWkTot = jnp.where(cond_low, P_low, jnp.where(cond_high, P_high, P_mid))
    return k, PNWkTot