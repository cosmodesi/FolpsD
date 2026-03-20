import warnings
import numpy as np
from scipy import interpolate
from scipy import special
from scipy.integrate import quad


def legendre(ell):
    """
    Return Legendre polynomial of given order.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Legendre_polynomials
    """
    if ell == 0:

        return lambda x: np.ones_like(x)

    if ell == 2:

        return lambda x: 1. / 2. * (3 * x**2 - 1)

    if ell == 4:

        return lambda x: 1. / 8. * (35 * x**4 - 30 * x**2 + 3)

    raise NotImplementedError('Legendre polynomial for ell = {:d} not implemented'.format(ell))


def interp(k, x, y):
    from scipy import interpolate

    '''Cubic interpolator.
    
    Args:
        k: coordinates at which to evaluate the interpolated values.
        kev: x-coordinates of the data points.
        Table: list of 1-loop contributions for the wiggle and non-wiggle
    '''
    f = interpolate.interp1d(x, y, kind = 'cubic', fill_value = "extrapolate")
    
    return np.asarray(f(k), dtype=np.float64)

_NoValue = None


def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)


def true_divide(h0, h1, out=None, where=None):
    if out is None:
        out = np.zeros_like(h1)
    if where is None:
        out = out.at[...].set(h0 / h1)
        return out
    return np.where(np.asarray(where), h0 / h1, out)


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
        result = np.sum(y[slice0] + 4.0*y[slice1] + y[slice2], axis=axis)
        result *= dx / 3.0
    else:
        # Account for possibly different spacings.
        #    Simpson's rule changes a bit.
        h = np.diff(x, axis=axis)
        sl0 = tupleset(slice_all, axis, slice(start, stop, step))
        sl1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
        h0 = h[sl0].astype(float)
        h1 = h[sl1].astype(float)
        hsum = h0 + h1
        hprod = h0 * h1
        h0divh1 = true_divide(h0, h1, out=np.zeros_like(h0), where=h1 != 0)
        tmp = hsum/6.0 * (y[slice0] *
                          (2.0 - true_divide(1.0, h0divh1,
                                                out=np.zeros_like(h0divh1),
                                                where=h0divh1 != 0)) +
                          y[slice1] * (hsum *
                                       true_divide(hsum, hprod,
                                                      out=np.zeros_like(hsum),
                                                      where=hprod != 0)) +
                          y[slice2] * (2.0 - h0divh1))
        result = np.sum(tmp, axis=axis)
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
    >>> import numpy as np
    >>> x = np.arange(0, 10)
    >>> y = np.arange(0, 10)

    >>> integrate.simpson(y, x)
    40.5

    >>> y = np.power(x, 3)
    >>> integrate.simpson(y, x)
    1640.5
    >>> integrate.quad(lambda x: x**3, 0, 9)[0]
    1640.25

    >>> integrate.simpson(y, x, even='first')
    1644.5

    """
    y = np.asarray(y)
    nd = len(y.shape)
    N = y.shape[axis]
    last_dx = dx
    first_dx = dx
    returnshape = 0
    if x is not None:
        x = np.asarray(x)
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

            h = np.asarray([dx, dx], dtype=np.float64)
            if x is not None:
                # grab the last two spacings from the appropriate axis
                hm2 = tupleset(slice_all, axis, slice(-2, -1, 1))
                hm1 = tupleset(slice_all, axis, slice(-1, None, 1))

                diffs = np.float64(np.diff(x, axis=axis))
                h = [np.squeeze(diffs[hm2], axis=axis),
                     np.squeeze(diffs[hm1], axis=axis)]

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
                out=np.zeros_like(den),
                where=den != 0
            )

            num = h[1] ** 2 + 3.0 * h[0] * h[1]
            den = 6 * h[0]
            beta = true_divide(
                num,
                den,
                out=np.zeros_like(den),
                where=den != 0
            )

            num = 1 * h[1] ** 3
            den = 6 * h[0] * (h[0] + h[1])
            eta = true_divide(
                num,
                den,
                out=np.zeros_like(den),
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
        xm = np.mean(x)
        ym = np.mean(y)
        npts = len(x)

        SS_xy = np.sum(x * y) - npts * xm * ym
        SS_xx = np.sum(x**2) - npts * xm**2
        m = SS_xy / SS_xx

        b = ym - m * xm
        return (m, b)

    m, b = linear_regression(x, y)
    return (xq, m * xq + b)


def extrapolate_linear_loglog(k, pk, kcut, k_extrapolate, is_high_k=True):
    """
    Generic extrapolation function for high-k or low-k values using logarithmic scaling.

    Args:
        k: Array of k-coordinates.
        pk: Array of power spectrum values corresponding to k.
        kcut: Value of 'k' from which extrapolation will begin.
        k_extrapolate: Value of 'k' up to which extrapolation will be performed.
        is_high_k: Boolean indicating if extrapolation is for high-k (True) or low-k (False).
    
    Returns:
        Extrapolated k and pk arrays from 'kcut' to 'k_extrapolate'.
    """
    if is_high_k:
        cutrange = np.where(k <= kcut)
        index_slice = slice(-6, None)
    else:
        cutrange = np.where(k > kcut)
        index_slice = slice(None, 5)
    
    k_cut = k[cutrange]
    pk_cut = pk[cutrange]
    k_to_extrapolate = k_cut[index_slice]
    pk_to_extrapolate = pk_cut[index_slice]

    delta_log_k = np.log10(k_to_extrapolate[2]) - np.log10(k_to_extrapolate[1])
    last_or_first_k = np.log10(k_to_extrapolate[-1]) if is_high_k else np.log10(k_to_extrapolate[0])
    
    log_k_list = []
    while last_or_first_k <= np.log10(k_extrapolate) if is_high_k else last_or_first_k > np.log10(k_extrapolate):
        last_or_first_k += delta_log_k if is_high_k else -delta_log_k
        log_k_list.append(last_or_first_k)
    
    log_k_list = np.array(log_k_list) if is_high_k else np.array(list(reversed(log_k_list)))
    
    sign = np.sign(pk_to_extrapolate[1])
    pk_to_extrapolate_log = np.log10(np.abs(pk_to_extrapolate))
    log_extrapolated_k, log_extrapolated_pk = extrapolate(np.log10(k_to_extrapolate), pk_to_extrapolate_log, log_k_list)
    
    extrapolated_k = 10**log_extrapolated_k
    extrapolated_pk = sign * 10**log_extrapolated_pk
    
    if is_high_k:
        k_result = np.concatenate((k_cut, extrapolated_k))
        pk_result = np.concatenate((pk_cut, extrapolated_pk))
    else:
        k_result = np.concatenate((extrapolated_k, k_cut))
        pk_result = np.concatenate((extrapolated_pk, pk_cut))
    
    return k_result, pk_result


def extrapolate_high_k(k, pk, kcutmax, kmax):
    """
    Extrapolate for high-k values using logarithmic scaling.

    Args:
        k: Array of k-coordinates.
        pk: Array of power spectrum values corresponding to k.
        kcutmax: Value of 'k' from which extrapolation will begin.
        kmax: Value of 'k' up to which extrapolation will be performed.
    
    Returns:
        Extrapolated k and pk arrays for high-k values.
    """
    return extrapolate_linear_loglog(k, pk, kcutmax, kmax, is_high_k=True)


def extrapolate_low_k(k, pk, kcutmin, kmin):
    """
    Extrapolate for low-k values using logarithmic scaling.

    Args:
        k: Array of k-coordinates.
        pk: Array of power spectrum values corresponding to k.
        kcutmin: Value of 'k' from which extrapolation will begin.
        kmin: Value of 'k' up to which extrapolation will be performed.
    
    Returns:
        Extrapolated k and pk arrays for low-k values.
    """
    return extrapolate_linear_loglog(k, pk, kcutmin, kmin, is_high_k=False)


def extrapolate_k(k, pk, kcutmin, kmin, kcutmax, kmax):
    """
    Extrapolate for both low-k and high-k values.

    Args:
        k: Array of k-coordinates.
        pk: Array of power spectrum values corresponding to k.
        kcutmin, kcutmax: Values of 'k' from which extrapolation will begin.
        kmin, kmax: Values of 'k' up to which extrapolation will be performed.
    
    Returns:
        Extrapolated k and pk arrays for both low-k and high-k values.
    """
    k_high, pk_high = extrapolate_high_k(k, pk, kcutmax, kmax)
    return extrapolate_low_k(k_high, pk_high, kcutmin, kmin)


def extrapolate_pklin(k, pk):
    """
    Extrapolate the input linear power spectrum to low-k or high-k if needed.

    Args:
        k: Array of k-coordinates.
        pk: Array of power spectrum values corresponding to k.
    
    Returns:
        Extrapolated k and pk arrays for the full desired k-range.
    """
    kmin = 1e-5
    kmax = 200
    kcutmin = min(k)
    kcutmax = max(k)

    if kmin < kcutmin or kmax > kcutmax:
        return extrapolate_k(k, pk, kcutmin, kmin, kcutmax, kmax)
    else:
        return k, pk
    
    

def get_pknow(k, pk, h):
    """
    Routine (based on J. Hamann et. al. 2010, arXiv:1003.3999) to get the non-wiggle piece of the linear power spectrum.

    Args:
        k: wave-number.
        pk: linear power spectrum.
        h: H0/100.
    Returns:
        non-wiggle piece of the linear power spectrum.
    """
    def interp(k, x, y):  # out-of-range below
        from scipy.interpolate import CubicSpline
        return CubicSpline(x, y)(k)

    from scipy.fft import dst, idst  # not in jax yet...
    #kmin(max): k-range and nk: points
    kmin = 7 * 10**(-5) / h; kmax = 7 / h; nk = 2**16

    #sample ln(kP_L(k)) in nk points, k range (equidistant)
    ksT = kmin + np.arange(nk) * (kmax - kmin) / (nk - 1)
    PSL = interp(ksT, k, pk)
    logkpk = np.log(ksT * PSL)

    #Discrete sine transf., check documentation
    FSTlogkpkT = dst(np.array(logkpk), type=1, norm="ortho")
    FSTlogkpkOddT = FSTlogkpkT[::2]
    FSTlogkpkEvenT = FSTlogkpkT[1::2]

    #cut range (remove the harmonics around BAO peak)
    mcutmin = 120; mcutmax = 240

    #Even
    xEvenTcutmin = np.arange(1, mcutmin - 1, 1)
    xEvenTcutmax = np.arange(mcutmax + 2, len(FSTlogkpkEvenT) + 1, 1)
    EvenTcutmin = FSTlogkpkEvenT[0:mcutmin - 2]
    EvenTcutmax = FSTlogkpkEvenT[mcutmax + 1:len(FSTlogkpkEvenT)]
    xEvenTcuttedT = np.concatenate((xEvenTcutmin, xEvenTcutmax))
    nFSTlogkpkEvenTcuttedT = np.concatenate((EvenTcutmin, EvenTcutmax))

    #Odd
    xOddTcutmin = np.arange(1, mcutmin, 1)
    xOddTcutmax = np.arange(mcutmax + 1, len(FSTlogkpkEvenT) + 1, 1)
    OddTcutmin = FSTlogkpkOddT[0:mcutmin - 1]
    OddTcutmax = FSTlogkpkOddT[mcutmax:len(FSTlogkpkEvenT)]
    xOddTcuttedT = np.concatenate((xOddTcutmin, xOddTcutmax))
    nFSTlogkpkOddTcuttedT = np.concatenate((OddTcutmin, OddTcutmax))

    #Interpolate the FST harmonics in the BAO range
    PreEvenT = interp(np.arange(2, mcutmax + 1, 1.), xEvenTcuttedT, nFSTlogkpkEvenTcuttedT)
    PreOddT = interp(np.arange(0, mcutmax - 1, 1.), xOddTcuttedT, nFSTlogkpkOddTcuttedT)
    preT = np.column_stack([PreOddT[mcutmin:mcutmax - 1], PreEvenT[mcutmin:mcutmax - 1]]).ravel()
    preT = np.concatenate([FSTlogkpkT[:2 * mcutmin], preT, FSTlogkpkT[2 * mcutmax - 2:]])

    #Inverse Sine transf.
    FSTofFSTlogkpkNWT = idst(np.array(preT), type=1, norm="ortho")
    PNWT = np.exp(FSTofFSTlogkpkNWT)/ksT

    PNWk = interp(k, ksT, PNWT)
    DeltaAppf = k*(PSL[7]-PNWT[7])/PNWT[7]/ksT[7]

    irange1 = k < 1e-3
    PNWk1 = pk[irange1] / (DeltaAppf[irange1] + 1)

    irange2 = (1e-3 <= k) & (k <= ksT[len(ksT)-1])
    PNWk2 = PNWk[irange2]

    irange3 = (k > ksT[len(ksT)-1])
    PNWk3 = pk[irange3]

    PNWkTot = np.concatenate([PNWk1, PNWk2, PNWk3])

    return(k, PNWkTot)



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
    
    p = np.geomspace(10**(-6), 0.4, num=100)
    PSL_NW = interp(p, kT, pk_nw)
    sigma2_NW = 1 / (6 * np.pi**2) * simpson(PSL_NW * (1 - special.spherical_jn(0, p * rbao) + 2 * special.spherical_jn(2, p * rbao)), x=p)
    pk_IRs = pk_nw + np.exp(-kT**2 * sigma2_NW)*(pk - pk_nw)
    
    mask = (kT >= kmin) & (kT <= kmax) & (np.arange(len(kT)) % 2 == 0)
    newkT = kT[mask]
    newpk = pk_IRs[mask]
    
    output = (kT, pk_IRs) if fullrange else (newkT, newpk)
                             
    if saveout:
        np.savetxt('pk_IR.txt', np.array(output).T, delimiter=' ')

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
    sigma2 = 1 / (6 * np.pi**2) * simpson(integrand, x=p)

    # Apply IR resummation damping
    pkl_IR = pklnw + np.exp(-k**2 * sigma2) * (pkl - pklnw)

    return k, pkl_IR



#AP factors

def Hubble(Om, z_ev):
    return ((Om) * (1 + z_ev)**3. + (1 - Om))**0.5

def DA(Om, z_ev):
    r = quad(lambda x: 1. / Hubble(Om, x), 0, z_ev)[0]
    return r / (1 + z_ev)

def qpar_qperp(Omega_fid, Omega_m, z_pk, cosmo=None):
    """
    Compute qpar and qperp using analytical formulas or a cosmo object from CLASS.
    """

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


### new debugging ###

# Finally I ended using:
#def interp(k, x, y):  # out-of-range below
#        from scipy.interpolate import CubicSpline
#        return CubicSpline(x, y)(k)

#and interp_new() for tool_jax.py

#def interp_new(xq, x, f, method='cubic'):
#    from jax import numpy as jnp
#    import interpax

#    """
#    Interpolate a 1d function.

#    Note
#    ----
#    Using interpax: https://github.com/f0uriest/interpax

#    Parameters
#    ----------
#    xq : ndarray, shape(Nq,)
#        query points where interpolation is desired
#    x : ndarray, shape(Nx,)
#        coordinates of known function values ("knots")
#    f : ndarray, shape(Nx,...)
#        function values to interpolate
#    method : str
#        method of interpolation

#        - ``'nearest'``: nearest neighbor interpolation
#        - ``'linear'``: linear interpolation
#        - ``'cubic'``: C1 cubic splines (aka local splines)
#        - ``'cubic2'``: C2 cubic splines (aka natural splines)
#        - ``'catmull-rom'``: C1 cubic centripetal "tension" splines
#        - ``'cardinal'``: C1 cubic general tension splines. If used, can also pass
#          keyword parameter ``c`` in float[0,1] to specify tension
#        - ``'monotonic'``: C1 cubic splines that attempt to preserve monotonicity in the
#          data, and will not introduce new extrema in the interpolated points
#        - ``'monotonic-0'``: same as ``'monotonic'`` but with 0 first derivatives at
#          both endpoints

#    derivative : int >= 0
#        derivative order to calculate
#    extrap : bool, float, array-like
#        whether to extrapolate values beyond knots (True) or return nan (False),
#        or a specified value to return for query points outside the bounds. Can
#        also be passed as a 2 element array or tuple to specify different conditions
#        for xq<x[0] and x[-1]<xq
#    period : float > 0, None
#        periodicity of the function. If given, function is assumed to be periodic
#        on the interval [0,period]. None denotes no periodicity

#    Returns
#    -------
#    fq : ndarray, shape(Nq,...)
#        function value at query points
#    """
#    method = {1: 'linear', 3: 'cubic'}.get(method, method)
#    xq = jnp.asarray(xq)
#    shape = xq.shape
#    return interpax.interp1d(xq.reshape(-1), x, f, method=method, extrap=False).reshape(shape + f.shape[1:])



#### debugging (old routine) ###
from scipy.fft import dst, idst

def pknwJ(k, PSLk, h):
    '''Routine (based on J. Hamann et. al. 2010, arXiv:1003.3999) to get the non-wiggle piece of the linear power spectrum.    
    
    Args:
        k: wave-number.
        PSLk: linear power spectrum.
        h: H0/100.
    Returns:
        non-wiggle piece of the linear power spectrum.
    '''
    #ksmin(max): k-range and Nks: points
    ksmin = 7*10**(-5)/h; ksmax = 7/h; Nks = 2**16

    #sample ln(kP_L(k)) in Nks points, k range (equidistant)
    ksT = [ksmin + ii*(ksmax-ksmin)/(Nks-1) for ii in range(Nks)]
    PSL = interp(ksT, k, PSLk)
    logkpkT = np.log(ksT*PSL)
        
    #Discrete sine transf., check documentation
    FSTtype = 1; m = int(len(ksT)/2)
    FSTlogkpkT = dst(logkpkT, type = FSTtype, norm = "ortho")
    FSTlogkpkOddT = FSTlogkpkT[::2]
    FSTlogkpkEvenT = FSTlogkpkT[1::2]
        
    #cut range (remove the harmonics around BAO peak)
    mcutmin = 120; mcutmax = 240;
        
    #Even
    xEvenTcutmin = np.linspace(1, mcutmin-2, mcutmin-2)
    xEvenTcutmax = np.linspace(mcutmax+2, len(FSTlogkpkEvenT), len(FSTlogkpkEvenT)-mcutmax-1)
    EvenTcutmin = FSTlogkpkEvenT[0:mcutmin-2] 
    EvenTcutmax = FSTlogkpkEvenT[mcutmax+1:len(FSTlogkpkEvenT)]
    xEvenTcuttedT = np.concatenate((xEvenTcutmin, xEvenTcutmax))
    nFSTlogkpkEvenTcuttedT = np.concatenate((EvenTcutmin, EvenTcutmax))


    #Odd
    xOddTcutmin = np.linspace(1, mcutmin-1, mcutmin-1)
    xOddTcutmax = np.linspace(mcutmax+1, len(FSTlogkpkEvenT), len(FSTlogkpkEvenT)-mcutmax)
    OddTcutmin = FSTlogkpkOddT[0:mcutmin-1]
    OddTcutmax = FSTlogkpkOddT[mcutmax:len(FSTlogkpkEvenT)]
    xOddTcuttedT = np.concatenate((xOddTcutmin, xOddTcutmax))
    nFSTlogkpkOddTcuttedT = np.concatenate((OddTcutmin, OddTcutmax))

    #Interpolate the FST harmonics in the BAO range
    preT, = map(np.zeros,(len(FSTlogkpkT),))
    PreEvenT = interp(np.linspace(2, mcutmax, mcutmax-1), xEvenTcuttedT, nFSTlogkpkEvenTcuttedT)
    PreOddT = interp(np.linspace(0, mcutmax-2, mcutmax-1), xOddTcuttedT, nFSTlogkpkOddTcuttedT)
    for ii in range(m):
        if (mcutmin < ii+1 < mcutmax):
            preT[2*ii+1] = PreEvenT[ii]
            preT[2*ii] = PreOddT[ii]
        if (mcutmin >= ii+1 or mcutmax <= ii+1):
            preT[2*ii+1] = FSTlogkpkT[2*ii+1]
            preT[2*ii] = FSTlogkpkT[2*ii]
                
        
    #Inverse Sine transf.
    FSTofFSTlogkpkNWT = idst(preT, type = FSTtype, norm = "ortho")
    PNWT = np.exp(FSTofFSTlogkpkNWT)/ksT

    PNWk = interp(k, ksT, PNWT)
    DeltaAppf = k*(PSL[7]-PNWT[7])/PNWT[7]/ksT[7]

    irange1 = np.where((k < 1e-3))
    PNWk1 = PSLk[irange1]/(DeltaAppf[irange1] + 1)

    irange2 = np.where((1e-3 <= k) & (k <= ksT[len(ksT)-1]))
    PNWk2 = PNWk[irange2]
        
    irange3 = np.where((k > ksT[len(ksT)-1]))
    PNWk3 = PSLk[irange3]
        
    PNWkTot = np.concatenate((PNWk1, PNWk2, PNWk3))
        
    return(k, PNWkTot)

