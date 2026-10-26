r"""Direct-quadrature reference for the selected one-loop spectrum (validation only).

Independent of the FFTLog templates and of the generated coefficient data: kernels come
from the field algebra of :mod:`folps.selection_kernels` and integrals are done by brute
force over ``q`` (radial, polar angle and *azimuth*, since a general LOS breaks the
azimuthal symmetry about ``k``).  Slow, host NumPy only; never used in likelihoods.

P22: ``2 \int_q [Z2(q, k-q)^2 P(q) P(|k-q|) - Z2(q, -q)^2 P(q)^2]``.  The integrand is
symmetric under ``q <-> k - q``, so the integral is twice the integral over the region
``|q| < |k - q|``, in which the only singular point is the soft limit ``q -> 0`` (handled
by a logarithmic radial grid).  The zero-lag term is integrated separately over all ``q``.

P13: ``6 Z1(k) P(k) \int_q [Z3(k, q, -q) - Z3_UV(mu, q-hat)] P(q)`` where the subtracted
constant is the large-``q`` limit of the angular-averaged kernel (a renormalization of
``b1`` and ``b_eta``), evaluated numerically from the same kernel at ``q/k = 1e6``.  The
radial grid is split at ``q = k`` where the angular-averaged kernel has an integrable
logarithmic singularity.  No order-``k^2 sigma^2`` term is removed (bare scheme).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from . import selection_kernels as sk

__all__ = ["LinearSpectrum", "direct_p22", "direct_p13_ratio", "direct_p13", "mapped_field", "exact_map_field_from_djs"]


class LinearSpectrum:
    """log-log cubic spline of a tabulated spectrum, zero outside ``[k[0], k[-1]]``."""

    def __init__(self, k, pk):
        self.k = np.asarray(k, float)
        self.pk = np.asarray(pk, float)
        self._s = CubicSpline(np.log(self.k), np.log(self.pk))

    def __call__(self, x):
        x = np.asarray(x, float)
        inside = (x >= self.k[0]) & (x <= self.k[-1])
        xc = np.clip(x, self.k[0], self.k[-1])
        return np.where(inside, np.exp(self._s(np.log(xc))), 0.0)


def _los(mu):
    return np.array([np.sqrt(max(0.0, 1.0 - mu * mu)), 0.0, mu])


def mapped_field(coeffs):
    """DJS mapped field from a ``DJSCoefficients``-like mapping."""
    c = coeffs._asdict() if hasattr(coeffs, "_asdict") else dict(coeffs)
    return sk.mapped_djs_field(c)


def exact_map_field_from_djs(coeffs):
    """Exact redshift-space map of the rest-frame density reproducing the DJS coefficients."""
    c = coeffs._asdict() if hasattr(coeffs, "_asdict") else dict(coeffs)
    rest = dict(b1=c["b1"], bR_eta=c["b_eta"] + 1.0, b2=c["b2"], b_K2=c["b_K2"],
                bR_delta_eta=c["b_delta_eta"] + c["b1"], bR_eta2=c["b_eta2"] + c["b_eta"],
                bR_KK_par=c["b_KK_par"], bR_Pi2_par=c["b_Pi2_par"], b_td=c["b_td"],
                bR_delta_Pi2_par=c["b_delta_Pi2_par"], bR_eta_Pi2_par=c["b_eta_Pi2_par"] + c["b_Pi2_par"],
                bR_KPi2_par=c["b_KPi2_par"], bR_Pi3_par=c["b_Pi3_par"])
    return sk.exact_map_field(sk.rest_frame_density(rest))


def _gl(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def direct_p22(k, mu, P, field, f, qmin=1e-5, qmax=200.0, nq=400, nnu=48, nphi=24, chunk=40):
    """Constant-subtracted selected P22 at a single ``(k, mu)`` by direct quadrature.

    Radial: Gauss-Legendre in ``ln q`` on ``[qmin, k/2]``, ``[k/2, 4k]`` and ``[4k, qmax]``
    (``nq`` total nodes); polar: Gauss-Legendre in ``nu`` on ``[-1, min(1, k/(2q))]``;
    azimuth: ``nphi``-point trapezoid (exact for trigonometric polynomials of lower degree).
    """
    n = _los(mu)
    kv = np.array([0.0, 0.0, k])
    segs = [(np.log(qmin), np.log(0.5 * k), int(nq * 0.4)), (np.log(0.5 * k), np.log(4 * k), int(nq * 0.3)),
            (np.log(4 * k), np.log(qmax), nq - int(nq * 0.4) - int(nq * 0.3))]
    phi = 2 * np.pi * np.arange(nphi) / nphi
    total = 0.0
    for a, b, m in segs:
        lq, wl = _gl(a, b, m)
        for i0 in range(0, m, chunk):
            q = np.exp(lq[i0:i0 + chunk])
            w_r = wl[i0:i0 + chunk] * q ** 3 / (2 * np.pi) ** 3  # d^3q = q^3 dlnq dOmega
            numax = np.minimum(1.0, k / (2 * q))
            xg, wg = np.polynomial.legendre.leggauss(nnu)
            nu = 0.5 * (numax[:, None] + 1) * xg[None, :] + 0.5 * (numax[:, None] - 1)
            wnu = 0.5 * (numax[:, None] + 1) * wg[None, :]
            st = np.sqrt(np.clip(1 - nu ** 2, 0, None))
            qv = q[:, None, None, None] * np.stack(
                [st[..., None] * np.cos(phi), st[..., None] * np.sin(phi), np.broadcast_to(nu[..., None], st.shape + (nphi,))], -1)
            ev = sk.Evaluator([qv, kv - qv], n, f)
            Z2 = field.at(ev, 2).real
            Qn = np.linalg.norm(kv - qv, axis=-1)
            integ = 2 * Z2 ** 2 * P(q)[:, None, None] * P(Qn)
            ang = np.sum(wnu[..., None] * integ, axis=(1, 2)) * (2 * np.pi / nphi)
            total += 2 * np.sum(w_r * ang)  # x2: region |q| < |k-q| only
    # zero-lag: 2 \int_q Z2(q,-q)^2 P(q)^2 over all q
    lq, wl = _gl(np.log(qmin), np.log(qmax), nq)
    q = np.exp(lq)
    xg, wg = np.polynomial.legendre.leggauss(nnu)
    st = np.sqrt(1 - xg ** 2)
    qv = q[:, None, None, None] * np.stack([st[:, None] * np.cos(phi), st[:, None] * np.sin(phi),
                                            np.broadcast_to(xg[:, None], (nnu, nphi))], -1)[None]
    ev = sk.Evaluator([qv, -qv], n, f)
    Z0 = field.at(ev, 2).real
    ang = np.sum(wg[:, None] * 2 * Z0 ** 2, axis=(1, 2)) * (2 * np.pi / nphi)
    zero = np.sum(wl * q ** 3 / (2 * np.pi) ** 3 * ang * P(q) ** 2)
    return total - zero


def _nu_rule(x, nnu):
    """Polar rule on [-1, 1] adapted to the near-poles of 1/|k -/+ q|^2 at nu = +/-1, whose
    distance is d = (1-x)^2/(2x): on each half, 1 - |nu| = d (exp(s) - 1), s uniform-GL."""
    d = max((1.0 - x) ** 2 / (2.0 * x), 1e-300)
    xg, wg = np.polynomial.legendre.leggauss(nnu // 2)
    smax = np.log1p(1.0 / d)
    s = 0.5 * smax * (xg + 1)
    ws = 0.5 * smax * wg
    t = d * np.expm1(s)               # t = 1 - |nu| in [0, 1]
    wt = ws * (t + d)                 # dt = (t + d) ds
    nu = np.concatenate([1 - t, -(1 - t)])
    w = np.concatenate([wt, wt])
    return nu, w


def _avg13(k, x, n, field, f, nnu, nphi):
    """Angular average of 3 Z3(k, q, -q) at q = x k (k along z)."""
    phi = 2 * np.pi * np.arange(nphi) / nphi
    x = np.atleast_1d(x)
    out = np.empty(len(x))
    for i, xi in enumerate(x):
        nu, w = _nu_rule(xi, nnu)
        st = np.sqrt(np.clip(1 - nu ** 2, 0, None))
        qv = (xi * k) * np.stack([st[:, None] * np.cos(phi), st[:, None] * np.sin(phi),
                                   np.broadcast_to(nu[:, None], (len(nu), nphi))], -1)
        kv = np.broadcast_to(np.array([0.0, 0.0, k]), qv.shape)
        ev = sk.Evaluator([kv, qv, -qv], n, f)
        Z3 = field.at(ev, 3).real
        out[i] = np.sum(w[:, None] * 3 * Z3) / (2 * nphi)
    return out


def _radial_rule(k, qmin, qmax, nq):
    """ln-q Gauss-Legendre away from q = k, exponential clustering on both sides of q = k
    (integrable logarithmic singularity of the angular average there)."""
    m_out, m_in = int(0.3 * nq), int(0.2 * nq)
    pts, wts = [], []
    lq, wl = _gl(np.log(qmin), np.log(0.5 * k), m_out)
    pts.append(np.exp(lq)); wts.append(wl * np.exp(lq))
    U = 30.0
    u, wu = _gl(0.0, U, m_in)
    x = 1 - 0.5 * np.exp(-u)          # [k/2, k)
    pts.append(k * x); wts.append(k * 0.5 * np.exp(-u) * wu)
    x = 1 + np.exp(-u)                # (k, 2k]
    pts.append(k * x); wts.append(k * np.exp(-u) * wu)
    lq, wl = _gl(np.log(2 * k), np.log(qmax), nq - m_out - 2 * m_in)
    pts.append(np.exp(lq)); wts.append(wl * np.exp(lq))
    return np.concatenate(pts), np.concatenate(wts)


def direct_p13_ratio(k, mu, P, field, f, qmin=1e-5, qmax=200.0, nq=400, nnu=96, nphi=16):
    r"""``R = 3 \int_q [Z3(k,q,-q) - Z3_UV] P(q)`` (so that ``P13 = 2 Z1 P(k) R``)."""
    n = _los(mu)
    uv = _avg13(k, np.array([1e6]), n, field, f, nnu, nphi)[0]
    q, w = _radial_rule(k, qmin, qmax, nq)
    g = _avg13(k, q / k, n, field, f, nnu, nphi)
    return np.sum(w * q ** 2 / (2 * np.pi ** 2) * (g - uv) * P(q))


def direct_p13(k, mu, P, field, f, Z1, **kw):
    """Bare (sigma_0^2-renormalized) selected P13 at one ``(k, mu)``."""
    return 2.0 * Z1 * P(k) * direct_p13_ratio(k, mu, P, field, f, **kw)
