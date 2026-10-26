r"""Cosmology-independent FFTLog matrices for the selection one-loop templates.

The reduced EdS basis of the selection one-loop spectrum consists of 23 P22-type
functions and 5 P13-type functions (equation note E6, E7):

    I_mr(k) = 2 \int_q nu^m q^{r-2} k^{6-r} |k-q|^{-4} P(q) P(|k-q|)
    J_1(k)  = k^2 \int_q (k.q)/(q^2 Q^2) (1-nu^2) P(q)
    J_2(k)  = k^2 \int_q (1-nu^2)/Q^2 P(q)
    J_3(k)  = \int_q [q^2 (1-nu^2)/Q^2 - 2/3] P(q)
    J_4(k)  = \int_q [q^2 nu^2 (1-nu^2)/Q^2 - 2/15] P(q)
    J_5(k)  = \int_q [q^2 nu^4 (1-nu^2)/Q^2 - 2/35] P(q)

with ``nu = k.q/(kq)`` and ``Q = |k-q|``.  Writing ``nu = (k^2+q^2-Q^2)/(2kq)`` reduces
every term to the master integral (E10)

    \int_q q^{-2a} Q^{-2b} = k^{3-2a-2b} I(a, b),
    I(a, b) = Gamma(3/2-a) Gamma(3/2-b) Gamma(a+b-3/2) / (8 pi^{3/2} Gamma(a) Gamma(b) Gamma(3-a-b)),

and with ``P(q) = sum_m c_m q^{-2 nu_m}`` (FolpsD's ``get_cm`` decomposition) each function
becomes ``K^3 vec @ M @ vec`` (P22) or ``K^3 vec @ M`` (P13), exactly as in
``MatrixCalculator``.  The matrices are evaluated in the analytically continued
(dimensional-regularization) sense at the FFTLog tilt; the power-law UV pieces that this
drops are restored explicitly by :mod:`folps.selection_templates` (scheme terms).

Everything here runs on host NumPy/SciPy in complex128, outside any JIT.  Results are
cached in memory and optionally on disk under a key that includes the generator version,
the FFTLog grid, the tilt, the dtype and the checksum of the coefficient data.
"""
from __future__ import annotations

import hashlib
import json
import os
from fractions import Fraction
from math import factorial

import numpy as _np
from scipy import special as _special

from .selection_coefficients import COEFFICIENT_DATA_SHA256, P22_BASIS, GENERATOR_VERSION

__all__ = ["P22_BASIS", "p22_monomials", "j13_monomials", "master_integral",
           "SelectionMatrices", "get_selection_matrices", "fftlog_nu_grid", "MATRIX_FORMAT_VERSION"]

MATRIX_FORMAT_VERSION = "selection-fftlog-1"
_MEMORY_CACHE = {}


def fftlog_nu_grid(nfftlog, b_nu, kmin=1e-7, kmax=100.0):
    """FolpsD's FFTLog exponent grid: ``P(k) = sum_m c_m k^{-2 nu_m}``."""
    jj = _np.arange(nfftlog + 1)
    return -0.5 * (b_nu + (2 * _np.pi * 1j / _np.log(kmax / kmin)) * (jj - nfftlog / 2) * (nfftlog - 1) / nfftlog)


def master_integral(a, b):
    """``I(a, b)``; vanishes identically when ``b`` (or ``a``) is a nonpositive integer
    (scaleless integrals)."""
    g = _special.gamma
    with _np.errstate(all="ignore"):
        out = (g(1.5 - a) * g(1.5 - b) * g(a + b - 1.5)) / (8 * _np.pi ** 1.5 * g(a) * g(b) * g(3.0 - a - b))
    return out


def _nu_power_expansion(m):
    """``nu^m = (k^2 + q^2 - Q^2)^m / (2 k q)^m`` as ``[(coeff, i, j)]`` with
    ``coeff * q^{2i} Q^{2j} k^{2(m-i-j)} / (2kq)^m``."""
    out = []
    for i in range(m + 1):
        for j in range(m + 1 - i):
            c = Fraction(factorial(m), factorial(i) * factorial(j) * factorial(m - i - j)) * (-1) ** j
            out.append((c / 2 ** m, i, j))
    return out


def p22_monomials(m, r):
    """Monomials of ``2 nu^m q^{r-2} Q^{-4}`` (k = 1): list of ``(coeff, alpha, beta)``
    meaning ``coeff * q^alpha Q^beta``."""
    out = []
    for c, i, j in _nu_power_expansion(m):
        out.append((2 * c, r - 2 - m + 2 * i, -4 + 2 * j))
    return out


def j13_monomials(n):
    """Monomials of the J_n integrands (k = 1), constants dropped (scaleless in the
    analytically continued integral): list of ``(coeff, alpha, beta)``."""
    # integrand polynomial in nu: sum_p a_p nu^p, times q^s Q^-2
    if n == 1:
        poly, s = {1: 1, 3: -1}, -1          # (k.q)/(q^2 Q^2) (1-nu^2) k^2 = nu (1-nu^2) q^-1 Q^-2
    elif n == 2:
        poly, s = {0: 1, 2: -1}, 0
    elif n == 3:
        poly, s = {0: 1, 2: -1}, 2
    elif n == 4:
        poly, s = {2: 1, 4: -1}, 2
    elif n == 5:
        poly, s = {4: 1, 6: -1}, 2
    else:
        raise ValueError(n)
    out = []
    for p, ap in poly.items():
        for c, i, j in _nu_power_expansion(p):
            beta = -2 + 2 * j
            if beta >= 0:
                continue  # scaleless: polynomial in k times \int q^n P(q), zero when continued
            out.append((Fraction(ap) * c, s - p + 2 * i, beta))
    return out


def _matrix_from_monomials(monos, nu1, nu2):
    acc = 0.0
    for c, alpha, beta in monos:
        acc = acc + float(c) * master_integral(nu1 - alpha / 2.0, nu2 - beta / 2.0)
    return acc


class SelectionMatrices:
    """Container of the cosmology-independent selection FFTLog matrices.

    Attributes
    ----------
    p22 : complex array ``(23, N+1, N+1)`` in ``P22_BASIS`` order
    j13 : complex array ``(5, N+1)`` for ``J_1..J_5``
    metadata : dict (generator/coefficient versions, grid, tilt, dtype, cache key)
    """

    def __init__(self, p22, j13, metadata):
        self.p22 = p22
        self.j13 = j13
        self.metadata = dict(metadata)

    @property
    def key(self):
        return self.metadata["cache_key"]


def _cache_key(meta):
    payload = json.dumps({k: meta[k] for k in sorted(meta) if k != "cache_key"}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def get_selection_matrices(nfftlog=128, b_nu=-0.1, kmin=1e-7, kmax=100.0, cache_dir=None):
    """Return :class:`SelectionMatrices` (computed once per configuration).

    ``b_nu=-0.1`` is FolpsD's matter tilt; it lies inside the infrared-convergence strip of
    every physical selection integrand, so only power-law ultraviolet pieces are
    analytically continued (and restored by the template scheme terms).
    """
    meta = dict(format=MATRIX_FORMAT_VERSION, generator=GENERATOR_VERSION,
                coefficients_sha256=COEFFICIENT_DATA_SHA256, nfftlog=int(nfftlog), b_nu=float(b_nu),
                kmin=float(kmin), kmax=float(kmax), dtype="complex128",
                p22_basis=[list(x) for x in P22_BASIS])
    key = _cache_key(meta)
    meta["cache_key"] = key
    if key in _MEMORY_CACHE:
        return _MEMORY_CACHE[key]
    path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"selection_matrices_{key}.npz")
        if os.path.exists(path):
            with _np.load(path, allow_pickle=False) as data:
                stored = json.loads(str(data["metadata_json"]))
                if stored != meta:
                    raise ValueError(f"Incompatible selection-matrix cache {path}: metadata mismatch")
                mats = SelectionMatrices(data["p22"], data["j13"], stored)
            _MEMORY_CACHE[key] = mats
            return mats
    nu = fftlog_nu_grid(nfftlog, b_nu, kmin, kmax)
    nu1, nu2 = _np.meshgrid(nu, nu, indexing="ij")  # row index <-> P(q), column <-> P(|k-q|)
    p22 = _np.array([_matrix_from_monomials(p22_monomials(m, r), nu1, nu2) for (m, r) in P22_BASIS])
    j13 = _np.array([_matrix_from_monomials(j13_monomials(n), nu, _np.zeros_like(nu)) for n in range(1, 6)])
    if not (_np.all(_np.isfinite(p22)) and _np.all(_np.isfinite(j13))):
        raise FloatingPointError("non-finite selection FFTLog matrix entries")
    mats = SelectionMatrices(p22, j13, meta)
    if path is not None:
        _np.savez(path, p22=p22, j13=j13, metadata_json=json.dumps(meta))
    _MEMORY_CACHE[key] = mats
    return mats
