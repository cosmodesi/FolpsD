r"""Loader for the generated EdS selection coefficient data.

The data module ``folps/_selection_coefficients_data.py`` is produced by
``scripts/generate_selection_coefficients.py`` and stores exact rationals.  This module
converts them to float64 host-NumPy arrays with documented axes:

``P22_T[a, b, n, j]``   (10, 10, 5, 23)  symmetric in ``a, b`` (Galilean-safe kernel basis,
                        see ``QUADRATIC_KERNELS``): coefficient of
                        ``mu^{2n}`` times basis function ``j`` (``P22_BASIS[j] = (m, r)``)
                        in the azimuthally averaged product ``S_a S_b`` of growth-stripped
                        quadratic kernels; ``P22 = sum_ab c_a c_b f^{nf_a+nf_b} sum_nj
                        mu^{2n} P22_T[a,b,n,j] I_j(k)``.
``P22_UV[j, 2]``        UV order-``k^2`` scheme coefficients ``(a_R0, a_S)`` of ``I_j``.
``CUBIC_M[a, n, i]``    (16, 3, 5) growth-stripped reduced cubic matrices (single 13 contraction).
``CUBIC_NF[a]``         growth powers of the sixteen reduced operators.
``J_UV[i]``             (5,) ``J_i(true) - J_i(continued) = J_UV[i] k^2 sigma_{-1}^2``.
``BRIDGE_D[key][p, n]`` (5, 5) bare ``k^2 sigma_{-1}^2`` coefficient of the single 13
                        contraction of the exact redshift-space map, per rest-frame
                        coefficient ``key`` (and ``'matter'``), as ``f^p mu^{2n}``.

``COEFFICIENT_DATA_SHA256`` is the SHA-256 of the data file and enters every cache key.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as _np

from . import _selection_coefficients_data as _data

__all__ = ["GENERATOR_VERSION", "COEFFICIENT_DATA_SHA256", "P22_BASIS", "QUADRATIC_NF", "P22_T",
           "QUADRATIC_SINGULAR", "P22_PAIR_SINGULAR",
           "P22_UV", "CUBIC_M", "CUBIC_NF", "CUBIC_D", "CUBIC_C", "J_UV", "BRIDGE_D", "BRIDGE_C",
           "REST_KEYS", "CUBIC_OPERATORS", "QUADRATIC_KERNELS", "as_fraction_table"]

GENERATOR_VERSION = _data.GENERATOR_VERSION
COEFFICIENT_DATA_SHA256 = hashlib.sha256(Path(_data.__file__).read_bytes()).hexdigest()
P22_BASIS = tuple(tuple(x) for x in _data.P22_BASIS)
QUADRATIC_KERNELS = tuple(_data.QUADRATIC_KERNELS)
QUADRATIC_NF = _np.array(_data.QUADRATIC_NF)
QUADRATIC_SINGULAR = tuple(_data.QUADRATIC_SINGULAR)
# pairs (a, b) whose kernels are both soft-singular and UV-suppressed (evaluated at the matter tilt)
P22_PAIR_SINGULAR = _np.array([[a in QUADRATIC_SINGULAR and b in QUADRATIC_SINGULAR for b in range(10)]
                               for a in range(10)])
CUBIC_OPERATORS = tuple(_data.CUBIC_OPERATORS)
CUBIC_NF = _np.array(_data.CUBIC_NF)
REST_KEYS = tuple(_data.REST_KEYS)


def _val(pair):
    return pair[0] / pair[1]


def _array(nested):
    if isinstance(nested, tuple) and len(nested) == 2 and all(isinstance(x, int) for x in nested):
        return _val(nested)
    return [_array(x) for x in nested]


def _p22_tensor():
    T = _np.zeros((10, 10, 5, len(P22_BASIS)))
    index = {mr: j for j, mr in enumerate(P22_BASIS)}
    for (a, b, n, m, r), pair in _data.P22_TENSOR.items():
        v = _val(pair)
        T[a, b, n, index[(m, r)]] = v
        T[b, a, n, index[(m, r)]] = v
    return T


P22_T = _p22_tensor()
P22_UV = _np.array([[_val(_data.P22_UV_K2[mr][0]), _val(_data.P22_UV_K2[mr][1])] for mr in P22_BASIS])
J_UV = _np.array([_val(x) for x in _data.J_UV])
CUBIC_M = _np.array(_array(_data.CUBIC_M), dtype=float)
CUBIC_D = _np.array(_array(_data.CUBIC_D), dtype=float)
CUBIC_C = _np.array(_array(_data.CUBIC_C), dtype=float)
BRIDGE_D = {k: _np.array(_array(v), dtype=float) for k, v in _data.BRIDGE_D.items()}
BRIDGE_C = {k: _np.array(_array(v), dtype=float) for k, v in _data.BRIDGE_C.items()}


def as_fraction_table(name):
    """Exact rational form of one of the stored arrays (for documentation and tests)."""
    from fractions import Fraction

    def conv(x):
        if isinstance(x, tuple) and len(x) == 2 and all(isinstance(y, int) for y in x):
            return Fraction(*x)
        return [conv(y) for y in x]
    return conv(getattr(_data, name))
