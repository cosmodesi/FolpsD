r"""Cosmology-dependent selection templates and their contraction with coefficients.

A :class:`SelectionLoopTable` holds, on FolpsD's output grid ``kTout``:

``k``             (nk,)      wavenumbers [h/Mpc]
``pk``            (nk,)      linear spectrum used for this table [(Mpc/h)^3]
``I22``           (2, 23, nk) ``I_mr(k)`` in ``P22_BASIS`` order [(Mpc/h)^3], evaluated with
                             FFTLog at two tilts: ``I22[0]`` at the matter tilt (used for pairs
                             of soft-singular, UV-suppressed kernels) and ``I22[1]`` at
                             ``P22_TILT_LOCAL`` (all other pairs; there the continued integral
                             drops exactly the zero-lag constant and keeps the order-k^2 terms)
``J13``           (5, nk)    continued ``J_n(k)`` (dimensionless)
``sigma_m1_sq``   scalar     ``sigma_{-1}^2 = \int_q P/q^2 = (1/2 pi^2) \int dq P``  [(Mpc/h)^2]
``R0, S``         scalars    ``\int dq P^2``, ``\int dq q^2 (dP/dq)^2`` (diagnostics of the P22
                             UV order-k^2 content; not needed for the prediction)
``f0``            scalar     growth rate (EdS: scale independent)

plus static ``metadata`` (kernels, FFTLog grid/tilt, input checksum, coefficient and
generator versions, validity domain).  Full-shape and no-wiggle tables are bundled in
:class:`SelectionLoopTables` and must come from the *same* calculator call (identical
input conventions: ``loop(P_full) - loop(P_nw)`` is the wiggle loop).

Renormalized one-loop selection contribution in the declared scheme (equation note):

    L(b) = P22(b) + P13(b)
    P22  = sum_{ab} c_a c_b f^{nf_a+nf_b} sum_{n,j} mu^{2n} T[a,b,n,j] I_j^{(class(a,b))}
    P13  = 2 Z1 P_L { sum_a c13_a f^{nf_a} (1, mu^2, mu^4) M_a . (J + J_UV k^2 sigma_{-1}^2)
                      + B(mu; b, f) k^2 sigma_{-1}^2 }

``P22`` is the full selected P22 minus its zero-lag constant (absorbed in the shot noise);
``B`` is the bare-derivative bridge of the exact redshift-space map; constant
``sigma_0^2`` pieces of P13 (renormalizations of ``b1`` and ``b_eta``) are not included.
"""
from __future__ import annotations

import hashlib

import numpy as _np

from . import selection_coefficients as _sc
from .selection_matrices import get_selection_matrices
from .selection_operators import (cubic_coefficient_vector, quadratic_coefficient_vector,
                                  rest_frame_coefficients)

__all__ = ["SelectionLoopTable", "SelectionLoopTables", "build_selection_tables",
           "selection_loop", "selection_loop_increment", "interp_selection_table",
           "selection_loop_cross", "selection_loop_cross_increment",
           "TEMPLATE_FORMAT_VERSION", "P22_TILT_LOCAL", "p22_uv_k2_content"]

TEMPLATE_FORMAT_VERSION = "selection-template-2"
# FFTLog tilt for P22 pairs involving at least one soft-safe (local) kernel.  It must lie in
# (-3/2, -1/2): there the continued integral drops the zero-lag constant and keeps the
# order-k^2 piece, and the soft singularity (at most 1/q) is integrable.  Within the strip,
# the periodic images of the FFTLog representation leak into the k^2 and constant pieces
# with weights ~ (kmax/kmin)^(1+2b) and ~ (kmax/kmin)^-(3+2b) respectively (measured:
# k^2 error -4151, -64, -1.0 at b = -0.6, -0.7, -0.8 for the K^2 x K^2 pair), so the tilt
# sits near the middle; b = -1 itself hits a spurious pole of the reduced basis (Re nu = 1/2).
P22_TILT_LOCAL = -0.9


def _backend_np():
    from .folps import np as bnp
    return bnp


class SelectionLoopTable:
    """Named EdS selection template table (see module docstring for axes and units)."""
    _array_fields = ("k", "pk", "I22", "J13", "sigma_m1_sq", "R0", "S", "f0")

    def __init__(self, k, pk, I22, J13, sigma_m1_sq, R0, S, f0, metadata):
        self.k, self.pk, self.I22, self.J13 = k, pk, I22, J13
        self.sigma_m1_sq, self.R0, self.S, self.f0 = sigma_m1_sq, R0, S, f0
        self.metadata = dict(metadata)

    def arrays(self):
        return tuple(getattr(self, n) for n in self._array_fields)

    def __repr__(self):
        m = self.metadata
        return (f"SelectionLoopTable(role={m.get('role')!r}, nk={len(_np.asarray(self.k))}, "
                f"I22={tuple(_np.shape(self.I22))}, J13={tuple(_np.shape(self.J13))}, "
                f"kernels={m.get('kernels')!r}, key={m.get('template_key')!r})")


class SelectionLoopTables:
    """Pair of full-shape and no-wiggle :class:`SelectionLoopTable` objects."""

    def __init__(self, full, nowiggle):
        if full.metadata.get("pair_id") != nowiggle.metadata.get("pair_id"):
            raise ValueError("full and no-wiggle selection tables come from different calculator calls")
        self.full, self.nowiggle = full, nowiggle

    def __iter__(self):
        return iter((self.full, self.nowiggle))

    def __repr__(self):
        return f"SelectionLoopTables(full={self.full!r}, nowiggle={self.nowiggle!r})"


def _register_pytrees():
    import os
    import sys
    if os.environ.get("FOLPS_BACKEND", "numpy").lower() != "jax" and "jax" not in sys.modules:
        return  # NumPy route: never import JAX
    try:
        import jax
    except Exception:  # pragma: no cover - JAX optional
        return

    def flat(t):
        return t.arrays(), tuple(sorted((k, str(v)) for k, v in t.metadata.items()))

    def unflat(aux, children):
        return SelectionLoopTable(*children, metadata=dict(aux))
    jax.tree_util.register_pytree_node(SelectionLoopTable, flat, unflat)
    jax.tree_util.register_pytree_node(
        SelectionLoopTables, lambda t: ((t.full, t.nowiggle), None),
        lambda aux, ch: SelectionLoopTables.__new__(SelectionLoopTables)._set(*ch))


def _set(self, full, nowiggle):
    self.full, self.nowiggle = full, nowiggle
    return self


SelectionLoopTables._set = _set
_register_pytrees()


def _hash_arrays(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        try:
            h.update(_np.ascontiguousarray(_np.asarray(a, dtype=float)).tobytes())
        except Exception:  # traced values (JAX): no content hash
            return "traced"
    return h.hexdigest()[:16]


def _concrete_float(x):
    """``float(x)``, or None for traced values (JAX tracing, e.g. emulator fitting): metadata only."""
    try:
        return float(_np.asarray(x, dtype=float))
    except Exception:
        return None


def build_selection_tables(nonlinear, matrices=None, cache_dir=None):
    """Build :class:`SelectionLoopTables` from a ``NonLinearPowerSpectrumCalculator``.

    ``nonlinear.calculate_loop_table(...)`` must have been called (it fixes the extrapolated
    input spectra, the no-wiggle spectrum and the FFTLog coefficients that are reused here).
    Only ``kernels='EdS'`` is supported: the selection kernels are derived for EdS growth.
    """
    bnp = _backend_np()
    if getattr(nonlinear, "kernels", None) != "EdS":
        raise NotImplementedError(
            "Full one-loop selection templates are derived for EdS kernels with scale-independent "
            "growth only; got kernels=%r. Use kernels='EdS' (the fk selection kernels require a "
            "separate derivation; see docs/selection_1loop_equations.md)." % getattr(nonlinear, "kernels", None))
    for attr in ("cmT", "cmT_NW", "precvec", "inputpkT", "inputpkT_NW", "pk_l", "pk_l_NW", "f0"):
        if not hasattr(nonlinear, attr):
            raise ValueError("call nonlinear.calculate_loop_table(...) before building selection tables")
    from .folps import get_cm, simpson
    grid = dict(nfftlog=nonlinear.nfftlog, kmin=nonlinear.kmin, kmax=nonlinear.kmax, cache_dir=cache_dir)
    if matrices is None:
        matrices = (get_selection_matrices(b_nu=nonlinear.b_nu, **grid),
                    get_selection_matrices(b_nu=P22_TILT_LOCAL, **grid))
    mats_sing, mats_loc = matrices
    for mats, tilt in ((mats_sing, nonlinear.b_nu), (mats_loc, P22_TILT_LOCAL)):
        mm = mats.metadata
        if (mm["nfftlog"], mm["b_nu"], mm["kmin"], mm["kmax"]) != (int(nonlinear.nfftlog), float(tilt),
                                                                  float(nonlinear.kmin), float(nonlinear.kmax)):
            raise ValueError("selection matrices were built for a different FFTLog grid/tilt than required")
    p22s = bnp.asarray(mats_sing.p22)
    p22l = bnp.asarray(mats_loc.p22)
    j13 = bnp.asarray(mats_sing.j13)
    K = nonlinear.K
    pair_id = _hash_arrays(nonlinear.inputpkT[1], nonlinear.inputpkT_NW[1], K)
    jj = _np.arange(nonlinear.nfftlog + 1)
    ietam = (2 * _np.pi * 1j / _np.log(nonlinear.kmax / nonlinear.kmin)) * (jj - nonlinear.nfftlog / 2) \
        * (nonlinear.nfftlog - 1) / nonlinear.nfftlog
    precvec_loc = K[:, None] ** bnp.asarray(P22_TILT_LOCAL + ietam)

    def one(cm, inputpk, pk_out, role):
        vec = cm * nonlinear.precvec
        cm_loc = get_cm(nonlinear.kmin, nonlinear.kmax, nonlinear.nfftlog, P22_TILT_LOCAL, inputpk)
        vecl = cm_loc * precvec_loc
        I22 = bnp.stack([
            bnp.stack([K ** 3 * bnp.sum((vec @ p22s[j]) * vec, axis=-1).real for j in range(p22s.shape[0])]),
            bnp.stack([K ** 3 * bnp.sum((vecl @ p22l[j]) * vecl, axis=-1).real for j in range(p22l.shape[0])])])
        J13 = bnp.stack([K ** 3 * (vec @ j13[n]).real for n in range(j13.shape[0])])
        q, P = inputpk[0], inputpk[1]
        sigma_m1_sq = simpson(P, x=q) / (2 * _np.pi ** 2)
        dP = bnp.gradient(P, q)
        R0 = simpson(P ** 2, x=q)
        S = simpson(q ** 2 * dP ** 2, x=q)
        meta = dict(role=role, kernels="EdS", template_format=TEMPLATE_FORMAT_VERSION,
                    generator=_sc.GENERATOR_VERSION, coefficients_sha256=_sc.COEFFICIENT_DATA_SHA256,
                    matrix_keys=(mats_sing.key, mats_loc.key), nfftlog=int(nonlinear.nfftlog),
                    b_nu=float(nonlinear.b_nu), p22_tilt_local=P22_TILT_LOCAL,
                    input_sha=_hash_arrays(q, P), pair_id=pair_id,
                    k_min=_concrete_float(K[0]), k_max=_concrete_float(K[-1]))
        meta["template_key"] = hashlib.sha256(repr(sorted(meta.items())).encode()).hexdigest()[:16]
        return SelectionLoopTable(K, pk_out, I22, J13, sigma_m1_sq, R0, S, nonlinear.f0, meta)
    full = one(nonlinear.cmT, nonlinear.inputpkT, nonlinear.pk_l, "full")
    nw = one(nonlinear.cmT_NW, nonlinear.inputpkT_NW, nonlinear.pk_l_NW, "nowiggle")
    return SelectionLoopTables(full, nw)


# ------------------------------------------------------------------------------ evaluation
def interp_selection_table(k, table):
    """Interpolate a table to ``k`` (any shape) with the same scheme FolpsD uses for its
    legacy tables (SciPy cubic spline for NumPy, backend ``interp`` for JAX)."""
    from .folps import backend, interp
    bnp = _backend_np()
    n22 = table.I22.shape[1]
    rows = [table.pk] + [table.I22[c, j] for c in range(2) for j in range(n22)] + \
        [table.J13[n] for n in range(table.J13.shape[0])]
    if backend == "jax":
        out = [interp(k, table.k, r) for r in rows]
    else:
        from scipy.interpolate import CubicSpline
        out = [CubicSpline(_np.asarray(table.k), _np.asarray(r), extrapolate=True)(k) for r in rows]
    pk = out[0]
    I22 = bnp.stack([bnp.stack(out[1:1 + n22]), bnp.stack(out[1 + n22:1 + 2 * n22])])
    J13 = bnp.stack(out[1 + 2 * n22:])
    return pk, I22, J13


def _p22_weights(c2_outer, f):
    """``W[class, n, j] = sum_{ab in class} C_ab f^{nf_a+nf_b} T[a,b,n,j]`` (class 0: both
    kernels soft-singular; class 1: all other pairs) for a 10x10 coefficient matrix."""
    bnp = _backend_np()
    nf = _sc.QUADRATIC_NF
    fpow = f ** bnp.asarray(_np.add.outer(nf, nf), dtype=float)
    mask = bnp.asarray(_sc.P22_PAIR_SINGULAR.astype(float))
    T = bnp.asarray(_sc.P22_T)
    Cf = c2_outer * fpow
    return bnp.stack([bnp.einsum("ab,abnj->nj", Cf * mask, T), bnp.einsum("ab,abnj->nj", Cf * (1 - mask), T)])


def _p22_contract(W, mup, I22):
    bnp = _backend_np()
    return bnp.einsum("cnj,n...,cj...->...", W, mup, I22)


def _p13_weights(c13, f):
    """``V[n, i] = sum_a c13_a f^{nf_a} M[a, n, i]``."""
    bnp = _backend_np()
    fp = bnp.stack([c13[a] * f ** int(_sc.CUBIC_NF[a]) for a in range(16)])
    return bnp.einsum("a,ani->ni", fp, bnp.asarray(_sc.CUBIC_M))


def _bridge(rest, f, mu, include_matter=True):
    """Bare ``k^2 sigma_{-1}^2`` coefficient ``B(mu)`` of ``3 int Z3 P`` (exact map); affine
    in the rest-frame coefficients ``rest`` (the constant part is the matter term)."""
    bnp = _backend_np()
    shape = _sc.BRIDGE_D["matter"].shape
    fpow = bnp.stack([f ** p for p in range(shape[0])])
    mupow = bnp.stack([mu ** (2 * n) for n in range(shape[1])])

    def poly(arr):
        return bnp.einsum("p,pn,n...->...", fpow, bnp.asarray(arr), mupow)
    out = poly(_sc.BRIDGE_D["matter"]) if include_matter else 0.0
    for key, val in zip(_sc.REST_KEYS[:-1], rest):
        out = out + val * poly(_sc.BRIDGE_D[key])
    return out


def _mu_poly(W, mu):
    bnp = _backend_np()
    return bnp.stack([mu ** (2 * n) for n in range(W.shape[0])])


def _components(pk, I22, J13, tab, c, c0, k, mu):
    """Return (P22, P13) of coefficient set ``c`` and, if ``c0`` is given, the exact
    differences (P22(c) - P22(c0), P13(c) - P13(c0)) computed without cancellation."""
    bnp = _backend_np()
    f = tab.f0
    k2 = k ** 2
    Jt = J13 + bnp.asarray(_sc.J_UV).reshape((-1,) + (1,) * k.ndim) * (k2 * tab.sigma_m1_sq)[None]
    c2 = bnp.stack(quadratic_coefficient_vector(c))
    c13 = cubic_coefficient_vector(c)
    rest = rest_frame_coefficients(c)
    Z1 = c.b1 - c.b_eta * f * mu ** 2
    mup = bnp.stack([mu ** (2 * n) for n in range(5)])
    if c0 is None:
        C = bnp.outer(c2, c2)
        P22 = _p22_contract(_p22_weights(C, f), mup, I22)
        V = _p13_weights(c13, f)
        R = bnp.einsum("ni,n...,i...->...", V, mup[:3], Jt) + k2 * tab.sigma_m1_sq * _bridge(rest, f, mu)
        return P22, 2.0 * Z1 * pk * R
    c20 = bnp.stack(quadratic_coefficient_vector(c0))
    dc2 = c2 - c20
    dC = bnp.outer(dc2, c2) + bnp.outer(c20, dc2)           # c c^T - c0 c0^T
    dP22 = _p22_contract(_p22_weights(dC, f), mup, I22)
    c130 = cubic_coefficient_vector(c0)
    dc13 = [a - b for a, b in zip(c13, c130)]
    rest0 = rest_frame_coefficients(c0)
    drest = [a - b for a, b in zip(rest, rest0)]
    V0 = _p13_weights(c130, f)
    R0 = bnp.einsum("ni,n...,i...->...", V0, mup[:3], Jt) + k2 * tab.sigma_m1_sq * _bridge(rest0, f, mu)
    dV = _p13_weights(dc13, f)
    # the bridge is affine in the rest-frame coefficients: difference = linear part only
    dB = _bridge(drest, f, mu, include_matter=False)
    dR = bnp.einsum("ni,n...,i...->...", dV, mup[:3], Jt) + k2 * tab.sigma_m1_sq * dB
    Z10 = c0.b1 - c0.b_eta * f * mu ** 2
    dP13 = 2.0 * pk * ((Z1 - Z10) * R0 + Z1 * dR)
    return dP22, dP13


def _p13_response(c, f, mu, k2, Jt, tab, mup, c0=None):
    """``R = 3 int Z3 P_L`` of one tracer (single contraction, bare exact-map scheme), or with
    ``c0`` the exact difference ``R(c) - R(c0)``: ``c13`` is linear in the coefficients, the
    bridge affine in the rest-frame ones (its matter part cancels in the difference)."""
    bnp = _backend_np()
    c13 = cubic_coefficient_vector(c)
    rest = rest_frame_coefficients(c)
    if c0 is not None:
        c13 = [a - b for a, b in zip(c13, cubic_coefficient_vector(c0))]
        rest = [a - b for a, b in zip(rest, rest_frame_coefficients(c0))]
    V = _p13_weights(c13, f)
    return (bnp.einsum("ni,n...,i...->...", V, mup[:3], Jt)
            + k2 * tab.sigma_m1_sq * _bridge(rest, f, mu, include_matter=c0 is None))


def _components_cross(pk, I22, J13, tab, cA, cB, cA0=None, cB0=None, k=None, mu=None):
    r"""Two-tracer (A x B) one-loop selection contraction in the same scheme as :func:`_components`.

    ``P22_AB = 2 int Z2_A Z2_B P P`` is the bilinear form of the symmetric tensor ``T`` with the
    two coefficient vectors (symmetrized, ``(c_A c_B^T + c_B c_A^T) / 2``), and
    ``P13_AB = P_L (Z1_A R_B + Z1_B R_A)`` with ``R_X = 3 int Z3_X(k, q, -q) P_L(q)`` the single
    contraction of tracer X (the auto spectrum ``2 Z1 P_L R`` is the case A = B).  The bare
    ``k^2 sigma_{-1}^2`` bridge enters each ``R_X`` with that tracer's own rest-frame
    coefficients; its matter part gives ``-k^2 sigma_psi^2 [1 + f mu^2 (2 + f)] Z1_A Z1_B P_L``.

    With ``cA0, cB0`` given, returns the exact increments
    ``(P22_AB(cA, cB) - P22_AB(cA0, cB0), P13_AB(cA, cB) - P13_AB(cA0, cB0))``, formed from
    coefficient differences (no cancellation; identically zero when ``cA == cA0`` and ``cB == cB0``).
    """
    bnp = _backend_np()
    f = tab.f0
    k2 = k ** 2
    Jt = J13 + bnp.asarray(_sc.J_UV).reshape((-1,) + (1,) * k.ndim) * (k2 * tab.sigma_m1_sq)[None]
    mup = bnp.stack([mu ** (2 * n) for n in range(5)])
    c2A = bnp.stack(quadratic_coefficient_vector(cA))
    c2B = bnp.stack(quadratic_coefficient_vector(cB))
    Z1A = cA.b1 - cA.b_eta * f * mu ** 2
    Z1B = cB.b1 - cB.b_eta * f * mu ** 2

    def sym(x, y):
        return 0.5 * (bnp.outer(x, y) + bnp.outer(y, x))

    if cA0 is None:
        P22 = _p22_contract(_p22_weights(sym(c2A, c2B), f), mup, I22)
        RA = _p13_response(cA, f, mu, k2, Jt, tab, mup)
        RB = _p13_response(cB, f, mu, k2, Jt, tab, mup)
        return P22, pk * (Z1A * RB + Z1B * RA)
    c2A0 = bnp.stack(quadratic_coefficient_vector(cA0))
    c2B0 = bnp.stack(quadratic_coefficient_vector(cB0))
    # c_A c_B^T - c_A0 c_B0^T = dc_A c_B^T + c_A0 dc_B^T, then symmetrized
    dC = sym(c2A - c2A0, c2B) + sym(c2A0, c2B - c2B0)
    dP22 = _p22_contract(_p22_weights(dC, f), mup, I22)
    RA0 = _p13_response(cA0, f, mu, k2, Jt, tab, mup)
    RB0 = _p13_response(cB0, f, mu, k2, Jt, tab, mup)
    dRA = _p13_response(cA, f, mu, k2, Jt, tab, mup, c0=cA0)
    dRB = _p13_response(cB, f, mu, k2, Jt, tab, mup, c0=cB0)
    Z1A0 = cA0.b1 - cA0.b_eta * f * mu ** 2
    Z1B0 = cB0.b1 - cB0.b_eta * f * mu ** 2
    # Z1_A R_B - Z1_A0 R_B0 = (Z1_A - Z1_A0) R_B0 + Z1_A dR_B, and A <-> B
    dP13 = pk * ((Z1A - Z1A0) * RB0 + Z1A * dRB + (Z1B - Z1B0) * RA0 + Z1B * dRA)
    return dP22, dP13


def selection_loop_cross(k, mu, table, cA, cB, interpolated=None):
    """Complete renormalized one-loop selected cross contribution ``(P22_AB, P13_AB)``."""
    bnp = _backend_np()
    k = bnp.asarray(k)
    mu = bnp.asarray(mu)
    pk, I22, J13 = interpolated if interpolated is not None else interp_selection_table(k, table)
    return _components_cross(pk, I22, J13, table, cA, cB, k=k, mu=mu)


def selection_loop_cross_increment(k, mu, table, cA, cB, cA0, cB0, interpolated=None):
    """Selected-minus-unselected cross increments; exactly zero when ``cA == cA0`` and ``cB == cB0``."""
    bnp = _backend_np()
    k = bnp.asarray(k)
    mu = bnp.asarray(mu)
    pk, I22, J13 = interpolated if interpolated is not None else interp_selection_table(k, table)
    return _components_cross(pk, I22, J13, table, cA, cB, cA0, cB0, k=k, mu=mu)


def selection_loop(k, mu, table, c, interpolated=None):
    """Complete renormalized one-loop selection contribution ``(P22, P13)`` for mapped DJS
    coefficients ``c`` (:class:`DJSCoefficients`) at ``(k, mu)`` from one table."""
    bnp = _backend_np()
    k = bnp.asarray(k)
    mu = bnp.asarray(mu)
    pk, I22, J13 = interpolated if interpolated is not None else interp_selection_table(k, table)
    return _components(pk, I22, J13, table, c, None, k, mu)


def selection_loop_increment(k, mu, table, c, c0, interpolated=None):
    """``(P22(c) - P22(c0), P13(c) - P13(c0))``; exactly zero when ``c == c0``."""
    bnp = _backend_np()
    k = bnp.asarray(k)
    mu = bnp.asarray(mu)
    pk, I22, J13 = interpolated if interpolated is not None else interp_selection_table(k, table)
    return _components(pk, I22, J13, table, c, c0, k, mu)


def p22_uv_k2_content(c, f, table):
    """Diagnostic: order-k^2 large-q (UV) content of the selected P22,
    ``k^2 sum_n a_n mu^{2n}``, returned as the array ``a_n`` [(Mpc/h)^5].  Its ``mu^4..mu^8``
    entries quantify non-analytic-in-k, UV-sensitive pieces (see the validation report)."""
    bnp = _backend_np()
    c2 = bnp.stack(quadratic_coefficient_vector(c))
    W = _p22_weights(bnp.outer(c2, c2), f).sum(axis=0)
    U = bnp.asarray(_sc.P22_UV)
    uv = (U[:, 0] * table.R0 + U[:, 1] * table.S) / (2 * _np.pi ** 2)
    return bnp.einsum("nj,j->n", W, uv)
