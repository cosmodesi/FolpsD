r"""Explicit EdS perturbation-theory kernels for line-of-sight (LOS) selection.

This module is a *validation oracle*.  It evaluates, at arbitrary numerical momentum
configurations, the symmetrized kernels of every field that enters the one-loop galaxy
power spectrum with LOS-dependent selection.  It is deliberately generic and slow:

* EdS density / velocity-divergence kernels ``F_n, G_n`` from the standard recursion
  (unsymmetrized recursion, then a normalized average over permutations);
* the convective tidal hierarchy ``Pi^[a]`` from its time-derivative definition
  (``T^[a](n)`` recursion of the equation note, E3);
* normalized symmetrized products of fields, LOS projections, matrix products,
  first-order displacement ``s`` and LOS velocity ``u_par`` advection terms;
* the DJS compact mapped field (E4) and, independently, the exact redshift-space
  coordinate map of a rest-frame selected density (E4, "alternative exact map").

It uses host NumPy only (never JAX), is not used by the production FFTLog templates, and
must not be called inside likelihood evaluations.  Conventions (equation note E1):

* ``F(k) = \int d^3x e^{-ik.x} F(x)``; no external ``1/n!`` in the PT expansion;
* ``vartheta = -div v / (H f)`` has kernels ``G_n``; ``u = v / H`` so that
  ``u_par^(n)(K) = i f K_par / K^2 G_n``, ``eta^(n) = d_par u_par = -f mu_K^2 G_n``;
* ``s^(1)(K) = i K / K^2 delta_L`` (Lagrangian displacement, first order only);
* ``Sym`` is the normalized average over distinct assignments of momenta to factors.

Collapsed configurations: when a block of two or more momenta sums exactly to zero (the
``(q, -q)`` pair of a 13 contraction or the zero-lag ``(q, -q)`` of P22), every term in
which a zero-momentum *velocity* or *density* sub-kernel multiplies an inverse power of
that momentum is evaluated by its continuous limit, which is zero after the normalized
permutation average (see ``docs/selection_1loop_equations.md``, section on collapsed
limits).  No numerical epsilon is introduced.
"""
from __future__ import annotations

import itertools
import math
from functools import reduce

import numpy as np

__all__ = [
    "Evaluator", "Field", "density", "theta", "eta", "u_par", "displacement_s1",
    "pi_tensor", "tidal_K", "projector_R", "product", "dot_vector", "par", "matmul_par",
    "trace", "trace_product", "grad_par", "grad", "scale", "add", "linear_combination",
    "mapped_djs_field", "exact_map_field", "rest_frame_density",
    "reduced_cubic_operator_fields", "quadratic_table_kernels", "F2_closed_form", "G2_closed_form",
    "DJS_QUADRATIC_ORDER", "DJS_CUBIC_REDUCED_ORDER",
]

_ZERO_REL_TOL = 1.0e-24


def _dot(a, b):
    return np.sum(a * b, axis=-1)


class Evaluator:
    """Evaluate symmetrized field kernels at a fixed list of momenta.

    Parameters
    ----------
    ks : sequence of arrays, each of shape ``(..., 3)``
        Momenta ``k_1, ..., k_n`` (broadcast against each other).
    n_los : array of shape ``(3,)`` or ``(..., 3)``
        Unit line-of-sight vector.
    f : float
        Linear growth rate multiplying velocities (EdS time dependence is kept in the
        kernels; ``f`` only normalizes ``u`` and ``eta``).
    """

    def __init__(self, ks, n_los, f=1.0):
        self.ks = [np.asarray(k, dtype=float) for k in ks]
        shape = np.broadcast_shapes(*(k.shape for k in self.ks), np.shape(n_los))
        self.ks = [np.broadcast_to(k, shape) for k in self.ks]
        self.n = np.broadcast_to(np.asarray(n_los, dtype=float), shape)
        self.f = float(f)
        scale = sum(_dot(k, k) for k in self.ks)
        self._tol = _ZERO_REL_TOL * scale
        self._cache = {}

    # ----------------------------------------------------------------------------- helpers
    def K(self, idx):
        return reduce(lambda a, b: a + b, (self.ks[i] for i in idx))

    def K2(self, idx):
        K = self.K(idx)
        return _dot(K, K)

    def is_zero(self, idx):
        """Boolean mask: the block ``idx`` (>= 2 momenta) has exactly zero total momentum."""
        if len(idx) < 2:
            return np.zeros(self.K2(idx).shape, dtype=bool)
        return self.K2(idx) <= self._tol

    def safe_inv(self, x, mask):
        return np.where(mask, 0.0, 1.0 / np.where(mask, 1.0, x))

    def unit_par(self, idx):
        """``mu_K = K.n / |K|`` (set to 0 on zero-momentum blocks, where it only ever
        multiplies a vanishing kernel)."""
        K = self.K(idx)
        K2 = _dot(K, K)
        z = self.is_zero(idx)
        return np.where(z, 0.0, _dot(K, self.n) * self.safe_inv(np.sqrt(np.where(z, 1.0, K2)), z))

    def cached(self, key, func):
        if key not in self._cache:
            self._cache[key] = func()
        return self._cache[key]

    # ------------------------------------------------------------------- F_n, G_n recursion
    def FG_unsym(self, order):
        """Unsymmetrized EdS (F_n, G_n) for an ordered tuple of momentum indices."""
        order = tuple(order)
        key = ("FGu", order)
        if key in self._cache:
            return self._cache[key]
        n = len(order)
        if n == 1:
            one = np.ones(self.K2(order).shape)
            res = (one, one)
        else:
            Fn = 0.0
            Gn = 0.0
            for m in range(1, n):
                left, right = order[:m], order[m:]
                p, r = self.K(left), self.K(right)
                p2, r2 = _dot(p, p), _dot(r, r)
                zl, zr = self.is_zero(left), self.is_zero(right)
                dead = zl | zr  # continuous limit: these terms vanish after symmetrization
                pr = p + r
                alpha = _dot(pr, p) * self.safe_inv(p2, dead)
                beta = _dot(pr, pr) * _dot(p, r) * self.safe_inv(2.0 * p2 * r2, dead)
                Gm = self.FG_unsym(left)[1]
                Fr, Gr = self.FG_unsym(right)
                pref = Gm / ((2 * n + 3) * (n - 1))
                Fn = Fn + np.where(dead, 0.0, pref * ((2 * n + 1) * alpha * Fr + 2 * beta * Gr))
                Gn = Gn + np.where(dead, 0.0, pref * (3 * alpha * Fr + 2 * n * beta * Gr))
            res = (Fn, Gn)
        self._cache[key] = res
        return res

    def FG_sym(self, idx):
        idx = tuple(sorted(idx))
        key = ("FGs", idx)
        if key not in self._cache:
            perms = list(itertools.permutations(idx))
            F = sum(self.FG_unsym(p)[0] for p in perms) / len(perms)
            G = sum(self.FG_unsym(p)[1] for p in perms) / len(perms)
            z = self.is_zero(idx)
            # exact zero-total-momentum blocks: symmetrized EdS kernels vanish (momentum conservation)
            self._cache[key] = (np.where(z, 0.0, F), np.where(z, 0.0, G))
        return self._cache[key]

    # -------------------------------------------------------------------------- Pi hierarchy
    def R(self, idx):
        K = self.K(idx)
        K2 = _dot(K, K)
        z = self.is_zero(idx)
        inv = self.safe_inv(K2, z)
        return K[..., :, None] * K[..., None, :] * inv[..., None, None]

    def T(self, a, idx):
        """Symmetrized kernel of ``Pi^[a]_ij`` at order ``len(idx)`` (EdS recursion E3)."""
        idx = tuple(sorted(idx))
        n = len(idx)
        key = ("T", a, idx)
        if key in self._cache:
            return self._cache[key]
        shape = self.K2(idx).shape + (3, 3)
        if n < a:
            res = np.zeros(shape)
        elif a == 1:
            res = self.R(idx) * self.FG_sym(idx)[0][..., None, None]
        else:
            res = (n - a + 1) * self.T(a - 1, idx)
            for m in range(1, n):
                acc = np.zeros(shape)
                subsets = list(itertools.combinations(idx, m))
                for vel in subsets:
                    ten = tuple(i for i in idx if i not in vel)
                    if len(ten) < a - 1:
                        continue
                    p, r = self.K(vel), self.K(ten)
                    zp = self.is_zero(vel)
                    w = _dot(p, r) * self.safe_inv(_dot(p, p), zp) * self.FG_sym(vel)[1]
                    acc = acc + np.where(zp, 0.0, w)[..., None, None] * self.T(a - 1, ten)
                res = res - acc / len(subsets)
            res = res / math.factorial(a - 1)
        self._cache[key] = res
        return res


# ================================================================================== fields
class Field:
    """A perturbative field: ``kernel(ev, idx)`` returns its symmetrized kernel at order
    ``len(idx)`` (an array with trailing component shape ``()``, ``(3,)`` or ``(3, 3)``).
    ``min_order`` is the lowest nonvanishing order; ``name`` is used as a cache key."""

    _counter = itertools.count()

    def __init__(self, kernel, rank=0, min_order=1, name=None):
        self._kernel = kernel
        self.rank = rank
        self.min_order = min_order
        self.uid = next(Field._counter)  # cache key: unique per field object
        self.name = name if name is not None else f"field{self.uid}"

    def __call__(self, ev, idx):
        idx = tuple(sorted(idx))
        if len(idx) < self.min_order:
            return np.zeros(ev.K2(idx).shape + (3,) * self.rank, dtype=complex)
        return ev.cached(("field", self.uid, idx), lambda: self._kernel(ev, idx))

    def at(self, ev, n):
        """Kernel at order ``n`` using momenta ``0..n-1`` of the evaluator."""
        return self(ev, tuple(range(n)))


def density():
    return Field(lambda ev, idx: ev.FG_sym(idx)[0].astype(complex), name="delta")


def theta():
    return Field(lambda ev, idx: ev.FG_sym(idx)[1].astype(complex), name="vartheta")


def eta():
    """``eta = d_par u_par``: kernel ``-f mu_K^2 G_n``."""
    return Field(lambda ev, idx: (-ev.f * ev.unit_par(idx) ** 2 * ev.FG_sym(idx)[1]).astype(complex),
                 name="eta")


def u_par():
    """LOS velocity ``u_par = v_par / H``: kernel ``i f K_par / K^2 G_n``."""
    def k(ev, idx):
        z = ev.is_zero(idx)
        Kpar = _dot(ev.K(idx), ev.n)
        return 1j * ev.f * Kpar * ev.safe_inv(ev.K2(idx), z) * ev.FG_sym(idx)[1]
    return Field(k, name="u_par")


def displacement_s1():
    """First-order Lagrangian displacement ``s^(1)_i = i K_i / K^2 delta_L`` (vector field
    whose only nonvanishing order is the first)."""
    def k(ev, idx):
        if len(idx) != 1:
            return np.zeros(ev.K2(idx).shape + (3,), dtype=complex)
        return 1j * ev.K(idx) / ev.K2(idx)[..., None]
    return Field(k, rank=1, min_order=1, name="s1")


def pi_tensor(a):
    return Field(lambda ev, idx: ev.T(a, idx).astype(complex), rank=2, min_order=a, name=f"Pi{a}")


def projector_R():
    """``R_ij(K) delta``, i.e. ``Pi^[1]``."""
    return pi_tensor(1)


def tidal_K():
    """``K_ij = (d_i d_j / nabla^2 - delta_ij / 3) delta`` with kernel ``(R_ij(K) - delta_ij/3) F_n``."""
    def k(ev, idx):
        F = ev.FG_sym(idx)[0]
        return ((ev.R(idx) - np.eye(3) / 3.0) * F[..., None, None]).astype(complex)
    return Field(k, rank=2, name="K")


def product(A, B, contract=None, rank=None, name=None):
    """Normalized symmetrized product ``A B`` with an optional component contraction."""
    if contract is None:
        def contract(a, b):
            return a * b
        rank = A.rank + B.rank if rank is None else rank
        assert A.rank == 0 or B.rank == 0
    nm = name or f"({A.name}*{B.name})"

    def k(ev, idx):
        n = len(idx)
        out = 0.0
        for m in range(1, n):
            subsets = list(itertools.combinations(idx, m))
            acc = 0.0
            for sa in subsets:
                sb = tuple(i for i in idx if i not in sa)
                if len(sa) < A.min_order or len(sb) < B.min_order:
                    continue
                acc = acc + contract(A(ev, sa), B(ev, sb))
            out = out + acc / len(subsets)
        if np.isscalar(out):
            out = np.zeros(ev.K2(idx).shape + (3,) * rank, dtype=complex)
        return out
    return Field(k, rank=rank, min_order=A.min_order + B.min_order, name=nm)


def dot_vector(A, B, name=None):
    return product(A, B, contract=lambda a, b: np.sum(a * b, axis=-1), rank=0,
                   name=name or f"({A.name}.{B.name})")


def matmul(A, B, name=None):
    return product(A, B, contract=lambda a, b: np.einsum("...ij,...jk->...ik", a, b), rank=2,
                   name=name or f"({A.name}@{B.name})")


def trace_product(A, B, name=None):
    return product(A, B, contract=lambda a, b: np.einsum("...ij,...ji->...", a, b), rank=0,
                   name=name or f"tr({A.name}{B.name})")


def par(A, name=None):
    """LOS projection ``n_i A_ij n_j`` of a rank-2 field."""
    return Field(lambda ev, idx: np.einsum("...i,...ij,...j->...", ev.n, A(ev, idx), ev.n),
                 rank=0, min_order=A.min_order, name=name or f"par({A.name})")


def matmul_par(A, B, name=None):
    return par(matmul(A, B), name=name or f"par({A.name}@{B.name})")


def trace(A, name=None):
    return Field(lambda ev, idx: np.einsum("...ii->...", A(ev, idx)), rank=0,
                 min_order=A.min_order, name=name or f"tr({A.name})")


def grad_par(A, name=None):
    """``d_par A``: multiply the kernel by ``i K_par``."""
    return Field(lambda ev, idx: 1j * _dot(ev.K(idx), ev.n) * A(ev, idx), rank=A.rank,
                 min_order=A.min_order, name=name or f"dpar({A.name})")


def grad(A, name=None):
    """``d_i A`` of a scalar field (vector field)."""
    assert A.rank == 0
    return Field(lambda ev, idx: 1j * ev.K(idx) * A(ev, idx)[..., None], rank=1,
                 min_order=A.min_order, name=name or f"grad({A.name})")


def scale(c, A, name=None):
    return Field(lambda ev, idx: c * A(ev, idx), rank=A.rank, min_order=A.min_order,
                 name=name or f"({c!r}*{A.name})")


def add(*fields, name=None):
    rank = fields[0].rank
    return Field(lambda ev, idx: sum(F(ev, idx) for F in fields), rank=rank,
                 min_order=min(F.min_order for F in fields),
                 name=name or "(" + "+".join(F.name for F in fields) + ")")


def linear_combination(coeffs, fields, name=None):
    """``sum_a c_a O_a`` (scalar fields)."""
    terms = [(c, F) for c, F in zip(coeffs, fields)]
    return Field(lambda ev, idx: sum(c * F(ev, idx) for c, F in terms), rank=0,
                 min_order=min(F.min_order for _, F in terms), name=name)


def truncate_order(A, orders, name=None):
    """Keep only the listed perturbative orders of ``A`` (e.g. ``u^(2)`` alone)."""
    orders = tuple(orders)

    def k(ev, idx):
        if len(idx) in orders:
            return A(ev, idx)
        return np.zeros(ev.K2(idx).shape + (3,) * A.rank, dtype=complex)
    return Field(k, rank=A.rank, min_order=max(A.min_order, min(orders)),
                 name=name or f"{A.name}|{orders}")


# ======================================================================= operator library
def _operator_library():
    """Return a dict of named fields used by the DJS basis (full nonlinear kernels)."""
    d = density()
    e = eta()
    Kt = tidal_K()
    P2 = pi_tensor(2)
    P3 = pi_tensor(3)
    up = u_par()
    lib = dict(delta=d, eta=e, K=Kt, Pi2=P2, Pi3=P3, u_par=up)
    lib["delta2"] = product(d, d, name="delta2")
    lib["K2"] = trace_product(Kt, Kt, name="K2")
    lib["delta_eta"] = product(d, e, name="delta_eta")
    lib["eta2"] = product(e, e, name="eta2")
    lib["KK_par"] = matmul_par(Kt, Kt, name="KK_par")
    lib["Pi2_par"] = par(P2, name="Pi2_par")
    lib["Pi3_par"] = par(P3, name="Pi3_par")
    lib["delta_Pi2_par"] = product(d, lib["Pi2_par"], name="delta_Pi2_par")
    lib["eta_Pi2_par"] = product(e, lib["Pi2_par"], name="eta_Pi2_par")
    lib["KPi2_par"] = matmul_par(Kt, P2, name="KPi2_par")
    # O_td = 8/21 K_ij D_ij (delta^2 - 3/2 K^2) with *linear* fields
    dlin = truncate_order(d, (1,), name="delta_lin")
    Klin = truncate_order(Kt, (1,), name="K_lin")
    src = add(product(dlin, dlin, name="d2lin"), scale(-1.5, trace_product(Klin, Klin, name="K2lin")),
              name="td_source")

    def Dij_src(ev, idx):
        return ((ev.R(idx) - np.eye(3) / 3.0) * src(ev, idx)[..., None, None])
    Dsrc = Field(Dij_src, rank=2, min_order=2, name="D_td_source")
    lib["O_td"] = scale(8.0 / 21.0, trace_product(Klin, Dsrc, name="tr_K_Dsrc"), name="O_td")
    return lib


DJS_QUADRATIC_ORDER = ("delta^(2)", "eta^(2)", "delta^2", "K^2", "delta eta", "eta^2",
                       "(KK)_par", "Pi^[2]_par", "u_par d_par delta", "u_par d_par eta")

DJS_CUBIC_REDUCED_ORDER = ("delta^(3)", "eta^(3)", "2tr(K K^(2))", "delta eta^(2)", "2 eta eta^(2)",
                           "2(K K^(2))_par", "O_td", "delta Pi^[2]_par", "eta Pi^[2]_par",
                           "(K Pi^[2])_par", "s^i d_i Pi^[2]_par", "u^(2)_par d_par delta",
                           "u^(2)_par d_par eta", "u_par d_par eta^(2)", "u_par d_par Pi^[2]_par",
                           "Pi^[3]_par")


def mapped_djs_field(coeffs):
    """DJS compact mapped field (equation note E4) with *mapped* coefficients.

    ``coeffs`` is a mapping with keys ``b1, b_eta, b2, b_K2, b_delta_eta, b_eta2, b_KK_par,
    b_Pi2_par, b_td, b_delta_Pi2_par, b_eta_Pi2_par, b_KPi2_par, b_Pi3_par``.

        delta_gs = sum_O b_O [O] - u_par d_par (b1 delta + b_eta eta + b_Pi2 Pi^[2]_par)

    All operators carry their full nonlinear kernels.  Valid for the one-loop P(k, mu)
    after renormalization (see the equation note); not a field identity in general.
    """
    lib = _operator_library()
    c = dict(coeffs)
    terms = [
        (c["b1"], lib["delta"]), (c["b_eta"], lib["eta"]), (0.5 * c["b2"], lib["delta2"]),
        (c["b_K2"], lib["K2"]), (c["b_delta_eta"], lib["delta_eta"]), (c["b_eta2"], lib["eta2"]),
        (c["b_KK_par"], lib["KK_par"]), (c["b_Pi2_par"], lib["Pi2_par"]), (c["b_td"], lib["O_td"]),
        (c["b_delta_Pi2_par"], lib["delta_Pi2_par"]), (c["b_eta_Pi2_par"], lib["eta_Pi2_par"]),
        (c["b_KPi2_par"], lib["KPi2_par"]), (c["b_Pi3_par"], lib["Pi3_par"]),
    ]
    adv_target = linear_combination([c["b1"], c["b_eta"], c["b_Pi2_par"]],
                                    [lib["delta"], lib["eta"], lib["Pi2_par"]], name="adv_target")
    adv = product(lib["u_par"], grad_par(adv_target), name="u_dpar_target")
    fields = [F for _, F in terms] + [adv]
    cs = [w for w, _ in terms] + [-1.0]
    return linear_combination(cs, fields, name="djs_mapped")


def rest_frame_density(coeffs_rest):
    """Rest-frame (pre-RSD) selected galaxy density with *rest-frame* coefficients.

    ``coeffs_rest`` keys: ``b1, bR_eta, b2, b_K2, bR_delta_eta, bR_eta2, bR_KK_par,
    bR_Pi2_par, b_td, bR_delta_Pi2_par, bR_eta_Pi2_par, bR_KPi2_par, bR_Pi3_par``.
    The galaxies move with the matter velocity (no velocity bias).
    """
    lib = _operator_library()
    c = dict(coeffs_rest)
    pairs = [
        (c["b1"], lib["delta"]), (c["bR_eta"], lib["eta"]), (0.5 * c["b2"], lib["delta2"]),
        (c["b_K2"], lib["K2"]), (c["bR_delta_eta"], lib["delta_eta"]), (c["bR_eta2"], lib["eta2"]),
        (c["bR_KK_par"], lib["KK_par"]), (c["bR_Pi2_par"], lib["Pi2_par"]), (c["b_td"], lib["O_td"]),
        (c["bR_delta_Pi2_par"], lib["delta_Pi2_par"]), (c["bR_eta_Pi2_par"], lib["eta_Pi2_par"]),
        (c["bR_KPi2_par"], lib["KPi2_par"]), (c["bR_Pi3_par"], lib["Pi3_par"]),
    ]
    return linear_combination([w for w, _ in pairs], [F for _, F in pairs], name="rest_density")


def exact_map_field(D):
    r"""Exact plane-parallel redshift-space map of a rest-frame density ``D``:

        delta_s(k) = \int d^3x e^{-ik.x} [(1 + D) e^{-i k_par u_par} - 1]

    expanded through third order.  With ``U = u_par`` this gives (E4)

        Z1 = D1 - i k_par U1
        Z2 = D2 - i k_par [U2 + Sym(D1 U1)] - k_par^2/2 U1 U1
        Z3 = D3 - i k_par [U3 + Sym(D1 U2) + Sym(D2 U1)]
             - k_par^2 [Sym(U1 U2) + 1/2 Sym(D1 U1 U1)] + i/6 k_par^3 U1 U1 U1

    implemented here as products of fields, so the combinatorics are generated.
    """
    up = u_par()
    uD = product(up, D, name="uD")
    uu = product(up, up, name="uu")
    uuD = product(uu, D, name="uuD")
    uuu = product(uu, up, name="uuu")

    def k(ev, idx):
        kpar = _dot(ev.K(idx), ev.n)
        mik = -1j * kpar
        out = D(ev, idx) + mik * (up(ev, idx) + uD(ev, idx))
        out = out + 0.5 * mik ** 2 * (uu(ev, idx) + uuD(ev, idx))
        out = out + (mik ** 3 / 6.0) * uuu(ev, idx)
        return out
    return Field(k, rank=0, name=f"exactmap({D.name})")


def reduced_cubic_operator_fields():
    """The sixteen *physical* cubic operators of the reduced P13 list (E7), as fields whose
    only nonvanishing order is 3, in ``DJS_CUBIC_REDUCED_ORDER``.  Growth factors are those
    of the physical operators (divide by ``f**n_f`` for the growth-stripped matrices)."""
    lib = _operator_library()
    d, e, Kt, P2 = lib["delta"], lib["eta"], lib["K"], lib["Pi2"]
    up = lib["u_par"]
    d1 = truncate_order(d, (1,))
    e1, e2, e3 = truncate_order(e, (1,)), truncate_order(e, (2,)), truncate_order(e, (3,))
    K1, K2_ = truncate_order(Kt, (1,)), truncate_order(Kt, (2,))
    u1, u2 = truncate_order(up, (1,)), truncate_order(up, (2,))
    P2par = lib["Pi2_par"]
    s1 = displacement_s1()
    ops = [
        truncate_order(d, (3,), name="d3"),
        e3,
        scale(2.0, trace_product(K1, K2_), name="2trKK2"),
        product(d1, e2, name="d1e2"),
        scale(2.0, product(e1, e2), name="2e1e2"),
        scale(2.0, matmul_par(K1, K2_), name="2KK2par"),
        lib["O_td"],
        product(d1, P2par, name="d1Pi2"),
        product(e1, P2par, name="e1Pi2"),
        matmul_par(K1, P2, name="K1Pi2par"),
        dot_vector(s1, grad(P2par), name="s1gradPi2"),
        product(u2, grad_par(d1), name="u2dd1"),
        product(u2, grad_par(e1), name="u2de1"),
        product(u1, grad_par(e2), name="u1de2"),
        product(u1, grad_par(P2par), name="u1dPi2"),
        lib["Pi3_par"],
    ]
    return [truncate_order(op, (3,), name=op.name + "|3") for op in ops]


# ============================================================ closed forms for independent tests
def F2_closed_form(k1, k2):
    k1n, k2n = np.sqrt(_dot(k1, k1)), np.sqrt(_dot(k2, k2))
    nu = _dot(k1, k2) / (k1n * k2n)
    r = k1n / k2n
    return 5 / 7 + 0.5 * nu * (r + 1 / r) + 2 / 7 * nu ** 2


def G2_closed_form(k1, k2):
    k1n, k2n = np.sqrt(_dot(k1, k1)), np.sqrt(_dot(k2, k2))
    nu = _dot(k1, k2) / (k1n * k2n)
    r = k1n / k2n
    return 3 / 7 + 0.5 * nu * (r + 1 / r) + 4 / 7 * nu ** 2


def quadratic_table_kernels(k1, k2, n_los, f):
    """The ten quadratic kernels of the equation-note table (E5), written out by hand
    from the printed formulas (*not* generated by the field algebra), in
    ``DJS_QUADRATIC_ORDER``.  Used as an independent check of the oracle and of the
    symbolic coefficient generator."""
    k1n, k2n = np.sqrt(_dot(k1, k1)), np.sqrt(_dot(k2, k2))
    K = k1 + k2
    Kn = np.sqrt(_dot(K, K))
    mu1, mu2 = _dot(k1, n_los) / k1n, _dot(k2, n_los) / k2n
    with np.errstate(invalid="ignore", divide="ignore"):
        muK = np.where(Kn > 0, _dot(K, n_los) / np.where(Kn > 0, Kn, 1.0), 0.0)
    nu = _dot(k1, k2) / (k1n * k2n)
    r = k1n / k2n
    S = [
        F2_closed_form(k1, k2),
        -f * muK ** 2 * G2_closed_form(k1, k2),
        np.ones_like(nu),
        nu ** 2 - 1 / 3,
        -f * (mu1 ** 2 + mu2 ** 2) / 2,
        f ** 2 * mu1 ** 2 * mu2 ** 2,
        mu1 * mu2 * nu - (mu1 ** 2 + mu2 ** 2) / 3 + 1 / 9,
        mu1 * mu2 * nu + 5 / 7 * muK ** 2 * (1 - nu ** 2),
        -f * mu1 * mu2 * (r + 1 / r) / 2,
        f ** 2 * mu1 * mu2 * (mu1 ** 2 * r + mu2 ** 2 / r) / 2,
    ]
    return np.array(S)
