#!/usr/bin/env python
r"""Generate the coefficient data of the EdS one-loop LOS-selection templates.

Output: ``folps/_selection_coefficients_data.py`` (pure-Python rational data) and
``docs/selection_1loop_generation_report.json`` (residuals of every identity check).

Stages
------
1. **P22 tensor** (SymPy, exact): expand every product ``S_a S_b`` of the ten quadratic
   kernels of the equation note (E5, written from the printed formulas), average
   analytically over the azimuth of ``q`` about ``k`` with a general LOS, and reduce to
   ``mu^{2n} nu^m q^{r-2} k^{6-r} / |k-q|^4``.  Asserts that only the 23 DJS functions occur.
2. **UV scheme coefficients** (SymPy, exact): large-``q`` expansion of each ``I_mr`` and of
   each ``J_n`` to order ``k^2``; these are the pieces dropped by the analytically continued
   FFTLog integrals and are restored as explicit scheme terms.
3. **Reduced cubic matrices** (exact identification): for each of the sixteen reduced
   operators, the azimuthally and polar-angle averaged single 13 contraction
   ``3 Z3(k, q, -q)`` computed by the independent kernel oracle
   (:mod:`folps.selection_kernels`) is decomposed *pointwise in x = q/k* on an
   over-determined grid in the complete basis ``{j_1..j_5, 1/x^2, 1}``; the residual must
   be at round-off level and the coefficients are rationalized and re-verified.
4. **Bare bridge**: the same decomposition for the full DJS mapped field (A) and for the
   exact redshift-space map of a rest-frame density (B), coefficient by coefficient and as
   polynomials in ``f`` and ``mu^2``.  Asserts (i) the J parts of (A) and (B) both equal the
   reduced list under the documented coefficient maps, and (ii) (A) and (B) differ only by
   ``1/x^2`` (``k^2 sigma_{-1}^2``) and constant (``sigma_0^2``) shapes.  The (B)
   ``1/x^2`` coefficients define the bare derivative bridge used in production.

Usage::

    python scripts/generate_selection_coefficients.py            # full (~30-40 min, 1 core)
    python scripts/generate_selection_coefficients.py --check     # regenerate and compare

The production package never runs this script; it only imports the generated data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("FOLPS_BACKEND", "numpy")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util  # noqa: E402

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402


def load_oracle():
    """Import folps/selection_kernels.py standalone (it depends on NumPy only), so that the
    generator never imports the package data it is producing."""
    path = ROOT / "folps" / "selection_kernels.py"
    spec = importlib.util.spec_from_file_location("_selection_kernels_oracle", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

GENERATOR_VERSION = "selection-eds-gen-1.0"
DATA_PATH = ROOT / "folps" / "_selection_coefficients_data.py"
REPORT_PATH = ROOT / "docs" / "selection_1loop_generation_report.json"

# Quadratic kernel basis used for the stored tensor: the ten kernels of the equation note
# (E5) except that the two LOS-advection kernels are replaced by Galilean-safe combinations
#   S'_8 = (u_par d_par delta) + (delta eta),   S'_9 = (u_par d_par eta) + eta^2,
# which are suppressed as (k/q)^2 at large q.  The coefficients transform as
#   c'_4 = b_delta_eta + b1 (= s_delta_eta),  c'_5 = b_eta2 + b_eta (= s_eta2 + s_eta).
QUAD_NAMES = ["delta^(2)", "eta^(2)", "delta^2", "K^2", "delta eta", "eta^2", "(KK)_par",
              "Pi^[2]_par", "u_par d_par delta + delta eta", "u_par d_par eta + eta^2"]
QUAD_NF = [0, 1, 0, 0, 1, 2, 0, 0, 1, 2]
# kernels singular (~1/q) in the soft limit and suppressed as (k/q)^2 at large q
QUAD_SINGULAR = [0, 1, 8, 9]
P22_BASIS = [(m, r) for m in range(9) for r in range(7) if (r - m) % 2 == 0 and m <= r + 2]
CUBIC_NF = [0, 1, 0, 1, 2, 0, 0, 0, 1, 0, 0, 1, 2, 2, 1, 0]
DJS_KEYS = ["b1", "b_eta", "b2", "b_K2", "b_delta_eta", "b_eta2", "b_KK_par", "b_Pi2_par", "b_td",
            "b_delta_Pi2_par", "b_eta_Pi2_par", "b_KPi2_par", "b_Pi3_par"]
REST_KEYS = ["b1", "bR_eta", "b2", "b_K2", "bR_delta_eta", "bR_eta2", "bR_KK_par", "bR_Pi2_par", "b_td",
             "bR_delta_Pi2_par", "bR_eta_Pi2_par", "bR_KPi2_par", "bR_Pi3_par"]


# ====================================================================== stage 1: P22 tensor
def stage_p22():
    q, Q, nu, mu, c, w, s = sp.symbols("q Q nu mu c w s", real=True)
    R = sp.Rational
    mu1 = s * w * c + mu * nu          # q-hat . n with k along z, n = (s, 0, mu), s = sqrt(1-mu^2)
    mu2 = (mu - q * mu1) / Q           # (k-q)-hat . n   (k = 1)
    nu12 = (nu - q) / Q                # q-hat . (k-q)-hat
    r = q / Q
    F2 = R(5, 7) + nu12 * (r + 1 / r) / 2 + R(2, 7) * nu12 ** 2
    G2 = R(3, 7) + nu12 * (r + 1 / r) / 2 + R(4, 7) * nu12 ** 2
    S = [F2, -mu ** 2 * G2, sp.Integer(1), nu12 ** 2 - R(1, 3), -(mu1 ** 2 + mu2 ** 2) / 2,
         mu1 ** 2 * mu2 ** 2, mu1 * mu2 * nu12 - (mu1 ** 2 + mu2 ** 2) / 3 + R(1, 9),
         mu1 * mu2 * nu12 + R(5, 7) * mu ** 2 * (1 - nu12 ** 2),
         -mu1 * mu2 * (r + 1 / r) / 2, mu1 * mu2 * (mu1 ** 2 * r + mu2 ** 2 / r) / 2]  # growth stripped
    S[8] = S[8] + S[4]   # Galilean-safe advection combinations (see QUAD_NAMES)
    S[9] = S[9] + S[5]
    cavg = {j: (Fraction(int(sp.factorial2(j - 1)), int(sp.factorial2(j))) if j % 2 == 0 else Fraction(0))
            for j in range(1, 24)}
    cavg[0] = Fraction(1)
    tensor = {}
    used = set()
    for a in range(10):
        for b in range(a, 10):
            e = sp.expand(S[a] * S[b] * q ** 2 * Q ** 4)
            out = 0
            for (j,), coeff in sp.Poly(e, c).terms():
                out += coeff * sp.Rational(cavg[j].numerator, cavg[j].denominator)
            out = sp.expand(sp.expand(out).subs({w ** 2: 1 - nu ** 2}).subs({s ** 2: 1 - mu ** 2}))
            assert not out.has(w) and not out.has(s), (a, b)
            num = 0
            for (j,), coeff in sp.Poly(out, Q).terms():
                assert j % 2 == 0, (a, b, j)
                num += coeff * (1 + q ** 2 - 2 * q * nu) ** (j // 2)
            for (pmu, m, p), coeff in sp.Poly(sp.expand(num), mu, nu, q).terms():
                assert pmu % 2 == 0
                assert (m, p) in P22_BASIS, ("basis function outside the 23-set", a, b, m, p)
                used.add((m, p))
                tensor[(a, b, pmu // 2, m, p)] = Fraction(int(sp.numer(coeff)), int(sp.denom(coeff)))
    assert used == set(P22_BASIS), sorted(set(P22_BASIS) - used)
    return tensor


# =============================================================== stage 2: UV scheme pieces
def stage_uv():
    r"""Order-k^2 large-q coefficients.

    P22:  I_mr^true - I_mr^continued = k^2/(2 pi^2) [a_mr R0 + b_mr S],  R0 = \int dq P^2,
          S = \int dq q^2 P'^2 (from integration by parts of the P P' and P P'' terms).
    P13:  J_n^true - J_n^continued = dJ_n k^2 sigma_{-1}^2,  sigma_{-1}^2 = \int_q P/q^2.
    """
    k, q, nu, P0, P1, P2 = sp.symbols("k q nu P0 P1 P2", positive=True)
    Q = sp.sqrt(q ** 2 - 2 * k * q * nu + k ** 2)
    dQ = sp.series(Q - q, k, 0, 3).removeO()
    PQ = P0 + dQ * P1 + dQ ** 2 * P2 / 2
    p22 = {}
    for (m, r) in P22_BASIS:
        integrand = 2 * nu ** m * q ** (r - 2) * k ** (6 - r) * sp.series(Q ** -4, k, 0, 3).removeO() * PQ * P0
        c2 = sp.expand(sp.series(sp.expand(integrand), k, 0, 3).removeO()).coeff(k, 2)
        c2 = sp.integrate(sp.expand(c2), (nu, -1, 1)) / 2        # angular average
        c2 = sp.expand(c2 * q ** 2)                               # \int_q X = (1/2pi^2) \int dq q^2 X
        aP00 = c2.coeff(P0, 2)
        aP01 = sp.expand(c2.coeff(P0, 1).coeff(P1, 1))
        aP02 = sp.expand(c2.coeff(P0, 1).coeff(P2, 1))
        # each must be q^0 * (P0^2), q^1 (P0 P1), q^2 (P0 P2) for a dimensionless convergent piece
        a00 = sp.nsimplify(aP00)
        a01 = sp.nsimplify(sp.simplify(aP01 / q))
        a02 = sp.nsimplify(sp.simplify(aP02 / q ** 2))
        assert not a00.has(q) and not a01.has(q) and not a02.has(q), (m, r, a00, a01, a02)
        # \int q P P' = -R0/2 ;  \int q^2 P P'' = R0 - S
        aR0 = a00 - a01 / 2 + a02
        aS = -a02
        p22[(m, r)] = (Fraction(str(aR0)), Fraction(str(aS)))
    x, nuu = sp.symbols("x nu", positive=True)
    Q2 = 1 + x ** 2 - 2 * x * nuu
    jint = [nuu * (1 - nuu ** 2) / (x * Q2), (1 - nuu ** 2) / Q2, x ** 2 * (1 - nuu ** 2) / Q2,
            x ** 2 * nuu ** 2 * (1 - nuu ** 2) / Q2, x ** 2 * nuu ** 4 * (1 - nuu ** 2) / Q2]
    dJ = []
    for e in jint:
        y = sp.symbols("y", positive=True)       # y = 1/x
        ser = sp.series(e.subs(x, 1 / y), y, 0, 3).removeO()
        c2 = sp.integrate(sp.expand(ser).coeff(y, 2), (nuu, -1, 1)) / 2
        dJ.append(Fraction(str(sp.nsimplify(c2))))
    return p22, dJ


def check_p22_classes(p22, uv22, report):
    """Every pair with a soft-singular kernel must have no UV constant; singular x singular
    pairs must also have no UV k^2 piece (so they are UV-convergent at the matter tilt)."""
    worst = {}
    for a in range(10):
        for b in range(a, 10):
            for n in range(5):
                const = sum(p22.get((a, b, n, m, 6), Fraction(0)) * Fraction(2, m + 1) for m in range(0, 9, 2))
                k2R0 = sum(p22.get((a, b, n, m, r), Fraction(0)) * uv22[(m, r)][0] for (m, r) in P22_BASIS)
                k2S = sum(p22.get((a, b, n, m, r), Fraction(0)) * uv22[(m, r)][1] for (m, r) in P22_BASIS)
                sing = (a in QUAD_SINGULAR) + (b in QUAD_SINGULAR)
                if sing >= 1:
                    assert const == 0, ("UV constant in a pair with a soft-singular kernel", a, b, n, const)
                if sing == 2:
                    assert k2R0 == 0 and k2S == 0, ("UV k^2 in a singular x singular pair", a, b, n)
                worst[f"{a},{b},{n}"] = [str(const), str(k2R0), str(k2S)]
    report["checks"]["p22_pair_uv_content(const,k2_R0,k2_S)"] = worst


# ======================================================== stage 3/4: oracle identification
class Identifier:
    def __init__(self, nnu=200, nphi=20, nx=30, mus=None, fs=None):
        self.sk = load_oracle()
        self.xnu, self.wnu = np.polynomial.legendre.leggauss(nnu)
        self.nphi = nphi
        phi = 2 * np.pi * np.arange(nphi) / nphi
        self.NU, self.PH = np.meshgrid(self.xnu, phi, indexing="ij")
        self.ST = np.sqrt(1 - self.NU ** 2)
        half = nx // 2
        self.xs = np.concatenate([np.geomspace(0.04, 0.8, half), np.geomspace(1.25, 25, nx - half)])
        self.mus = np.array([0.0, 0.25, 0.45, 0.6, 0.75, 0.9, 1.0]) if mus is None else np.asarray(mus)
        self.fs = np.array([0.4, 0.65, 0.9, 1.15, 1.4, 1.7]) if fs is None else np.asarray(fs)
        A = np.column_stack([np.array([self.jbasis(x) for x in self.xs]), 1 / self.xs ** 2, np.ones_like(self.xs)])
        self.A = A
        self.pinv = np.linalg.pinv(A)

    def jbasis(self, x):
        nu, w = self.xnu, 0.5 * self.wnu
        Q2 = 1 + x * x - 2 * x * nu
        return np.array([np.sum(w * nu * (1 - nu ** 2) / (x * Q2)), np.sum(w * (1 - nu ** 2) / Q2),
                         np.sum(w * x * x * (1 - nu ** 2) / Q2) - 2 / 3,
                         np.sum(w * x * x * nu ** 2 * (1 - nu ** 2) / Q2) - 2 / 15,
                         np.sum(w * x * x * nu ** 4 * (1 - nu ** 2) / Q2) - 2 / 35])

    def g(self, field, mu, f):
        n = np.array([np.sqrt(1 - mu ** 2), 0.0, mu])
        out = np.zeros(len(self.xs))
        for ix, x in enumerate(self.xs):
            qv = x * np.stack([self.ST * np.cos(self.PH), self.ST * np.sin(self.PH), self.NU], -1)
            kv = np.broadcast_to(np.array([0.0, 0.0, 1.0]), qv.shape)
            ev = self.sk.Evaluator([kv, qv, -qv], n, f)
            val = 3 * field(ev, (0, 1, 2))
            assert np.max(np.abs(val.imag)) < 1e-10 * max(1.0, np.max(np.abs(val.real)))
            out[ix] = 0.5 * np.sum(self.wnu[:, None] * val.real) / self.nphi
        return out

    def decompose(self, make_field, fdeg, mudeg=4):
        """Return coefficients ``[f_power, mu2_power, component]`` (components j1..j5, 1/x^2, 1)
        and the worst relative residuals of the pointwise fit and of the polynomial fits."""
        vals = np.zeros((len(self.fs), len(self.mus), 7))
        res = 0.0
        for i_f, f in enumerate(self.fs):
            field = make_field()
            for im, mu in enumerate(self.mus):
                G = self.g(field, mu, f)
                c = self.pinv @ G
                vals[i_f, im] = c
                res = max(res, np.max(np.abs(self.A @ c - G)) / max(np.max(np.abs(G)), 1e-30))
        Vf = np.vander(self.fs, fdeg + 1, increasing=True)
        Vm = np.vander(self.mus ** 2, mudeg + 1, increasing=True)
        coef = np.einsum("ai,bj,ijc->abc", np.linalg.pinv(Vf), np.linalg.pinv(Vm), vals)
        recon = np.einsum("ia,jb,abc->ijc", Vf, Vm, coef)
        polres = np.max(np.abs(recon - vals)) / max(np.max(np.abs(vals)), 1e-30)
        return coef, res, polres


NOISE_FLOOR = 1e-7   # all nonzero exact coefficients found are > 1e-4 in magnitude


def rationalize(arr, max_den=200000, tol=1e-8):
    """Exact rationals from identification output.  Entries below NOISE_FLOOR are fit noise
    and set to zero; every other entry must be reproduced by a rational with a modest
    denominator to ``tol`` (relative), otherwise the generator stops."""
    arr = np.asarray(arr, dtype=float)
    out = np.empty(arr.shape, dtype=object)
    worst = 0.0
    for idx, v in np.ndenumerate(arr):
        fr = Fraction(v).limit_denominator(max_den)
        if abs(v) < NOISE_FLOOR:
            fr = Fraction(0)
        err = abs(float(fr) - v)
        worst = max(worst, err)
        if err > tol * max(1.0, abs(v)):
            raise ValueError(f"could not rationalize {v!r} at {idx} (best {fr}, err {err:.2e})")
        out[idx] = fr
    return out, worst


def stage_cubic(ident, report):
    sk = ident.sk
    ops = sk.reduced_cubic_operator_fields()
    M = np.empty((16, 3, 5), dtype=object)
    D = np.empty((16, 4), dtype=object)
    C = np.empty((16, 4), dtype=object)
    for a, op in enumerate(ops):
        t0 = time.time()
        coef, res, polres = ident_decompose_fixed_f(ident, op, CUBIC_NF[a])
        rat, worst = rationalize(coef)
        M[a] = rat[:3, :5]
        D[a] = rat[:4, 5]
        C[a] = rat[:4, 6]
        assert all(v == 0 for v in rat[3, :5].ravel()), "unexpected mu^6 row in a reduced matrix"
        report["cubic"][sk.DJS_CUBIC_REDUCED_ORDER[a]] = dict(fit_rel_residual=res, mu_poly_rel_residual=polres,
                                                             rationalization_abs_error=worst,
                                                             seconds=round(time.time() - t0, 1))
        print(f"[cubic] {sk.DJS_CUBIC_REDUCED_ORDER[a]:28s} res={res:.1e} poly={polres:.1e}", flush=True)
    return M, D, C


def ident_decompose_fixed_f(ident, op, nf):
    """Growth-stripped decomposition of a single reduced operator: evaluate at f=1 (its
    kernel is f^nf times the stripped kernel) and fit a polynomial in mu^2 of degree 3."""
    vals = np.zeros((len(ident.mus), 7))
    res = 0.0
    for im, mu in enumerate(ident.mus):
        G = ident.g(op, mu, 1.0)
        c = ident.pinv @ G
        vals[im] = c
        res = max(res, np.max(np.abs(ident.A @ c - G)) / max(np.max(np.abs(G)), 1e-30))
    Vm = np.vander(ident.mus ** 2, 4, increasing=True)
    coef = np.linalg.pinv(Vm) @ vals
    polres = np.max(np.abs(Vm @ coef - vals)) / max(np.max(np.abs(vals)), 1e-30)
    return coef, res, polres


CACHE_DIR = None  # set by --resume-dir: intermediate decompositions are stored and reused


def cached_decompose(ident, tag, make_field):
    """Kernels are cubic in f and their 13 contractions are polynomials of degree <= 3 in
    mu^2, so degree-3 fits on the 6 x 7 (f, mu) grid are over-determined."""
    path = None
    if CACHE_DIR is not None:
        path = Path(CACHE_DIR) / (tag.replace(":", "_") + ".npz")
        if path.exists():
            d = np.load(path)
            return d["coef"], float(d["res"]), float(d["polres"])
    coef, res, polres = ident.decompose(make_field, fdeg=3, mudeg=3)
    if path is not None:
        np.savez(path, coef=coef, res=res, polres=polres)
    return coef, res, polres


def stage_bridge(ident, report):
    sk = ident.sk
    out = {"A": {}, "B": {}}
    for key in DJS_KEYS:
        c = {k: 0.0 for k in DJS_KEYS}
        c[key] = 1.0
        t0 = time.time()
        coef, res, polres = cached_decompose(ident, "A:" + key, lambda c=c: sk.mapped_djs_field(c))
        out["A"][key] = coef
        report["bridge"]["A:" + key] = dict(fit_rel_residual=res, poly_rel_residual=polres,
                                            seconds=round(time.time() - t0, 1))
        print(f"[bridge A] {key:16s} res={res:.1e} poly={polres:.1e}", flush=True)
    zero = {k: 0.0 for k in REST_KEYS}
    for key in REST_KEYS + ["matter"]:
        t0 = time.time()
        if key == "matter":
            def mk():
                return sk.exact_map_field(sk.rest_frame_density(zero))
        else:
            c = dict(zero)
            c[key] = 1.0

            def mk(c=c):
                Fa = sk.exact_map_field(sk.rest_frame_density(c))
                Fb = sk.exact_map_field(sk.rest_frame_density(zero))
                return sk.Field(lambda ev, idx: Fa(ev, idx) - Fb(ev, idx), name="diff")
        coef, res, polres = cached_decompose(ident, "B:" + key, mk)
        out["B"][key] = coef
        report["bridge"]["B:" + key] = dict(fit_rel_residual=res, poly_rel_residual=polres,
                                            seconds=round(time.time() - t0, 1))
        print(f"[bridge B] {key:16s} res={res:.1e} poly={polres:.1e}", flush=True)
    return out


# ------------------------------------------------------------------ coefficient bookkeeping
def reduced_c13_vectors():
    """For each DJS coefficient direction, the reduced-list coefficient vector c_13 (16)."""
    vec = {k: [Fraction(0)] * 16 for k in DJS_KEYS}
    vec["b1"][0] += 1; vec["b1"][11] -= 1
    vec["b_eta"][1] += 1; vec["b_eta"][12] -= 1; vec["b_eta"][13] -= 1
    vec["b_K2"][2] += 1
    vec["b_delta_eta"][3] += 1
    vec["b_eta2"][4] += 1
    vec["b_KK_par"][5] += 1
    vec["b_td"][6] += 1
    vec["b_delta_Pi2_par"][7] += 1
    vec["b_eta_Pi2_par"][8] += 1
    vec["b_KPi2_par"][9] += 1
    vec["b_Pi2_par"][10] -= 1; vec["b_Pi2_par"][14] -= 1; vec["b_Pi2_par"][15] += 2
    vec["b_Pi3_par"][15] += 1
    return vec


def rest_to_djs_linear():
    """DJS mapped coefficients as affine functions of rest-frame ones (derived in the
    equation note): returns ``(offset, matrix)`` with b_DJS = offset + matrix @ b_rest,
    where offsets/matrix entries are polynomials in nothing but constants and b1."""
    # b_eta = bR_eta - 1 ; b_delta_eta = bR_delta_eta - b1 ; b_eta2 = bR_eta2 - bR_eta + 1 ;
    # b_eta_Pi2 = bR_eta_Pi2 - bR_Pi2 ; others identical.
    M = {k: {kk: Fraction(0) for kk in REST_KEYS} for k in DJS_KEYS}
    off = {k: Fraction(0) for k in DJS_KEYS}
    ident = dict(zip(DJS_KEYS, REST_KEYS))
    for kd, kr in ident.items():
        M[kd][kr] = Fraction(1)
    off["b_eta"] = Fraction(-1)
    M["b_delta_eta"]["b1"] = Fraction(-1)
    M["b_eta2"]["bR_eta"] = Fraction(-1)
    off["b_eta2"] = Fraction(1)
    M["b_eta_Pi2_par"]["bR_Pi2_par"] = Fraction(-1)
    return off, M


def check_bridge(Mred, bridge, report):
    """(i) J parts of (A) per DJS coefficient = reduced list; (ii) J parts of (B) per rest
    coefficient = reduced list composed with the coefficient map; (iii) (A)-(B) differ only
    by derivative and constant shapes."""
    c13 = reduced_c13_vectors()
    fpow = np.array(CUBIC_NF)
    Mf = np.array([[[float(v) for v in row] for row in Mred[a]] for a in range(16)])  # (16,3,5)

    shape = bridge["B"]["matter"].shape[:2]

    def reduced_J(cvec):
        # returns array [fpow, mu2pow, 5]
        out = np.zeros(shape + (5,))
        for a in range(16):
            ca = float(cvec[a])
            if ca:
                out[fpow[a], :3, :] += ca * Mf[a]
        return out
    worstA = 0.0
    for key in DJS_KEYS:
        diff = bridge["A"][key][:, :, :5] - reduced_J(c13[key])
        worstA = max(worstA, np.max(np.abs(diff)))
    off, Mmap = rest_to_djs_linear()
    worstB = 0.0
    zero_rest = bridge["B"]["matter"]
    # matter part: rest coefficients all zero -> DJS offsets (b_eta=-1, b_eta2=+1)
    cvec = [Fraction(0)] * 16
    for kd in DJS_KEYS:
        if off[kd]:
            cvec = [x + off[kd] * y for x, y in zip(cvec, c13[kd])]
    worstB = max(worstB, np.max(np.abs(zero_rest[:, :, :5] - reduced_J(cvec))))
    for kr in REST_KEYS:
        cvec = [Fraction(0)] * 16
        for kd in DJS_KEYS:
            if Mmap[kd][kr]:
                cvec = [x + Mmap[kd][kr] * y for x, y in zip(cvec, c13[kd])]
        worstB = max(worstB, np.max(np.abs(bridge["B"][kr][:, :, :5] - reduced_J(cvec))))
    report["checks"]["A_J_part_equals_reduced_list_max_abs"] = float(worstA)
    report["checks"]["B_J_part_equals_mapped_reduced_list_max_abs"] = float(worstB)
    print(f"[check] A J-part vs reduced: {worstA:.2e}; B J-part vs mapped reduced: {worstB:.2e}")
    assert worstA < 1e-8 and worstB < 1e-8


def check_bridge_closed_form(bridge, report):
    """The bare k^2 sigma_{-1}^2 coefficient of the exact map is
    -1/6 [1 + f mu^2 (2 + f)] Z1 with Z1 = b1 - (bR_eta - 1) f mu^2: zero for every other
    rest-frame coefficient.  Verify on the fitted polynomials."""
    S = np.zeros((4, 4))           # 1 + 2 f mu^2 + f^2 mu^2  as [f^p, mu^2n]
    S[0, 0], S[1, 1], S[2, 1] = 1, 2, 1
    expect = {k: np.zeros((4, 4)) for k in REST_KEYS + ["matter"]}
    expect["b1"] = -S / 6
    expect["bR_eta"][1:, 1:] = (S / 6)[:-1, :-1]          # +1/6 f mu^2 S
    expect["matter"][1:, 1:] = -(S / 6)[:-1, :-1]        # -1/6 f mu^2 S
    worst = max(np.max(np.abs(bridge["B"][k][:, :, 5] - expect[k])) for k in expect)
    report["checks"]["bridge_closed_form_max_abs"] = float(worst)
    print(f"[check] bare bridge = -1/6 [1 + f mu^2 (2+f)] Z1: max abs {worst:.2e}")
    assert worst < 1e-8


# ----------------------------------------------------------------------------- writing
def frac_repr(fr):
    return f"({fr.numerator}, {fr.denominator})"


def write_data(p22, uv22, dJ, Mred, Dred, Cred, bridgeB, path):
    lines = ['"""Generated by scripts/generate_selection_coefficients.py -- do not edit by hand.',
             "",
             "Rational numbers are stored as (numerator, denominator) integer pairs.",
             '"""', "", f"GENERATOR_VERSION = {GENERATOR_VERSION!r}", "",
             f"QUADRATIC_KERNELS = {QUAD_NAMES!r}", f"QUADRATIC_NF = {QUAD_NF!r}",
             f"QUADRATIC_SINGULAR = {QUAD_SINGULAR!r}",
             f"P22_BASIS = {P22_BASIS!r}", "",
             "# P22_TENSOR[(a, b, n, m, r)] (a <= b): coefficient of mu^(2n) nu^m q^(r-2) k^(6-r) / |k-q|^4",
             "# in the azimuthal average of the growth-stripped product S_a S_b.",
             "P22_TENSOR = {"]
    for key in sorted(p22):
        lines.append(f"    {key!r}: {frac_repr(p22[key])},")
    lines += ["}", "",
              "# P22_UV_K2[(m, r)] = (a_R0, a_S): I_mr(true) - I_mr(continued) = k^2/(2 pi^2) (a_R0 R0 + a_S S),",
              "# R0 = int dq P^2, S = int dq q^2 (dP/dq)^2.",
              "P22_UV_K2 = {"]
    for key in P22_BASIS:
        a, b = uv22[key]
        lines.append(f"    {key!r}: ({frac_repr(a)}, {frac_repr(b)}),")
    lines += ["}", "", "# J_UV[n]: J_n(true) - J_n(continued) = J_UV[n] k^2 sigma_{-1}^2", f"J_UV = [{', '.join(frac_repr(x) for x in dJ)}]", ""]

    def arr_repr(arr):
        if isinstance(arr, np.ndarray) and arr.dtype == object and arr.ndim > 0:
            return "[" + ", ".join(arr_repr(x) for x in arr) + "]"
        return frac_repr(arr)
    lines += ["CUBIC_OPERATORS = " + repr(list(load_oracle().DJS_CUBIC_REDUCED_ORDER)),
              f"CUBIC_NF = {CUBIC_NF!r}",
              "# CUBIC_M[a][n][i]: growth-stripped single 13 contraction <delta_L O_a>' / P_L = sum_n mu^2n M[a][n] . J",
              "CUBIC_M = " + arr_repr(Mred),
              "# bare 1/x^2 (k^2 sigma_{-1}^2) and constant (sigma_0^2) coefficients of each reduced operator, [mu^0..mu^6]",
              "CUBIC_D = " + arr_repr(Dred),
              "CUBIC_C = " + arr_repr(Cred), ""]
    lines += [f"REST_KEYS = {REST_KEYS + ['matter']!r}",
              "# BRIDGE_B[key][p][n]: bare k^2 sigma_{-1}^2 coefficient of 3 int Z3 P for the exact map of a",
              "# rest-frame density, per rest-frame coefficient (and the matter part), coefficient of f^p mu^(2n).",
              "BRIDGE_D = {"]
    for key in REST_KEYS + ["matter"]:
        rat, _ = rationalize(bridgeB[key][:, :, 5])
        lines.append(f"    {key!r}: {arr_repr(rat)},")
    lines += ["}", "BRIDGE_C = {"]
    for key in REST_KEYS + ["matter"]:
        rat, _ = rationalize(bridgeB[key][:, :, 6])
        lines.append(f"    {key!r}: {arr_repr(rat)},")
    lines += ["}", ""]
    text = "\n".join(lines)
    Path(path).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_PATH))
    ap.add_argument("--report", default=str(REPORT_PATH))
    ap.add_argument("--check", action="store_true", help="write to a temporary file and compare with --out")
    ap.add_argument("--resume-dir", default=None, help="directory for intermediate decompositions (reused if present)")
    args = ap.parse_args()
    global CACHE_DIR
    if args.resume_dir:
        os.makedirs(args.resume_dir, exist_ok=True)
        CACHE_DIR = args.resume_dir
    report = {"generator": GENERATOR_VERSION, "cubic": {}, "bridge": {}, "checks": {}}
    t0 = time.time()
    p22 = stage_p22()
    report["checks"]["p22_terms"] = len(p22)
    print(f"[p22] {len(p22)} nonzero rational coefficients ({time.time() - t0:.0f}s)", flush=True)
    uv22, dJ = stage_uv()
    check_p22_classes(p22, uv22, report)
    report["checks"]["J_UV"] = [str(x) for x in dJ]
    print(f"[uv] dJ = {[str(x) for x in dJ]}", flush=True)
    ident = Identifier()
    Mred, Dred, Cred = stage_cubic(ident, report)
    bridge = stage_bridge(ident, report)
    check_bridge(Mred, bridge, report)
    check_bridge_closed_form(bridge, report)
    # (A) - (B) only derivative/constant shapes: their J parts agree (checked); record the k^2 parts
    out = args.out
    if args.check:
        out = str(Path(args.out).with_suffix(".check.py"))
    sha = write_data(p22, uv22, dJ, Mred, Dred, Cred, bridge["B"], out)
    report["output_sha256"] = sha
    report["seconds"] = round(time.time() - t0, 1)
    Path(args.report).write_text(json.dumps(report, indent=1, default=str))
    if args.check:
        same = Path(out).read_text() == Path(args.out).read_text()
        print("[check] regenerated data identical to committed:", same)
        os.remove(out)
        sys.exit(0 if same else 1)
    print("wrote", out, "sha256", sha)


if __name__ == "__main__":
    main()
