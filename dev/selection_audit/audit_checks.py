#!/usr/bin/env python
"""Independent equation-audit checks for the one-loop LOS-selection specification.

Provenance and status
---------------------
The pre-implementation audit (2026-10-26, "folps_orientation_selection_1loop_equations_audited"
and the "Pre-implementation equation audit" section of the Claude brief) refers to a
companion script ``audit_checks.py`` and its output ``audit_results.json``.  That script was
*not* supplied with the brief, so this file reconstructs the listed checks from their
descriptions in the audit, attributing the checks themselves to that review.

**This is not the production implementation.**  It deliberately does not import FolpsD: it
re-implements the EdS F_n/G_n recursion, normalized symmetrization and finite-domain
Gauss-Legendre quadrature in a few lines of NumPy, so that it can serve as an independent
cross-check of the equations (not of FolpsD's numerics).

Checks (as described in the audit):
 1. standard redshift-space Z2 vs the ten-kernel DJS contraction (300 random triangles);
 2. constant-f linear bA conversion (300 configurations);
 3. Pi^[2] recursion vs the explicit tensor formula (300 triangles);
 4. factored derivative sector vs its four polynomial coefficients (300 configurations);
 5. collapsed F3 and G3 single-contraction identities at k = 0.12, 0.4, 1 for the toy
    spectrum P(q) = q exp(-q^2), 1e-5 <= q <= 12, 200 log-radial and 180 angular nodes;
 6. the eta-row discrepancy: reduced eta/(f mu^2) response at k = 0.4 from the integrated G3
    (audit value 0.000382899420985) vs the printed DJS F.7 row (0.000617892107699).

Usage::

    python dev/selection_audit/audit_checks.py [--out audit_results.json]
"""
import argparse
import itertools
import json

import numpy as np

RNG = np.random.default_rng(1806040152)


def dot(a, b):
    return np.sum(a * b, axis=-1)


def fg_unsym(ks):
    """Unsymmetrized EdS (F_n, G_n) for an ordered list of momenta (zero-sum blocks -> 0)."""
    n = len(ks)
    if n == 1:
        one = np.ones(ks[0].shape[:-1])
        return one, one
    F = G = 0.0
    for m in range(1, n):
        p, r = sum(ks[:m]), sum(ks[m:])
        p2, r2 = dot(p, p), dot(r, r)
        dead = (p2 < 1e-28) | (r2 < 1e-28)
        sp, sr = np.where(dead, 1.0, p2), np.where(dead, 1.0, r2)
        alpha = dot(p + r, p) / sp
        beta = dot(p + r, p + r) * dot(p, r) / (2 * sp * sr)
        Gm = fg_unsym(ks[:m])[1]
        Fr, Gr = fg_unsym(ks[m:])
        pre = Gm / ((2 * n + 3) * (n - 1))
        F = F + np.where(dead, 0.0, pre * ((2 * n + 1) * alpha * Fr + 2 * beta * Gr))
        G = G + np.where(dead, 0.0, pre * (3 * alpha * Fr + 2 * n * beta * Gr))
    return F, G


def fg_sym(ks):
    perms = list(itertools.permutations(range(len(ks))))
    F = sum(fg_unsym([ks[i] for i in p])[0] for p in perms) / len(perms)
    G = sum(fg_unsym([ks[i] for i in p])[1] for p in perms) / len(perms)
    return F, G


def rand_vec(n):
    v = RNG.normal(size=(n, 3))
    return v * np.exp(RNG.uniform(-1, 1, (n, 1)))


def check_z2(n=300):
    k1, k2 = rand_vec(n), rand_vec(n)
    nl = RNG.normal(size=(n, 3))
    nl /= np.linalg.norm(nl, axis=1, keepdims=True)
    f, b1, b2, bK2 = 0.7, 1.8, -0.4, -0.2
    k1n, k2n = np.linalg.norm(k1, axis=1), np.linalg.norm(k2, axis=1)
    K = k1 + k2
    Kn = np.linalg.norm(K, axis=1)
    mu1, mu2, muK = dot(k1, nl) / k1n, dot(k2, nl) / k2n, dot(K, nl) / Kn
    nu, r = dot(k1, k2) / (k1n * k2n), k1n / k2n
    F2, G2 = fg_sym([k1, k2])
    S = [F2, -f * muK ** 2 * G2, 1.0, nu ** 2 - 1 / 3, -f * (mu1 ** 2 + mu2 ** 2) / 2, f ** 2 * mu1 ** 2 * mu2 ** 2,
         mu1 * mu2 * nu - (mu1 ** 2 + mu2 ** 2) / 3 + 1 / 9, mu1 * mu2 * nu + 5 / 7 * muK ** 2 * (1 - nu ** 2),
         -f * mu1 * mu2 * (r + 1 / r) / 2, f ** 2 * mu1 * mu2 * (mu1 ** 2 * r + mu2 ** 2 / r) / 2]
    b_eta, b_deta, b_eta2 = -1.0, -b1, 1.0
    c2 = [b1, b_eta, b2 / 2, bK2, b_deta, b_eta2, 0.0, 0.0, -b1, -b_eta]
    Z2 = sum(c * s for c, s in zip(c2, S))
    Zstd = (b1 * F2 + f * muK ** 2 * G2 + b2 / 2 + bK2 * (nu ** 2 - 1 / 3)
            + f * Kn * muK / 2 * (mu1 / k1n * (b1 + f * mu2 ** 2) + mu2 / k2n * (b1 + f * mu1 ** 2)))
    return float(np.max(np.abs(Z2 - Zstd)))


def check_bA(n=300):
    mu = RNG.uniform(-1, 1, n)
    b1F, bA, f = RNG.uniform(1, 2.5), RNG.uniform(-0.5, 0.5), RNG.uniform(0.4, 1.0)
    b1D, b_eta = b1F - bA / 3, -1 - bA / f
    return float(np.max(np.abs(b1F + f * mu ** 2 + bA * (mu ** 2 - 1 / 3) - (b1D - b_eta * f * mu ** 2))))


def check_pi2(n=300):
    k1, k2 = rand_vec(n), rand_vec(n)
    R = lambda v: v[:, :, None] * v[:, None, :] / dot(v, v)[:, None, None]  # noqa: E731
    F2, _ = fg_sym([k1, k2])
    # recursion T^[2](2) = R(K) F2 - Sym[(p.r/p^2) G1 T^[1](1)]
    T = R(k1 + k2) * F2[:, None, None] - 0.5 * (dot(k1, k2)[:, None, None] / dot(k1, k1)[:, None, None] * R(k2)
                                               + dot(k1, k2)[:, None, None] / dot(k2, k2)[:, None, None] * R(k1))
    nu = dot(k1, k2) / np.sqrt(dot(k1, k1) * dot(k2, k2))
    explicit = 0.5 * (R(k1) @ R(k2) + R(k2) @ R(k1)) + 10 / 21 * R(k1 + k2) * (1 - 1.5 * (nu ** 2 - 1 / 3))[:, None, None]
    return float(np.max(np.abs(T - explicit)))


def check_derivative(n=300):
    mu = RNG.uniform(-1, 1, n)
    b1, b_eta, f = 1.7, -1.3, 0.75
    d, v, w = 1.1, -0.6, 0.4
    a = b_eta * f
    direct = -2 * (b1 - a * mu ** 2) * (d - a * mu ** 2 * (v + w * mu ** 2))
    c = (-2 * b1 * d, 2 * a * (d + b1 * v), 2 * b1 * a * w - 2 * a * a * v, -2 * a * a * w)
    poly = c[0] + c[1] * mu ** 2 + c[2] * mu ** 4 + c[3] * mu ** 6
    c6 = -a / b1 * c[2] - a ** 2 / b1 ** 2 * c[1] - a ** 3 / b1 ** 3 * c[0]
    return float(max(np.max(np.abs(direct - poly)), abs(c6 - c[3])))


def collapsed(nq=200, nang=180):
    q = np.exp(np.linspace(np.log(1e-5), np.log(12.0), nq))
    lw = np.gradient(np.log(q))
    P = q * np.exp(-q ** 2)
    x, wx = np.polynomial.legendre.leggauss(nang)
    out = {}
    for k in (0.12, 0.4, 1.0):
        kv = np.array([0.0, 0.0, k])
        st = np.sqrt(1 - x ** 2)
        qv = q[:, None, None] * np.stack([st, np.zeros_like(x), x], -1)[None]   # azimuth irrelevant (isotropic)
        kk = np.broadcast_to(kv, qv.shape)
        F3, G3 = fg_sym([kk, qv, -qv])
        meas = q ** 3 * lw / (2 * np.pi ** 2) * P
        lhsF = 3 * np.sum(meas * 0.5 * np.sum(wx * F3, axis=-1))
        lhsG = 3 * np.sum(meas * 0.5 * np.sum(wx * G3, axis=-1))
        Q2 = k * k + q[:, None] ** 2 - 2 * k * q[:, None] * x[None]
        avg = lambda a: 0.5 * np.sum(wx * a, axis=-1)  # noqa: E731
        J1 = np.sum(meas * k ** 2 * avg(k * q[:, None] * x * (1 - x ** 2) / (q[:, None] ** 2 * Q2)))
        J2 = np.sum(meas * k ** 2 * avg((1 - x ** 2) / Q2))
        J3 = np.sum(meas * (avg(q[:, None] ** 2 * (1 - x ** 2) / Q2) - 2 / 3))
        s2 = np.sum(q * lw * P) / (2 * np.pi ** 2)
        rhsF = 2 / 21 * J1 + 1 / 14 * J2 - 1 / 6 * J3 - k * k * s2 / 6
        rhsG = 2 / 7 * J1 - 3 / 14 * J2 - 1 / 14 * J3 - k * k * s2 / 6
        out[str(k)] = dict(F3_abs_diff=float(abs(lhsF - rhsF)), G3_abs_diff=float(abs(lhsG - rhsG)),
                           eta_over_fmu2_from_G3=float(-(lhsG + k * k * s2 / 6)),
                           eta_over_fmu2_printed_F7=float((2 * J1 + 1.5 * J2 - 0.5 * J3) / 7),
                           eta_over_fmu2_adopted_row=float((-2 * J1 + 1.5 * J2 + 0.5 * J3) / 7))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="audit_results.json")
    args = ap.parse_args()
    res = dict(note="Reconstructed audit checks (not the production implementation).",
               Z2_standard_vs_ten_kernel_max_abs=check_z2(), bA_linear_identity_max_abs=check_bA(),
               Pi2_recursion_vs_explicit_max_abs=check_pi2(), derivative_sector_max_abs=check_derivative(),
               collapsed_identities=collapsed())
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
