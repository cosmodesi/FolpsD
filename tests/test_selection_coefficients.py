"""Generated coefficient data: independent re-derivations and external comparisons."""
from fractions import Fraction as Fr

import numpy as np
import pytest

from folps import selection_coefficients as sc
from folps import selection_kernels as sk
from folps.selection_operators import SelectionBias, djs_coefficients, quadratic_coefficient_vector
from folps.selection_reference import mapped_field

# GeneralBiasPk (https://github.com/JosephTomlinson/GeneralBiasPk) P13/M13.jl at commit
# d4d5bee2477d6efd719a4c7193fef0261d88334f, transcribed (rows mu^0, mu^2, mu^4; columns J1..J5),
# in the order of sc.CUBIC_OPERATORS.  An external comparison, not a gold standard.
_GB = {
    "Otd": [[4, -6, 2, 0, 0], [0] * 5, [0] * 5],
    "dPi2": [[0, 0, 5, -5, 0], [0, 0, -15, 15, 0], [0] * 5],
    "ePi2": [[0, 0, Fr(-15, 4), Fr(15, 2), Fr(-15, 4)], [-5, Fr(15, 2), 20, -60, Fr(75, 2)],
             [5, Fr(-15, 2), Fr(-65, 4), Fr(125, 2), Fr(-175, 4)]],
    "Pi2K": [[Fr(5, 4), Fr(-15, 8), Fr(35, 24), Fr(-5, 6), 0], [Fr(5, 4), Fr(-15, 8), Fr(-15, 8), Fr(5, 2), 0], [0] * 5],
    "u2dd": [[0, 0, 3, -3, 0], [0, -3, -6, 9, 0], [0] * 5],
    "u2de": [[0, 0, Fr(-9, 4), Fr(9, 2), Fr(-9, 4)], [Fr(-9, 4), Fr(27, 8), Fr(63, 8), Fr(-63, 2), Fr(45, 2)],
             [Fr(15, 4), Fr(-21, 8), Fr(-39, 8), 30, Fr(-105, 4)]],
    "sdPi2": [[Fr(5, 4), Fr(-15, 8), Fr(25, 8), Fr(-5, 2), 0], [Fr(-15, 4), Fr(45, 8), Fr(-75, 8), Fr(15, 2), 0], [0] * 5],
    "udPi2": [[0, 0, Fr(15, 4), Fr(-15, 2), Fr(15, 4)], [Fr(15, 4), Fr(-45, 8), Fr(-225, 8), Fr(135, 2), Fr(-75, 2)],
              [Fr(-25, 4), Fr(75, 8), Fr(225, 8), -75, Fr(175, 4)]],
    "Pi3": [[Fr(13, 8), Fr(-39, 16), Fr(65, 16), Fr(-13, 4), 0], [Fr(-101, 24), Fr(101, 16), Fr(-569, 48), Fr(39, 4), 0], [0] * 5],
    "d3": [[Fr(2, 3), Fr(1, 2), Fr(-7, 6), 0, 0], [0] * 5, [0] * 5],
    "e3": [[0] * 5, [-2, Fr(3, 2), Fr(1, 2), 0, 0], [0] * 5],
}


def _gb(name, factor=1):
    return np.array([[float(Fr(v) * factor / 7) for v in row] for row in _GB[name]])


GENERALBIASPK = [  # same order as sc.CUBIC_OPERATORS
    _gb("d3"), _gb("e3"), _gb("Otd", Fr(5, 2)), _gb("dPi2", Fr(-3, 5)), _gb("ePi2", Fr(-6, 5)),
    _gb("Pi2K", 2), _gb("Otd"), _gb("dPi2"), _gb("ePi2"), _gb("Pi2K"), _gb("sdPi2"), _gb("u2dd"),
    _gb("u2de"), _gb("udPi2", Fr(-3, 5)), _gb("udPi2"), _gb("Pi3"),
]


def test_metadata_and_shapes():
    assert len(sc.P22_BASIS) == 23
    assert set(sc.P22_BASIS) == {(m, r) for r in range(7) for m in range(r + 3) if (r - m) % 2 == 0}
    assert sc.P22_T.shape == (10, 10, 5, 23)
    assert np.allclose(sc.P22_T, np.swapaxes(sc.P22_T, 0, 1))
    assert sc.CUBIC_M.shape == (16, 3, 5)
    assert len(sc.COEFFICIENT_DATA_SHA256) == 64


def test_cubic_matrices_match_generalbiaspk():
    for a, ref in enumerate(GENERALBIASPK):
        assert np.allclose(sc.CUBIC_M[a], ref, rtol=0, atol=1e-14), sc.CUBIC_OPERATORS[a]


def test_eta_row_correction_of_printed_F7():
    """Adopted: M[f^-1 eta^(3)]_{mu^2} = (-2, 3/2, 1/2, 0, 0)/7; printed DJS F.7: (2, 3/2, -1/2)/7."""
    a = sc.CUBIC_OPERATORS.index("eta^(3)")
    assert np.allclose(sc.CUBIC_M[a, 1], np.array([-2, 1.5, 0.5, 0, 0]) / 7, atol=1e-15)
    assert not np.allclose(sc.CUBIC_M[a, 1], np.array([2, 1.5, -0.5, 0, 0]) / 7)
    assert np.allclose(sc.CUBIC_M[a, [0, 2]], 0)
    d = sc.CUBIC_OPERATORS.index("delta^(3)")
    assert np.allclose(sc.CUBIC_M[d, 0], [2 / 21, 1 / 14, -1 / 6, 0, 0], atol=1e-15)


def test_collapsed_F3_G3_identities_on_toy_spectrum():
    """3 int F3(k,q,-q) P = 2/21 J1 + 1/14 J2 - 1/6 J3 - 1/6 k^2 sigma_-1^2 (and the G3 analogue)
    by direct quadrature on the smooth toy spectrum P = q exp(-q^2) (audit check)."""
    q = np.exp(np.linspace(np.log(1e-5), np.log(12.0), 400))
    w = np.gradient(np.log(q)) * q          # dq
    P = q * np.exp(-q ** 2)
    xg, wg = np.polynomial.legendre.leggauss(180)
    for k in (0.12, 0.4, 1.0):
        Q2 = k * k + q[:, None] ** 2 - 2 * k * q[:, None] * xg[None, :]
        nu = xg[None, :]
        avg = lambda a: 0.5 * np.sum(wg[None, :] * a, axis=1)  # noqa: E731
        J = [k ** 2 * avg(k * q[:, None] * nu * (1 - nu ** 2) / (q[:, None] ** 2 * Q2)),
             k ** 2 * avg((1 - nu ** 2) / Q2), avg(q[:, None] ** 2 * (1 - nu ** 2) / Q2) - 2 / 3]
        J = [np.sum(w * q ** 2 / (2 * np.pi ** 2) * j * P) for j in J]
        s2 = np.sum(w * P) / (2 * np.pi ** 2)
        # direct 3 int F3 / G3 with the oracle (k along z, isotropic -> LOS irrelevant)
        nphi = 8
        phi = 2 * np.pi * np.arange(nphi) / nphi
        st = np.sqrt(1 - xg ** 2)
        tot_F = tot_G = 0.0
        for i in range(0, len(q), 50):
            qq = q[i:i + 50]
            qv = qq[:, None, None, None] * np.stack([st[:, None] * np.cos(phi), st[:, None] * np.sin(phi),
                                                      np.broadcast_to(xg[:, None], (180, nphi))], -1)[None]
            kv = np.broadcast_to([0.0, 0.0, k], qv.shape)
            F3, G3 = sk.Evaluator([kv, qv, -qv], np.array([0, 0, 1.0]), 1.0).FG_sym((0, 1, 2))
            aF = np.sum(wg[:, None] * F3, axis=(1, 2)) / (2 * nphi)
            aG = np.sum(wg[:, None] * G3, axis=(1, 2)) / (2 * nphi)
            sl = slice(i, i + 50)
            tot_F += np.sum(w[sl] * qq ** 2 / (2 * np.pi ** 2) * 3 * aF * P[sl])
            tot_G += np.sum(w[sl] * qq ** 2 / (2 * np.pi ** 2) * 3 * aG * P[sl])
        rF = 2 / 21 * J[0] + 1 / 14 * J[1] - 1 / 6 * J[2] - k * k * s2 / 6
        rG = 2 / 7 * J[0] - 3 / 14 * J[1] - 1 / 14 * J[2] - k * k * s2 / 6
        assert abs(tot_F - rF) < 1e-14 * max(1, abs(rF)) + 1e-15
        assert abs(tot_G - rG) < 1e-14 * max(1, abs(rG)) + 1e-15


def test_p22_tensor_reproduces_oracle_Z2_squared():
    rng = np.random.default_rng(3)
    f = 0.81
    sel = SelectionBias(s_eta=0.3, s_delta_eta=0.4, s_eta2=-0.35, s_KK_par=0.45, s_Pi2_par=-0.55)
    c = djs_coefficients(1.6, -0.5, 0.2, 0.0, sel)
    c2 = np.array(quadratic_coefficient_vector(c))
    Z = mapped_field(c._asdict())
    nphi = 64
    phi = 2 * np.pi * np.arange(nphi) / nphi
    fp = f ** np.add.outer(sc.QUADRATIC_NF, sc.QUADRATIC_NF)
    W = np.einsum("a,b,ab,abnj->nj", c2, c2, fp, sc.P22_T)
    for _ in range(25):
        x, nu, mu = np.exp(rng.uniform(np.log(0.05), np.log(8))), rng.uniform(-1, 1), rng.uniform(-1, 1)
        st = np.sqrt(1 - nu ** 2)
        q = x * np.stack([st * np.cos(phi), st * np.sin(phi), nu * np.ones(nphi)], -1)
        k = np.broadcast_to([0, 0, 1.0], q.shape)
        lhs = np.mean(Z.at(sk.Evaluator([q, k - q], np.array([np.sqrt(1 - mu ** 2), 0, mu]), f), 2).real ** 2)
        Q = np.sqrt(1 + x * x - 2 * x * nu)
        basis = np.array([nu ** m * x ** (r - 2) / Q ** 4 for (m, r) in sc.P22_BASIS])
        rhs = np.sum(W * np.array([mu ** (2 * n) for n in range(5)])[:, None] * basis[None, :])
        assert abs(lhs - rhs) < 1e-11 * abs(lhs)


def test_p22_pair_classes_have_no_forbidden_uv_content():
    """Pairs with a soft-singular kernel: no UV constant; singular x singular: no UV k^2."""
    const_w = np.array([2.0 / (m + 1) if r == 6 else 0.0 for (m, r) in sc.P22_BASIS])
    for a in range(10):
        for b in range(10):
            sa, sb = a in sc.QUADRATIC_SINGULAR, b in sc.QUADRATIC_SINGULAR
            if sa or sb:
                assert np.allclose(sc.P22_T[a, b] @ const_w, 0, atol=1e-13)
            if sa and sb:
                assert np.allclose(sc.P22_T[a, b] @ sc.P22_UV, 0, atol=1e-12)


def test_J_uv_coefficients():
    assert np.allclose(sc.J_UV, [0, 2 / 3, -2 / 15, 2 / 21, 22 / 315], atol=1e-15)


def test_bare_bridge_closed_form():
    """Bare k^2 sigma_-1^2 coefficient of the exact map: -1/6 [1 + f mu^2 (2+f)] Z1."""
    S = np.zeros_like(sc.BRIDGE_D["matter"])
    S[0, 0], S[1, 1], S[2, 1] = 1, 2, 1
    up = np.zeros_like(S)
    up[1:, 1:] = S[:-1, :-1]
    assert np.allclose(sc.BRIDGE_D["b1"], -S / 6, atol=1e-14)
    assert np.allclose(sc.BRIDGE_D["bR_eta"], up / 6, atol=1e-14)
    assert np.allclose(sc.BRIDGE_D["matter"], -up / 6, atol=1e-14)
    for key in sc.REST_KEYS:
        if key not in ("b1", "bR_eta", "matter"):
            assert np.allclose(sc.BRIDGE_D[key], 0, atol=1e-14), key


@pytest.mark.slow
def test_reidentify_selected_cubic_rows():
    """Independent re-identification of three operators on a different grid."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "gen", Path(__file__).resolve().parents[1] / "scripts" / "generate_selection_coefficients.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    ident = gen.Identifier(nnu=140, nphi=12, nx=22, mus=[0.0, 0.35, 0.7, 0.85, 1.0])
    ops = sk.reduced_cubic_operator_fields()
    for name in ("eta^(3)", "u_par d_par eta^(2)", "Pi^[3]_par"):
        a = sc.CUBIC_OPERATORS.index(name)
        coef, res, _ = gen.ident_decompose_fixed_f(ident, ops[a], sc.CUBIC_NF[a])
        assert res < 1e-11
        assert np.allclose(coef[:3, :5], sc.CUBIC_M[a], atol=1e-9)
