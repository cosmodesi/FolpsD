"""Kernel and algebra identities of the LOS-selection oracle (no generated data used)."""
import numpy as np
import pytest

from folps import selection_kernels as sk
from folps.selection_operators import (PhysicalDerivativeBias, SelectionBias, djs_coefficients,
                                       physical_to_polynomial_derivatives, quadratic_coefficient_vector,
                                       tree_bA_to_linear_djs)
from folps.selection_reference import exact_map_field_from_djs, mapped_field

RNG = np.random.default_rng(20261026)
N = 200
F_GROWTH = 0.73


def _vecs(n=N, spread=1.5):
    v = RNG.normal(size=(n, 3))
    return v * np.exp(RNG.uniform(-spread, spread, size=(n, 1)))


@pytest.fixture(scope="module")
def config():
    k1, k2, k3 = _vecs(), _vecs(), _vecs()
    n = RNG.normal(size=(N, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return k1, k2, k3, n


@pytest.fixture(scope="module")
def coeffs():
    sel = SelectionBias(s_eta=0.31, s_delta_eta=-0.42, s_eta2=0.27, s_KK_par=0.5, s_Pi2_par=-0.6,
                        s_delta_Pi2_par=0.33, s_eta_Pi2_par=-0.21, s_KPi2_par=0.44, s_Pi3_par=0.17)
    return djs_coefficients(1.7, -0.4, -0.3, 0.25, sel)


def test_F2_G2_closed_forms(config):
    k1, k2, k3, n = config
    ev = sk.Evaluator([k1, k2, k3], n, F_GROWTH)
    F2, G2 = ev.FG_sym((0, 1))
    assert np.max(np.abs(F2 - sk.F2_closed_form(k1, k2))) < 1e-13
    assert np.max(np.abs(G2 - sk.G2_closed_form(k1, k2))) < 1e-13


def test_F3_G3_symmetric_and_momentum_conservation(config):
    k1, k2, k3, n = config
    ev = sk.Evaluator([k1, k2, k3], n, 1.0)
    ev2 = sk.Evaluator([k3, k1, k2], n, 1.0)
    for a, b in zip(ev.FG_sym((0, 1, 2)), ev2.FG_sym((0, 1, 2))):
        assert np.max(np.abs(a - b)) < 1e-12
    # F3 -> 0 when the total momentum -> 0 (scales as K^2)
    eps = 1e-4
    kz = -(k1 + k2) + eps * k3
    F3, G3 = sk.Evaluator([k1, k2, kz], n, 1.0).FG_sym((0, 1, 2))
    scale = np.sum((k1 + k2) ** 2, axis=-1) / np.sum(k3 ** 2, axis=-1)
    assert np.max(np.abs(F3) / (eps ** 2 * (1 + 1 / scale) * 50)) < 1


def test_pi2_explicit_tensor_identity(config):
    k1, k2, _, n = config
    ev = sk.Evaluator([k1, k2], n, F_GROWTH)
    T2 = ev.T(2, (0, 1))
    R1, R2, RK = ev.R((0,)), ev.R((1,)), ev.R((0, 1))
    nu = np.sum(k1 * k2, -1) / np.linalg.norm(k1, axis=1) / np.linalg.norm(k2, axis=1)
    rhs = 0.5 * (R1 @ R2 + R2 @ R1) + 10 / 21 * RK * (1 - 1.5 * (nu ** 2 - 1 / 3))[:, None, None]
    assert np.max(np.abs(T2 - rhs)) < 1e-13


def test_pi_hierarchy_traces_against_fluid_equations(config):
    """tr Pi^[2] = vartheta - delta + delta vartheta (continuity) and an independently derived
    Euler/continuity identity for tr Pi^[3], both through third order."""
    k1, k2, k3, n = config
    ev = sk.Evaluator([k1, k2, k3], n, 1.0)
    d, th = sk.density(), sk.theta()
    dth = sk.product(d, th)
    for idx in ((0, 1), (0, 1, 2)):
        lhs = np.trace(ev.T(2, idx), axis1=-2, axis2=-1)
        rhs = (th(ev, idx) - d(ev, idx) + dth(ev, idx)).real
        assert np.max(np.abs(lhs - rhs)) < 1e-12
    Pth = sk.Field(lambda ev, i: (ev.R(i) * ev.FG_sym(i)[1][..., None, None]).astype(complex), rank=2)
    Q = sk.trace_product(Pth, Pth)
    th2, d2 = sk.product(th, th), sk.product(d, d)
    idx = (0, 1, 2)
    rhs3 = (7 / 4 * (d(ev, idx) - th(ev, idx)) + Q(ev, idx) / 2 + th2(ev, idx) / 2 + 0.75 * d2(ev, idx)
            - 7 / 4 * dth(ev, idx) + 0.5 * sk.product(d, th2)(ev, idx) + 0.5 * sk.product(d, Q)(ev, idx)).real
    assert np.max(np.abs(np.trace(ev.T(3, idx), axis1=-2, axis2=-1) - rhs3)) < 1e-12
    assert np.max(np.abs(ev.T(3, (0, 1)))) == 0.0


def test_quadratic_table_kernels_and_standard_Z2(config, coeffs):
    k1, k2, k3, n = config
    ev = sk.Evaluator([k1, k2, k3], n, F_GROWTH)
    ZA = mapped_field(coeffs._asdict()).at(ev, 2)
    assert np.max(np.abs(ZA.imag)) == 0
    S = sk.quadratic_table_kernels(k1, k2, n, F_GROWTH)
    c2 = np.array(quadratic_coefficient_vector(coeffs, galilean_safe=False))
    assert np.max(np.abs(ZA.real - np.einsum("a,a...->...", c2, S))) < 1e-12
    # Galilean-safe basis is the same field
    S_safe = S.copy()
    S_safe[8] = S[8] + S[4]
    S_safe[9] = S[9] + S[5]
    c2s = np.array(quadratic_coefficient_vector(coeffs, galilean_safe=True))
    assert np.max(np.abs(ZA.real - np.einsum("a,a...->...", c2s, S_safe))) < 1e-12
    # zero selection -> standard redshift-space Z2
    c0 = djs_coefficients(1.7, -0.4, -0.3, 0.25, SelectionBias())
    Z0 = mapped_field(c0._asdict()).at(ev, 2).real
    k1n, k2n = np.linalg.norm(k1, axis=1), np.linalg.norm(k2, axis=1)
    K = k1 + k2
    Kn = np.linalg.norm(K, axis=1)
    mu1, mu2, muK = (np.sum(k1 * n, -1) / k1n, np.sum(k2 * n, -1) / k2n, np.sum(K * n, -1) / Kn)
    nu = np.sum(k1 * k2, -1) / k1n / k2n
    f = F_GROWTH
    Zstd = (1.7 * sk.F2_closed_form(k1, k2) + f * muK ** 2 * sk.G2_closed_form(k1, k2) - 0.2
            - 0.3 * (nu ** 2 - 1 / 3) + f * Kn * muK / 2 * (mu1 / k1n * (1.7 + f * mu2 ** 2) + mu2 / k2n * (1.7 + f * mu1 ** 2)))
    assert np.max(np.abs(Z0 - Zstd)) < 1e-12


def test_exact_map_equals_mapped_field_through_second_order(config, coeffs):
    k1, k2, k3, n = config
    ev = sk.Evaluator([k1, k2, k3], n, F_GROWTH)
    ZA, ZB = mapped_field(coeffs._asdict()), exact_map_field_from_djs(coeffs._asdict())
    for order in (1, 2):
        assert np.max(np.abs(ZA.at(ev, order) - ZB.at(ev, order))) < 1e-12


def test_permutation_and_los_reversal(config, coeffs):
    k1, k2, k3, n = config
    ZA = mapped_field(coeffs._asdict())
    ZB = exact_map_field_from_djs(coeffs._asdict())
    for Z in (ZA, ZB):
        z3 = Z.at(sk.Evaluator([k1, k2, k3], n, F_GROWTH), 3)
        assert np.max(np.abs(z3.imag)) < 1e-12
        z3p = Z.at(sk.Evaluator([k3, k1, k2], -n, F_GROWTH), 3)
        assert np.max(np.abs(z3p - z3)) < 1e-11 * max(1, np.max(np.abs(z3)))


def test_growth_powers_of_selection_kernels(config):
    """Each quadratic kernel scales as f^{n_f}; the cubic reduced operators likewise."""
    k1, k2, k3, n = config
    nf2 = [0, 1, 0, 0, 1, 2, 0, 0, 1, 2]
    S1 = sk.quadratic_table_kernels(k1, k2, n, 1.0)
    S2 = sk.quadratic_table_kernels(k1, k2, n, 0.4)
    for a in range(10):
        assert np.allclose(S2[a], 0.4 ** nf2[a] * S1[a], rtol=1e-12, atol=1e-14)
    nf3 = [0, 1, 0, 1, 2, 0, 0, 0, 1, 0, 0, 1, 2, 2, 1, 0]
    ops = sk.reduced_cubic_operator_fields()
    e1, e2 = sk.Evaluator([k1, k2, k3], n, 1.0), sk.Evaluator([k1, k2, k3], n, 0.4)
    for a, op in enumerate(ops):
        v1, v2 = op.at(e1, 3), op.at(e2, 3)
        assert np.allclose(v2, 0.4 ** nf3[a] * v1, rtol=1e-11, atol=1e-13)


def test_full_spatial_advection_not_los_projection(config):
    """s^i d_i Pi^[2]_par differs from s_par d_par Pi^[2]_par (audit A01)."""
    k1, k2, k3, n = config
    ev = sk.Evaluator([k1, k2, k3], n, 1.0)
    full = sk.reduced_cubic_operator_fields()[10].at(ev, 3)
    s1 = sk.displacement_s1()
    P2par = sk.par(sk.pi_tensor(2))
    s_par = sk.Field(lambda ev, idx: np.einsum("...i,...i->...", s1(ev, idx), ev.n), name="s_par")
    los_only = sk.truncate_order(sk.product(s_par, sk.grad_par(P2par)), (3,)).at(ev, 3)
    assert np.max(np.abs(full - los_only)) > 1e-2 * np.max(np.abs(full))


def test_collapsed_limit_is_continuous():
    """Z3(k, q, -q + eps) -> Z3(k, q, -q) (exact collapsed evaluation) as eps -> 0."""
    k = np.array([[0.0, 0.0, 1.0]])
    q = np.array([[0.3, 0.2, 0.5]])
    n = np.array([0.6, 0.0, 0.8])
    sel = SelectionBias(*([0.3] * 9))
    c = djs_coefficients(1.5, -0.3, 0.2, 0.1, sel)
    Z = exact_map_field_from_djs(c._asdict())
    exact = Z.at(sk.Evaluator([k, q, -q], n, 0.8), 3)
    d = np.array([[0.13, -0.41, 0.22]])
    prev = None
    for eps in (1e-3, 1e-4, 1e-5):
        # average over +/- eps to cancel the linear term in the approach direction
        vals = [Z.at(sk.Evaluator([k, q, -q + s * eps * d], n, 0.8), 3) for s in (1, -1)]
        err = abs(0.5 * (vals[0] + vals[1]) - exact)[0]
        if prev is not None:
            assert err < 0.05 * prev + 1e-10
        prev = err
    assert prev < 1e-8


def test_tree_bA_linear_identity():
    mu = RNG.uniform(-1, 1, 300)
    b1F, bA, f = 1.7, 0.35, 0.68
    b1D, s_eta = tree_bA_to_linear_djs(b1F, bA, f)
    lhs = b1F + f * mu ** 2 + bA * (mu ** 2 - 1 / 3)
    rhs = b1D - (-1 + s_eta) * f * mu ** 2
    assert np.max(np.abs(lhs - rhs)) < 1e-14


def test_physical_derivative_conversion_and_mu6_relation():
    mu = RNG.uniform(-1, 1, 300)
    b1, b_eta, f = 1.6, -1.2, 0.7
    phys = PhysicalDerivativeBias(b_nabla2delta=2.1, beta_nabla2v=-0.7, beta_par2v=1.3)
    c0, c2, c4, c6 = physical_to_polynomial_derivatives(b1, b_eta, f, phys)
    a = b_eta * f
    direct = -2 * (b1 - a * mu ** 2) * (phys.b_nabla2delta - a * mu ** 2 * (phys.beta_nabla2v + phys.beta_par2v * mu ** 2))
    assert np.max(np.abs(direct - (c0 + c2 * mu ** 2 + c4 * mu ** 4 + c6 * mu ** 6))) < 1e-13
    # three physical coefficients: c6 is fixed by the other three
    assert abs(c6 - (-a / b1 * c4 - a ** 2 / b1 ** 2 * c2 - a ** 3 / b1 ** 3 * c0)) < 1e-12


def test_ir_soft_limit_cancellation_exact_map():
    """q^2 [4 <Z2(q, k-q)^2> + 6 Z1 <Z3(k, q, -q)>] -> 0 as q -> 0 for generic selection with
    the exact redshift-space map (and fails for the reduced mapped field)."""
    f = 0.7
    xg, wg = np.polynomial.legendre.leggauss(48)
    nphi = 16
    phi = 2 * np.pi * np.arange(nphi) / nphi
    NU, PH = np.meshgrid(xg, phi, indexing="ij")
    st = np.sqrt(1 - NU ** 2)
    sel = SelectionBias(*([0.25, -0.35, 0.2, 0.5, -0.6, 0.3, -0.2, 0.4, 0.2]))
    c = djs_coefficients(1.6, -0.5, -0.2, 0.1, sel)

    def soft(field, mu, x):
        n = np.array([np.sqrt(1 - mu ** 2), 0, mu])
        qv = x * np.stack([st * np.cos(PH), st * np.sin(PH), NU], -1)
        kv = np.broadcast_to([0, 0, 1.0], qv.shape)
        Z2 = field.at(sk.Evaluator([qv, kv - qv], n, f), 2).real
        Z3 = field.at(sk.Evaluator([kv, qv, -qv], n, f), 3).real
        avg = lambda a: 0.5 * np.sum(wg[:, None] * a) / nphi  # noqa: E731
        Z1 = c.b1 - c.b_eta * f * mu ** 2
        return x ** 2 * 4 * avg(Z2 ** 2), x ** 2 * 6 * Z1 * avg(Z3)
    B = exact_map_field_from_djs(c._asdict())
    A = mapped_field(c._asdict())
    for mu in (0.3, 0.9):
        a2, b2 = soft(B, mu, 1e-3)
        a3, b3 = soft(B, mu, 1e-4)
        assert abs(a3 + b3) < 1e-6 * abs(a3)
        assert abs(a3 + b3) < 0.02 * abs(a2 + b2)      # residual ~ (q/k)^2
        aA, bA_ = soft(A, mu, 1e-4)
        assert abs(aA + bA_) > 1e-2 * abs(aA)
