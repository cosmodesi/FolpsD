"""FFTLog matrices and templates against FolpsD's independently derived closed forms."""
import json

import numpy as np
import pytest

from folps import SelectionBias, djs_coefficients, get_selection_matrices
from folps import selection_coefficients as sc
from folps.selection_templates import (SelectionLoopTables, _components, build_selection_tables,
                                       selection_loop)


def _pair_term(T, a, b, w):
    cls = sc.P22_PAIR_SINGULAR[a, b]
    W = np.stack([sc.P22_T[a, b, 0] * cls, sc.P22_T[a, b, 0] * (1 - cls)])
    return w * np.einsum("cj,cjk->k", W, T.I22)


def test_J_basis_reproduces_folps_M13(eds_setup):
    nl, T = eds_setup["nl"], eds_setup["tabs"].full
    P13, _ = nl.calculate_P13()
    from folps.folps import simpson
    s2psi = simpson(nl.inputpkT[1], x=nl.inputpkT[0]) / (6 * np.pi ** 2)
    K, J = T.k, T.J13
    assert np.isclose(T.sigma_m1_sq, 3 * s2psi, rtol=1e-14)
    dd = 2 * (2 / 21 * J[0] + 1 / 14 * J[1] - 1 / 6 * J[2])
    assert np.max(np.abs(dd - (P13[0] + 61 / 105 * K ** 2 * s2psi))) < 1e-12 * np.max(np.abs(P13[0]))
    tt = 2 * (2 / 7 * J[0] - 3 / 14 * J[1] - 1 / 14 * J[2])
    assert np.max(np.abs(tt - (P13[2] + 9 / 5 * K ** 2 * s2psi))) < 1e-12 * np.max(np.abs(P13[2]))
    # full bare P13dd from the stored matrix, J UV pieces and bare 1/x^2 coefficient
    bare = 2 * (sc.CUBIC_M[0, 0] @ (J + sc.J_UV[:, None] * K ** 2 * T.sigma_m1_sq)
                + sc.CUBIC_D[0, 0] * K ** 2 * T.sigma_m1_sq)
    assert np.max(np.abs(bare - P13[0])) < 1e-12 * np.max(np.abs(P13[0]))


def test_P22_matter_and_bias_terms_vs_folps(eds_setup):
    """P22dd (matter tilt) to round-off; FolpsD's b2/bs2 terms (its tilt -1.51 with zero-lag
    extrapolation) vs our local-class tilt: FFTLog-resolution-limited at nfftlog=128."""
    nl, T, table = eds_setup["nl"], eds_setup["tabs"].full, eds_setup["table"]
    P22, _ = nl.calculate_P22()
    assert np.max(np.abs(_pair_term(T, 0, 0, 1.0) - P22[0])) < 1e-12 * np.max(np.abs(P22[0]))
    sel = (T.k > 0.005) & (T.k < 0.45)
    for (a, b, w, col, tol) in ((0, 2, 0.5, 6, 1e-3), (0, 3, 0.5, 7, 5e-4), (2, 2, 0.25, 8, 1e-4),
                                (2, 3, 0.25, 9, 1e-4), (3, 3, 0.25, 10, 1e-4)):
        mine, ref = _pair_term(T, a, b, w), table[col]
        assert np.max(np.abs(mine - ref)[sel]) < tol * np.max(np.abs(ref)[sel]), (a, b)


def test_amplitude_scaling(eds_setup, linear_input):
    """P_L -> A P_L: selected P22 and P13 scale as A^2 (tree as A)."""
    from folps import NonLinearPowerSpectrumCalculator
    from tests.conftest import COSMO
    k, pk = linear_input
    nl2 = NonLinearPowerSpectrumCalculator(mmatrices=eds_setup["mm"], kernels="EdS", **COSMO)
    nl2.calculate_loop_table(k=k, pklin=2.0 * pk, pknow=None, cosmo=None, **COSMO)
    T1, T2 = eds_setup["tabs"].full, build_selection_tables(nl2).full
    c = djs_coefficients(1.6, -0.4, -0.2, 0.1, SelectionBias(*([0.3, -0.2, 0.25, 0.4, -0.5, 0.2, -0.3, 0.3, 0.2])))
    kk = T1.k[20:110]
    mu = np.full_like(kk, 0.6)
    a22, a13 = selection_loop(kk, mu, T1, c)
    b22, b13 = selection_loop(kk, mu, T2, c)
    assert np.allclose(b22, 4 * a22, rtol=1e-6)
    assert np.allclose(b13, 4 * a13, rtol=1e-6)
    assert np.allclose(T2.pk, 2 * T1.pk, rtol=1e-12)


def test_increment_exactly_zero_and_consistent(eds_setup, generic_selection):
    T = eds_setup["tabs"].full
    c = djs_coefficients(1.6, -0.4, -0.2, 0.1, generic_selection)
    c0 = djs_coefficients(1.6, -0.4, -0.2, 0.1, SelectionBias())
    kk = T.k[5:115]
    mu = np.linspace(0, 1, kk.size)
    ip = (T.pk[5:115], T.I22[:, :, 5:115], T.J13[:, 5:115])
    z22, z13 = _components(*ip, T, c0, c0, kk, mu)
    assert np.all(z22 == 0) and np.all(z13 == 0)
    d22, d13 = _components(*ip, T, c, c0, kk, mu)
    f22, f13 = _components(*ip, T, c, None, kk, mu)
    g22, g13 = _components(*ip, T, c0, None, kk, mu)
    assert np.allclose(d22, f22 - g22, rtol=1e-10, atol=1e-8 * np.max(np.abs(f22)))
    assert np.allclose(d13, f13 - g13, rtol=1e-10, atol=1e-8 * np.max(np.abs(f13)))


def test_matrix_cache_roundtrip_and_rejection(tmp_path):
    m1 = get_selection_matrices(nfftlog=32, b_nu=-0.1, cache_dir=str(tmp_path))
    files = list(tmp_path.glob("selection_matrices_*.npz"))
    assert len(files) == 1
    from folps import selection_matrices as smod
    smod._MEMORY_CACHE.clear()
    m2 = get_selection_matrices(nfftlog=32, b_nu=-0.1, cache_dir=str(tmp_path))
    assert np.array_equal(m1.p22, m2.p22) and m1.key == m2.key
    # different grid -> different key
    m3 = get_selection_matrices(nfftlog=32, b_nu=-0.9, cache_dir=str(tmp_path))
    assert m3.key != m1.key
    # tampered metadata -> rejected
    with np.load(files[0]) as d:
        meta = json.loads(str(d["metadata_json"]))
        meta["coefficients_sha256"] = "0" * 64
        np.savez(files[0], p22=d["p22"], j13=d["j13"], metadata_json=json.dumps(meta))
    smod._MEMORY_CACHE.clear()
    with pytest.raises(ValueError, match="Incompatible"):
        get_selection_matrices(nfftlog=32, b_nu=-0.1, cache_dir=str(tmp_path))
    smod._MEMORY_CACHE.clear()


def test_tables_metadata_and_pairing(eds_setup):
    tabs = eds_setup["tabs"]
    assert tabs.full.metadata["kernels"] == "EdS"
    assert tabs.full.metadata["coefficients_sha256"] == sc.COEFFICIENT_DATA_SHA256
    assert tabs.full.I22.shape == (2, 23, 120) and tabs.full.J13.shape == (5, 120)
    other = type(tabs.nowiggle)(*tabs.nowiggle.arrays(), metadata=dict(tabs.nowiggle.metadata, pair_id="x"))
    with pytest.raises(ValueError, match="different calculator"):
        SelectionLoopTables(tabs.full, other)


def test_fk_calculator_rejected(eds_setup, linear_input):
    from folps import NonLinearPowerSpectrumCalculator
    from tests.conftest import COSMO
    k, pk = linear_input
    nl = NonLinearPowerSpectrumCalculator(mmatrices=eds_setup["mm"], kernels="fk", **COSMO)
    nl.calculate_loop_table(k=k, pklin=pk, cosmo=None, **COSMO)
    with pytest.raises(NotImplementedError, match="EdS"):
        build_selection_tables(nl)
