"""Legacy regression, explicit-zero equivalence, supported/unsupported modes, extensions."""
import numpy as np
import pytest

from folps import (EFTExtension, PhysicalDerivativeBias, RSDMultipolesPowerSpectrumCalculator, SelectionBias,
                   get_rsd_pkell_marg_const, get_rsd_pkell_marg_derivatives)
from tests.conftest import PARS, ROOT

AP_CASES = (("ap1", 1.0, 1.0), ("ap2", 1.03, 0.98))


@pytest.fixture(scope="module")
def frozen():
    return np.load(ROOT / "tests" / "data" / "eft_baseline_reference.npz")


@pytest.mark.parametrize("ir", [True, False])
@pytest.mark.parametrize("tag,qpar,qper", AP_CASES)
def test_legacy_matches_frozen_pre_edit_baseline(eds_setup, calc, frozen, ir, tag, qpar, qper):
    pl = calc.get_rsd_pkell(frozen["kobs"], qpar, qper, PARS, eds_setup["table"], eds_setup["table_now"],
                            damping=None, IR_resummation=ir)
    ref = frozen[f"pell_EdS_ir{int(ir)}_{tag}"]
    assert np.max(np.abs(pl - ref)) <= 1e-10 * np.max(np.abs(ref))


@pytest.mark.parametrize("ir", [True, False])
@pytest.mark.parametrize("tag,qpar,qper", AP_CASES)
def test_explicit_zero_selection_equals_legacy(eds_setup, calc, frozen, ir, tag, qpar, qper):
    args = (frozen["kobs"], qpar, qper, PARS, eds_setup["table"], eds_setup["table_now"])
    legacy = calc.get_rsd_pkell(*args, damping=None, IR_resummation=ir)
    zero = calc.get_rsd_pkell(*args, damping=None, IR_resummation=ir, selection=SelectionBias(),
                              selection_tables=eds_setup["tabs"])
    assert np.array_equal(legacy, zero)


def test_explicit_zero_with_other_bias_schemes(eds_setup, calc, frozen):
    pars_desi = [1.645, -0.46, -0.18, 0.1] + PARS[4:]
    args = (frozen["kobs"], 1.0, 1.0, pars_desi, eds_setup["table"], eds_setup["table_now"])
    a = calc.get_rsd_pkell(*args, bias_scheme="desi", damping=None)
    b = calc.get_rsd_pkell(*args, bias_scheme="desi", damping=None, selection=SelectionBias(),
                           selection_tables=eds_setup["tabs"])
    assert np.array_equal(a, b)


def test_fk_legacy_unchanged(eds_setup, frozen, linear_input):
    from folps import NonLinearPowerSpectrumCalculator
    from tests.conftest import COSMO
    k, pk = linear_input
    nl = NonLinearPowerSpectrumCalculator(mmatrices=eds_setup["mm"], kernels="fk", **COSMO)
    t, tn = nl.calculate_loop_table(k=k, pklin=pk, cosmo=None, **COSMO)
    c = RSDMultipolesPowerSpectrumCalculator(model="EFT")
    c._printed_model_damping_pk = True
    pl = c.get_rsd_pkell(frozen["kobs"], 1.03, 0.98, PARS, t, tn, damping=None)
    ref = frozen["pell_fk_ir1_ap2"]
    assert np.max(np.abs(pl - ref)) <= 1e-10 * np.max(np.abs(ref))


def test_nonzero_selection_changes_all_multipoles(eds_setup, calc, frozen, generic_selection):
    args = (frozen["kobs"], 1.0, 1.0, PARS, eds_setup["table"], eds_setup["table_now"])
    legacy = calc.get_rsd_pkell(*args, damping=None)
    sel = calc.get_rsd_pkell(*args, damping=None, selection=generic_selection, selection_tables=eds_setup["tabs"])
    assert np.all(np.isfinite(sel))
    for ell in range(3):
        assert np.max(np.abs(sel[ell] - legacy[ell])) > 1e-3 * np.max(np.abs(legacy[ell]))


def test_components_sum_to_total(eds_setup, calc, generic_selection):
    k = np.geomspace(0.01, 0.3, 25)[:, None] * np.ones((1, 4))
    mu = np.linspace(0.1, 0.95, 4)[None, :] * np.ones((25, 1))
    pars = list(PARS)
    pars[7] = 0.3  # ctilde
    ext = EFTExtension(alpha6=2.0, alphashot2_iso=1.5, alphashot2_mu4=-0.7)
    comp = calc.get_rsd_pkmu(k, mu, pars, eds_setup["table"], eds_setup["table_now"], damping=None,
                             selection=generic_selection, selection_tables=eds_setup["tabs"],
                             eft_extension=ext, return_components=True)
    total = (comp["tree_baseline"] + comp["loop_baseline"] + comp["counterterms"] + comp["stochastic"]
             + comp["ctilde"] + comp["delta_tree"] + comp["delta_loop"] + comp["delta_ctilde"])
    assert np.allclose(total, comp["total"], rtol=1e-12, atol=1e-9)
    assert np.allclose(comp["tree"], comp["tree_baseline"] + comp["delta_tree"], rtol=1e-12)


def test_extension_terms_have_declared_shapes(eds_setup, calc):
    k = np.geomspace(0.01, 0.3, 20)[:, None] * np.ones((1, 3))
    mu = np.array([0.2, 0.6, 0.9])[None, :] * np.ones((20, 1))
    t, tn = eds_setup["table"], eds_setup["table_now"]
    base = calc.get_rsd_pkmu(k, mu, PARS, t, tn, damping=None, eft_extension=EFTExtension())
    legacy = calc.get_rsd_pkmu(k, mu, PARS, t, tn, damping=None)
    assert np.array_equal(base, legacy)
    iso = calc.get_rsd_pkmu(k, mu, PARS, t, tn, damping=None, eft_extension=EFTExtension(alphashot2_iso=1.0))
    assert np.allclose(iso - base, PARS[10] * k ** 2, rtol=1e-10)
    m4 = calc.get_rsd_pkmu(k, mu, PARS, t, tn, damping=None, eft_extension=EFTExtension(alphashot2_mu4=1.0))
    assert np.allclose(m4 - base, PARS[10] * k ** 2 * mu ** 4, rtol=1e-10)
    comp = calc.get_rsd_pkmu(k, mu, PARS, t, tn, damping=None, eft_extension=EFTExtension(alpha6=1.0),
                             return_components=True)
    a6 = calc.get_rsd_pkmu(k, mu, PARS, t, tn, damping=None, eft_extension=EFTExtension(alpha6=1.0))
    p0 = list(PARS)
    p0[4] = 1.0
    p0[5] = p0[6] = 0.0
    c0 = calc.get_rsd_pkmu(k, mu, p0, t, tn, damping=None, return_components=True)
    # alpha6 term = mu^6 x (the IR-mixed k^2 P template of alpha0)
    assert np.allclose(a6 - base, mu ** 6 * c0["counterterms"], rtol=1e-10)
    assert "counterterms" in comp


def test_physical_derivative_mode(eds_setup, calc, generic_selection):
    from folps import physical_to_polynomial_derivatives
    k = np.geomspace(0.01, 0.3, 20)[:, None] * np.ones((1, 3))
    mu = np.array([0.2, 0.6, 0.9])[None, :] * np.ones((20, 1))
    t, tn, tabs = eds_setup["table"], eds_setup["table_now"], eds_setup["tabs"]
    phys = PhysicalDerivativeBias(b_nabla2delta=1.2, beta_nabla2v=-0.8, beta_par2v=0.5)
    p0 = list(PARS)
    p0[4] = p0[5] = p0[6] = 0.0
    a = calc.get_rsd_pkmu(k, mu, p0, t, tn, damping=None, selection=generic_selection, selection_tables=tabs,
                          physical_derivatives=phys)
    b_eta = -1.0 + generic_selection.s_eta
    c0, c2, c4, c6 = physical_to_polynomial_derivatives(PARS[0], b_eta, t[-1], phys)
    p1 = list(PARS)
    p1[4], p1[5], p1[6] = c0, c2, c4
    b = calc.get_rsd_pkmu(k, mu, p1, t, tn, damping=None, selection=generic_selection, selection_tables=tabs,
                          eft_extension=EFTExtension(alpha6=c6))
    assert np.allclose(a, b, rtol=1e-12)
    with pytest.raises(ValueError, match="replaces the polynomial"):
        calc.get_rsd_pkmu(k, mu, PARS, t, tn, damping=None, physical_derivatives=phys)


def test_marginalization_templates_match_numerical_derivatives(eds_setup, generic_selection):
    """Design vectors returned by the gated helpers equal derivatives of the complete
    selected prediction (with AP != 1), including the alpha6 / isotropic-noise extensions."""
    kobs = np.geomspace(0.02, 0.25, 15)
    t, tn, tabs = eds_setup["table"], eds_setup["table_now"], eds_setup["tabs"]
    ext = EFTExtension()
    kw = dict(selection=generic_selection, selection_tables=tabs, eft_extension=ext)
    const = get_rsd_pkell_marg_const(kobs, 1.02, 0.99, PARS, t, tn, damping=None, **kw)
    ders = get_rsd_pkell_marg_derivatives(kobs, 1.02, 0.99, PARS, t, tn, damping=None, **kw)
    assert len(ders) == 3 and ders[0].shape == (8, kobs.size)
    c = RSDMultipolesPowerSpectrumCalculator(model="EFT")
    c._printed_model_damping_pk = True
    full = c.get_rsd_pkell(kobs, 1.02, 0.99, PARS, t, tn, damping=None, selection=generic_selection,
                           selection_tables=tabs, eft_extension=ext)
    theta = [PARS[4], PARS[5], PARS[6], PARS[8], PARS[9], 0.0, 0.0, 0.0]
    recon = const + sum(theta[i] * np.stack([d[i] for d in ders]) for i in range(8))
    assert np.allclose(recon, full, rtol=1e-10, atol=1e-8 * np.max(np.abs(full)))
    # central finite difference in alpha2 of the complete model
    h = 0.5
    pp, pm = list(PARS), list(PARS)
    pp[5] += h
    pm[5] -= h
    fd = (c.get_rsd_pkell(kobs, 1.02, 0.99, pp, t, tn, damping=None, selection=generic_selection,
                          selection_tables=tabs) -
          c.get_rsd_pkell(kobs, 1.02, 0.99, pm, t, tn, damping=None, selection=generic_selection,
                          selection_tables=tabs)) / (2 * h)
    assert np.allclose(fd, np.stack([d[1] for d in ders]), rtol=1e-9, atol=1e-9 * np.max(np.abs(fd)))


def test_legacy_marginalization_unchanged_without_selection(eds_setup):
    """The gated helpers agree with the legacy ones at zero selection.  Note: the legacy
    derivative helper integrates over mu with ``nmu`` full-range Gauss nodes whereas the
    model uses ``2 nmu`` symmetric nodes; with IR resummation that makes its ell=4 design
    vector inconsistent with the model at ~3e-4 (pre-existing; documented, not changed).
    With matched nodes (legacy nmu=12 vs model nmu=6) they agree to round-off."""
    kobs = np.geomspace(0.02, 0.25, 15)
    t, tn = eds_setup["table"], eds_setup["table_now"]
    a = get_rsd_pkell_marg_const(kobs, 1.0, 1.0, PARS, t, tn, damping=None)
    b = get_rsd_pkell_marg_const(kobs, 1.0, 1.0, PARS, t, tn, damping=None, selection=SelectionBias(),
                                 selection_tables=eds_setup["tabs"])
    assert np.allclose(a, b, rtol=1e-12, atol=1e-9)
    da = get_rsd_pkell_marg_derivatives(kobs, 1.0, 1.0, PARS, t, tn, damping=None, nmu=12)
    db = get_rsd_pkell_marg_derivatives(kobs, 1.0, 1.0, PARS, t, tn, damping=None, selection=SelectionBias(),
                                        selection_tables=eds_setup["tabs"])
    for x, y in zip(da, db):
        assert np.allclose(x, y[:5], rtol=1e-10, atol=1e-10 * np.max(np.abs(x)))
    dl = get_rsd_pkell_marg_derivatives(kobs, 1.0, 1.0, PARS, t, tn, damping=None)
    assert np.max(np.abs(dl[2] - db[2][:5])) > 1e-5 * np.max(np.abs(dl[2]))  # the documented legacy mismatch


@pytest.mark.parametrize("bad,exc,msg", [
    (dict(selection_tables=None), ValueError, "selection_tables"),
    (dict(bA=0.1), ValueError, "bA"),
    (dict(pars_b=PARS, cross_nuisance=PARS[4:]), NotImplementedError, "cross"),
])
def test_invalid_modes_raise(eds_setup, calc, generic_selection, bad, exc, msg):
    kw = dict(selection=generic_selection, selection_tables=eds_setup["tabs"])
    kw.update(bad)
    with pytest.raises(exc, match=msg):
        calc.get_rsd_pkmu(np.array([0.1]), np.array([0.5]), PARS, eds_setup["table"], eds_setup["table_now"],
                          damping=None, **kw)


def test_non_eft_model_and_fk_tables_rejected(eds_setup, generic_selection, linear_input):
    from folps import NonLinearPowerSpectrumCalculator
    from tests.conftest import COSMO
    t, tn, tabs = eds_setup["table"], eds_setup["table_now"], eds_setup["tabs"]
    with pytest.raises(NotImplementedError, match="EFT"):
        RSDMultipolesPowerSpectrumCalculator(model="FOLPSD").get_rsd_pkmu(
            np.array([0.1]), np.array([0.5]), PARS, t, tn, damping="lor", selection=generic_selection,
            selection_tables=tabs)
    k, pk = linear_input
    nl = NonLinearPowerSpectrumCalculator(mmatrices=eds_setup["mm"], kernels="fk", **COSMO)
    tf, tfn = nl.calculate_loop_table(k=k, pklin=pk, cosmo=None, **COSMO)
    c = RSDMultipolesPowerSpectrumCalculator(model="EFT")
    with pytest.raises(NotImplementedError, match="scale-dependent"):
        c.get_rsd_pkmu(np.array([0.1]), np.array([0.5]), PARS, tf, tfn, damping=None, selection=generic_selection,
                       selection_tables=tabs)
    with pytest.raises(TypeError):
        c.get_rsd_pkmu(np.array([0.1]), np.array([0.5]), PARS, t, tn, damping=None, selection=[0.1] * 9,
                       selection_tables=tabs)


def test_bA_legacy_path_still_works(eds_setup, calc):
    k = np.array([[0.05, 0.1]])
    mu = np.array([[0.5, 0.5]])
    a = calc.get_rsd_pkmu(k, mu, PARS, eds_setup["table"], eds_setup["table_now"], damping=None, bA=0.2)
    b = calc.get_rsd_pkmu(k, mu, PARS, eds_setup["table"], eds_setup["table_now"], damping=None)
    assert np.all(np.isfinite(a)) and not np.allclose(a, b)


def test_ir_resummation_reexpands_to_one_loop(eds_setup, calc, generic_selection):
    """Scaling the BAO displacement variances Sigma^2, delta Sigma^2 by eps^2 (table entries),
    P_IR(eps) - P_IR=off vanishes as eps^2 for the selected model: IR resummation only
    reorganizes beyond one loop (tree, loop and counterterm mixing together)."""
    t, tn, tabs = eds_setup["table"], eds_setup["table_now"], eds_setup["tabs"]
    k = np.geomspace(0.02, 0.3, 30)[:, None] * np.ones((1, 3))
    mu = np.array([0.2, 0.6, 0.95])[None, :] * np.ones((30, 1))
    kw = dict(damping=None, selection=generic_selection, selection_tables=tabs)
    off = calc.get_rsd_pkmu(k, mu, PARS, t, tn, IR_resummation=False, **kw)

    def scaled(eps):
        tn_eps = list(tn)
        tn_eps[-3] = tn[-3] * eps ** 2
        tn_eps[-2] = tn[-2] * eps ** 2
        return calc.get_rsd_pkmu(k, mu, PARS, t, tuple(tn_eps), IR_resummation=True, **kw) - off
    norms = [np.max(np.abs(scaled(e))) for e in (0.1, 0.05, 0.025)]
    assert norms[0] < 1e-3 * np.max(np.abs(off))
    r1, r2 = norms[0] / norms[1], norms[1] / norms[2]
    # O(eps^2): halving eps divides by 4, up to the O(eps^4) remainder (measured 3.92, 3.98)
    assert abs(r2 - 4.0) < abs(r1 - 4.0) and abs(r2 - 4.0) < 0.05, (r1, r2)
