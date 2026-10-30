"""Selected two-tracer (cross) spectra: exchange, auto, zero-selection and Kaiser-only limits.

The selected cross spectrum is the legacy (Ruiyang) cross spectrum plus selected-minus-
unselected increments from the two-tracer contractions of
:func:`folps.selection_templates._components_cross`.  The direct-quadrature check of the
mixed contractions at unequal selections is in ``test_selection_cross_reference`` (slow).
"""
import numpy as np
import pytest

from folps import EFTExtension, SelectionBias, djs_coefficients, folps_bias_to_djs, tree_bA_to_linear_djs
from folps.selection_templates import _components, _components_cross
from tests.conftest import PARS

# a second tracer, clearly different from conftest.PARS (A), and a pair-level nuisance vector
PARS_B = [1.10, 0.35, -4.0 / 7.0 * (1.10 - 1.0), 32.0 / 315.0 * (1.10 - 1.0),
          -1.2, 7.5, -3.1, 0.0, 0.02, 2.4, 1.0 / 0.0003, 0.0]
CROSS = [0.7, -3.3, 2.2, 0.0, 0.004, -1.1, 1.0 / 0.00025, 0.0]
SEL_B = SelectionBias(s_eta=-0.17, s_delta_eta=0.41, s_eta2=-0.22, s_KK_par=-0.31, s_Pi2_par=0.28,
                      s_delta_Pi2_par=-0.12, s_eta_Pi2_par=0.37, s_KPi2_par=-0.26, s_Pi3_par=0.44)
KOBS = np.linspace(0.02, 0.25, 24)
AP = (1.03, 0.98)


def _ell(calc, s, pars, sel=None, pars_b=None, sel_b=None, cross=None, ap=AP, **kw):
    return calc.get_rsd_pkell(KOBS, *ap, pars, s["table"], s["table_now"], bias_scheme="folps", damping=None,
                              nmu=6, ells=(0, 2, 4), selection=sel, selection_b=sel_b, selection_tables=s["tabs"],
                              pars_b=pars_b, cross_nuisance=cross, **kw)


def _close(a, b, rtol=1e-12):
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b)) <= rtol * np.max(np.abs(b))


def test_exchange_symmetry(calc, eds_setup, generic_selection):
    ext = EFTExtension(alpha6=0.3, alphashot2_iso=-0.7, alphashot2_mu4=1.1)
    ab = _ell(calc, eds_setup, PARS, generic_selection, PARS_B, SEL_B, CROSS, eft_extension=ext)
    ba = _ell(calc, eds_setup, PARS_B, SEL_B, PARS, generic_selection, CROSS, eft_extension=ext)
    assert _close(ab, ba, 1e-13)
    # and the selection actually matters
    zero = _ell(calc, eds_setup, PARS, SelectionBias(), PARS_B, SelectionBias(), CROSS, eft_extension=ext)
    assert np.max(np.abs(ab - zero)) > 1e-2 * np.max(np.abs(zero))


def test_auto_limit(calc, eds_setup, generic_selection):
    """A = B, same selection, pair nuisance = the auto's: the selected auto spectrum (AP, IR, noise)."""
    ext = EFTExtension(alpha6=0.3, alphashot2_iso=-0.7, alphashot2_mu4=1.1)
    auto = _ell(calc, eds_setup, PARS, generic_selection, eft_extension=ext)
    cross = _ell(calc, eds_setup, PARS, generic_selection, list(PARS), generic_selection, PARS[4:], eft_extension=ext)
    assert _close(cross, auto, 1e-12)
    pk_auto = calc.get_rsd_pkmu(KOBS[:, None], np.linspace(0, 1, 7)[None], PARS, eds_setup["table"],
                                eds_setup["table_now"], damping=None, selection=generic_selection,
                                selection_tables=eds_setup["tabs"])
    pk_cross = calc.get_rsd_pkmu(KOBS[:, None], np.linspace(0, 1, 7)[None], PARS, eds_setup["table"],
                                 eds_setup["table_now"], damping=None, selection=generic_selection,
                                 selection_b=generic_selection, selection_tables=eds_setup["tabs"],
                                 pars_b=list(PARS), cross_nuisance=PARS[4:])
    assert _close(pk_cross, pk_auto, 1e-12)


def test_zero_selection_is_legacy_cross(calc, eds_setup):
    """Explicit zero selections run the whole machinery and give the Ruiyang cross spectrum."""
    legacy = _ell(calc, eds_setup, PARS, None, PARS_B, None, CROSS)
    zero = _ell(calc, eds_setup, PARS, SelectionBias(), PARS_B, SelectionBias(), CROSS)
    assert np.array_equal(legacy, zero) or _close(zero, legacy, 1e-15)
    # selection of one tracer only: the increment is that tracer's, the other's leg unselected
    one = _ell(calc, eds_setup, PARS, SEL_B, PARS_B, SelectionBias(), CROSS)
    assert np.max(np.abs(one - legacy)) > 1e-3 * np.max(np.abs(legacy))


def test_zero_selection_templates_vs_ruiyang_loop(calc, eds_setup):
    """Two-tracer template contraction at s = 0 vs the legacy cross loop (A/D/G and GTNS included),
    IR off, AP = 1, on the table grid. Same tolerance as the auto gate (nfftlog = 128)."""
    import folps.folps as F
    T, table = eds_setup["tabs"].full, eds_setup["table"]
    sel = slice(40, 115)
    k = np.asarray(T.k)[sel]
    zero_a, zero_b = list(PARS), list(PARS_B)
    for p in (zero_a, zero_b):
        p[4:10] = [0.0] * 6
    for mu in (0.0, 0.35, 0.7, 1.0):
        tab = calc.interp_table(k, table, F.A_full_status)
        legacy = calc.get_eft_pkmu(k, mu, zero_a, tab, damping=None, damping_method="loop+ctr",
                                   pars_b=zero_b, cross_nuisance=[0.0] * 6 + [1.0, 0.0])
        cA = djs_coefficients(*folps_bias_to_djs(*PARS[:4]), SelectionBias())
        cB = djs_coefficients(*folps_bias_to_djs(*PARS_B[:4]), SelectionBias())
        ip = (np.asarray(T.pk)[sel], np.asarray(T.I22)[:, :, sel], np.asarray(T.J13)[:, sel])
        p22, p13 = _components_cross(*ip, T, cA, cB, k=k, mu=np.full_like(k, mu))
        scale = np.max(np.abs(legacy))
        assert np.max(np.abs(p22 + p13 - legacy)) < 1.5e-3 * scale, mu


def test_polarization_of_p22(eds_setup, generic_selection):
    """P22_AB(cA, cB) = [Q(cA + cB) - Q(cA - cB)] / 4 with Q the validated auto P22 (bilinear form)."""
    from folps.selection_operators import DJSCoefficients
    T = eds_setup["tabs"].full
    i = np.arange(50, 110, 7)
    k = np.asarray(T.k)[i]
    ip = (np.asarray(T.pk)[i], np.asarray(T.I22)[:, :, i], np.asarray(T.J13)[:, i])
    cA = djs_coefficients(*folps_bias_to_djs(*PARS[:4]), generic_selection)
    cB = djs_coefficients(*folps_bias_to_djs(*PARS_B[:4]), SEL_B)
    plus = DJSCoefficients(*[a + b for a, b in zip(cA, cB)])
    minus = DJSCoefficients(*[a - b for a, b in zip(cA, cB)])
    for mu in (0.1, 0.6, 0.95):
        m = np.full_like(k, mu)
        q_plus = _components(*ip, T, plus, None, k, m)[0]
        q_minus = _components(*ip, T, minus, None, k, m)[0]
        p22 = _components_cross(*ip, T, cA, cB, k=k, mu=m)[0]
        assert _close(p22, 0.25 * (q_plus - q_minus), 1e-12)


def test_increment_equals_difference(eds_setup, generic_selection):
    """The exact-difference increment equals the difference of full contractions."""
    T = eds_setup["tabs"].full
    i = np.arange(30, 120, 9)
    k = np.asarray(T.k)[i]
    ip = (np.asarray(T.pk)[i], np.asarray(T.I22)[:, :, i], np.asarray(T.J13)[:, i])
    bA, bB = folps_bias_to_djs(*PARS[:4]), folps_bias_to_djs(*PARS_B[:4])
    cA, cA0 = djs_coefficients(*bA, generic_selection), djs_coefficients(*bA, SelectionBias())
    cB, cB0 = djs_coefficients(*bB, SEL_B), djs_coefficients(*bB, SelectionBias())
    m = np.full_like(k, 0.45)
    full = _components_cross(*ip, T, cA, cB, k=k, mu=m)
    full0 = _components_cross(*ip, T, cA0, cB0, k=k, mu=m)
    inc = _components_cross(*ip, T, cA, cB, cA0, cB0, k=k, mu=m)
    for x, y, z in zip(full, full0, inc):
        assert _close(z, x - y, 1e-10)


def test_kaiser_only_bA_cross(calc, eds_setup):
    """Legacy tree-only bA (M1) on a cross spectrum: the change is exactly the product of the two
    modified linear kernels times the IR-resummed linear spectrum; the loops are untouched."""
    import folps.folps as F
    k = KOBS[:, None]
    mu = np.linspace(0, 1, 9)[None]
    bAa, bAb = 0.31, -0.17
    t, tn = eds_setup["table"], eds_setup["table_now"]
    p0 = calc.get_rsd_pkmu(k, mu, PARS, t, tn, damping=None, pars_b=PARS_B, cross_nuisance=CROSS)
    p1 = calc.get_rsd_pkmu(k, mu, PARS, t, tn, damping=None, pars_b=PARS_B, cross_nuisance=CROSS, bA=(bAa, bAb))
    tab, tab_now = calc.interp_table(k, t, F.A_full_status), calc.interp_table(k, tn, F.A_full_status)
    f0, pkl, pkn = tab[-1], tab[0], tab_now[0]
    sigma2, dsigma2 = tab_now[-3:-1]
    s2t = (1 + f0 * mu ** 2 * (2 + f0)) * sigma2 + (f0 * mu) ** 2 * (mu ** 2 - 1) * dsigma2
    E = np.exp(-k ** 2 * s2t)
    tree = pkn + E * (pkl - pkn) * (1 + k ** 2 * s2t)
    za, zb = PARS[0] + f0 * mu ** 2, PARS_B[0] + f0 * mu ** 2
    expected = ((za + bAa * (mu ** 2 - 1 / 3)) * (zb + bAb * (mu ** 2 - 1 / 3)) - za * zb) * tree
    assert _close(p1 - p0, expected, 1e-11)


def test_matched_linear_kernel_M1_M2():
    """tree_bA_to_linear_djs reproduces the Kaiser-only bA kernel exactly at linear order."""
    f, b1, bA = 0.81, 1.9, 0.37
    b1d, s_eta = tree_bA_to_linear_djs(b1, bA, f)
    c = djs_coefficients(b1d, 0.0, 0.0, 0.0, SelectionBias(s_eta=s_eta))
    mu = np.linspace(0, 1, 11)
    assert np.allclose(c.b1 - c.b_eta * f * mu ** 2, b1 + f * mu ** 2 + bA * (mu ** 2 - 1 / 3), rtol=0, atol=1e-14)


def test_invalid_cross_requests(calc, eds_setup, generic_selection):
    with pytest.raises(ValueError):   # one selection missing
        _ell(calc, eds_setup, PARS, generic_selection, PARS_B, None, CROSS)
    with pytest.raises(ValueError):   # selection_b without a cross spectrum
        _ell(calc, eds_setup, PARS, generic_selection, None, SEL_B, None)
    with pytest.raises(ValueError):   # bA together with selection
        calc.get_rsd_pkmu(KOBS, 0.5, PARS, eds_setup["table"], eds_setup["table_now"], damping=None, bA=0.1,
                          selection=generic_selection, selection_b=SEL_B, selection_tables=eds_setup["tabs"],
                          pars_b=PARS_B, cross_nuisance=CROSS)
    from folps import PhysicalDerivativeBias
    with pytest.raises(NotImplementedError):
        _ell(calc, eds_setup, PARS, generic_selection, PARS_B, SEL_B, CROSS,
             physical_derivatives=PhysicalDerivativeBias(0.1, 0.0, 0.0))
