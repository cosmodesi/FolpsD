"""Independent checks of the selected two-tracer (cross) spectrum (audit 2026-10-30).

Unlike tests/test_selection_cross.py, these never compare the cross path with itself:

* the complete selected cross spectrum (tree, loops, GTNS, ctilde, IR, AP, EFTExtension) is rebuilt from three calls of the
  validated AUTO model, ``P_AB = 2 P(m) - [P(A) + P(B)] / 2`` with ``m`` the midpoint of the two tracers: exact for any affine
  field (the DJS coefficient map, the FolpsD bias adapter and the matched linear coordinates are all affine);
* the linear Kaiser limit with independent A_A, A_B, the GTNS term, the pair-level counterterms / noise and the selected ctilde
  term are compared with hand formulas.
"""
import numpy as np
import pytest

from folps import EFTExtension, SelectionBias, djs_coefficients, folps_bias_to_djs
from tests.conftest import PARS

PARS_A = list(PARS[:10]) + [1.0, 0.0]          # PshotP equal for both tracers (the noise term is PshotP x sn: bilinear)
PARS_B = [1.10, 0.35, -4.0 / 7.0 * 0.10, 32.0 / 315.0 * 0.10, -1.2, 7.5, -3.1, 0.0, 0.02, 2.4, 1.0, 0.0]
SEL_B = SelectionBias(s_eta=-0.17, s_delta_eta=0.41, s_eta2=-0.22, s_KK_par=-0.31, s_Pi2_par=0.28,
                      s_delta_Pi2_par=-0.12, s_eta_Pi2_par=0.37, s_KPi2_par=-0.26, s_Pi3_par=0.44)
KOBS = np.linspace(0.02, 0.25, 24)
AP = (1.03, 0.98)
A_A, A_B = 0.31, -0.17


def _mid(x, y):
    return [0.5 * (a + b) for a, b in zip(x, y)]


def _rel(a, b):
    return np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(np.asarray(b)))


def _linear(pars, A, f0):
    p = list(pars)
    p[0] -= A / 3.0
    return p, SelectionBias(s_eta=-A / f0)


def _pkell(calc, s, pars, sel=None, pars_b=None, sel_b=None, cross=None, ext=None, IR=True):
    kw = {}
    if sel is not None or ext is not None:
        kw = dict(selection=sel, selection_tables=s["tabs"] if sel is not None else None, eft_extension=ext)
        if pars_b is not None:
            kw["selection_b"] = sel_b
    return np.asarray(calc.get_rsd_pkell(KOBS, *AP, pars, s["table"], s["table_now"], bias_scheme="folps", damping=None, nmu=6,
                                         ells=(0, 2, 4), IR_resummation=IR, pars_b=pars_b, cross_nuisance=cross, **kw))


@pytest.mark.parametrize("case", ["M0", "M2_linear", "M3_full_ext"])
@pytest.mark.parametrize("IR", [True, False])
def test_cross_equals_three_autos(calc, eds_setup, generic_selection, case, IR):
    f0 = float(eds_setup["table"][-1])
    ext = None
    if case == "M0":
        pa, sa, pb, sb = PARS_A, None, PARS_B, None
    elif case == "M2_linear":
        (pa, sa), (pb, sb) = _linear(PARS_A, A_A, f0), _linear(PARS_B, A_B, f0)
    else:
        pa, sa, pb, sb = PARS_A, generic_selection, PARS_B, SEL_B
        ext = EFTExtension(alpha6=0.3, alphashot2_iso=-0.7, alphashot2_mu4=1.1)
    pm = _mid(pa, pb)
    sm = None if sa is None else SelectionBias(*_mid(sa, sb))
    cross = _pkell(calc, eds_setup, pa, sa, pb, sb, pm[4:], ext=ext, IR=IR)
    autos = [_pkell(calc, eds_setup, p, s, ext=ext, IR=IR) for p, s in ((pa, sa), (pb, sb), (pm, sm))]
    assert _rel(cross, 2 * autos[2] - 0.5 * (autos[0] + autos[1])) < 1e-10


def _tree_ir(calc, s, k, mu):
    import folps.folps as F
    tab, tabn = calc.interp_table(k, s["table"], F.A_full_status), calc.interp_table(k, s["table_now"], F.A_full_status)
    f0 = tab[-1]
    s2, ds2 = tabn[-3], tabn[-2]
    s2t = (1 + f0 * mu ** 2 * (2 + f0)) * s2 + (f0 * mu) ** 2 * (mu ** 2 - 1) * ds2
    E = np.exp(-k ** 2 * s2t)
    return tab[0] + 0 * E, tabn[0], E, tab, tabn, f0, tabn[0] + E * (tab[0] - tabn[0]) * (1 + k ** 2 * s2t)


def test_kaiser_independent_amplitudes_mixed_term(calc, eds_setup):
    """M1 cross: the A_A A_B (mu^2 - 1/3)^2 term is present (not ((A_A + A_B)/2)^2)."""
    k, mu = KOBS[:, None], np.linspace(0, 1, 9)[None]
    cross = _mid(PARS_A, PARS_B)[4:]
    get = lambda bA: calc.get_rsd_pkmu(k, mu, PARS_A, eds_setup["table"], eds_setup["table_now"], damping=None, pars_b=PARS_B,
                                       cross_nuisance=cross, bA=bA)
    mixed = get((A_A, A_B)) - get((A_A, 0.0)) - get((0.0, A_B)) + get(None)
    tree = _tree_ir(calc, eds_setup, k, mu)[-1]
    assert _rel(mixed, A_A * A_B * (mu ** 2 - 1 / 3) ** 2 * tree) < 1e-9


def test_gtns_cross_hand(calc, eds_setup):
    k, mu = KOBS[:, None], np.linspace(0, 1, 9)[None]
    cross = _mid(PARS_A, PARS_B)[4:]
    on, off = (calc.get_rsd_pkmu(k, mu, PARS_A, eds_setup["table"], eds_setup["table_now"], damping=None, pars_b=PARS_B,
                                 cross_nuisance=cross, use_GTNS=g) for g in (True, False))
    _, _, E, tab, tabn, f0, _ = _tree_ir(calc, eds_setup, k, mu)
    G = lambda t: -(k * mu * f0) ** 2 * t[33] * (PARS_A[0] + f0 * mu ** 2) * (PARS_B[0] + f0 * mu ** 2) * t[0]
    assert _rel(on - off, E * G(tab) + (1 - E) * G(tabn)) < 1e-11


def test_selected_ctilde_and_pair_counterterms_hand(calc, eds_setup, generic_selection):
    k, mu = KOBS[:, None], np.linspace(0, 1, 9)[None]
    _, _, E, tab, tabn, f0, _ = _tree_ir(calc, eds_setup, k, mu)
    kw = dict(selection=generic_selection, selection_b=SEL_B, selection_tables=eds_setup["tabs"])
    get = lambda n: calc.get_rsd_pkmu(k, mu, PARS_A, eds_setup["table"], eds_setup["table_now"], damping=None, pars_b=PARS_B,
                                      cross_nuisance=n, **kw)
    zero = [0.0] * 6 + [1.0, 0.0]
    nuis = [2.3, -7.1, 4.4, 0.37, 0.11, -3.3, 1.0, 0.0]
    cA = djs_coefficients(*folps_bias_to_djs(*PARS_A[:4]), generic_selection)
    cB = djs_coefficients(*folps_bias_to_djs(*PARS_B[:4]), SEL_B)
    ZA, ZB = cA.b1 - cA.b_eta * f0 * mu ** 2, cB.b1 - cB.b_eta * f0 * mu ** 2
    a0, a2, a4, ct, s0, s2 = nuis[:6]
    hand = ((a0 + a2 * mu ** 2 + a4 * mu ** 4) * k ** 2 * (E * tab[0] + (1 - E) * tabn[0]) + (s0 + s2 * (k * mu) ** 2)
            + ct * (mu * k * f0) ** 4 * ZA * ZB * (E * tab[33] ** 2 * tab[0] + (1 - E) * tabn[33] ** 2 * tabn[0]))
    assert _rel(get(nuis) - get(zero), hand) < 1e-11
