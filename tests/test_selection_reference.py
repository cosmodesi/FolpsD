"""Fast templates vs the independent direct-quadrature reference.

The default run compares with the shipped fixture ``tests/data/selection_reference_fixture.npz``
(direct quadrature of the exact redshift-space map with oracle kernels; regenerate with
``scripts/validate_selection_reference.py``).  ``--runslow`` recomputes points.

Tolerances are set from the measured convergence (validation report): at FolpsD's default
``nfftlog=128`` the FFTLog resolution limits P22 to ~3e-3 for k > 0.25; with 256 it is ~1e-5.
Residuals are normalized by the maximum |loop| (or |increment|) over k at fixed mu.
"""
import json

import numpy as np
import pytest

from folps import SelectionBias, djs_coefficients, folps_bias_to_djs
from folps.selection_templates import _components
from tests.conftest import ROOT


@pytest.fixture(scope="module")
def fixture():
    d = np.load(ROOT / "tests" / "data" / "selection_reference_fixture.npz")
    return d, json.loads(str(d["metadata_json"]))


def _fast(T, i, mu, c):
    ip = (T.pk[i:i + 1], T.I22[:, :, i:i + 1], T.J13[:, i:i + 1])
    return [float(x[0]) for x in _components(*ip, T, c, None, np.array([T.k[i]]), np.array([mu]))]


def test_fixture_provenance(fixture):
    import hashlib
    d, meta = fixture
    assert meta["input_sha256"] == hashlib.sha256((ROOT / "folps" / "inputpkT.txt").read_bytes()).hexdigest()
    assert len(meta["selection"]) == 9 and all(v != 0 for v in meta["selection"].values())


def test_fast_templates_vs_direct_quadrature(fixture, eds_setup):
    d, meta = fixture
    T = eds_setup["tabs"].full
    sel = SelectionBias(**meta["selection"])
    b = folps_bias_to_djs(*meta["pars"][:4])
    c, c0 = djs_coefficients(*b, sel), djs_coefficients(*b, SelectionBias())
    for mu in np.unique(d["mu"]):
        sub = np.where(d["mu"] == mu)[0]
        loop_scale = np.max(np.abs(d["ref_p22"][sub] + d["ref_p13"][sub]))
        inc_ref = (d["ref_p22"] + d["ref_p13"] - d["ref_p22_0"] - d["ref_p13_0"])[sub]
        inc_scale = np.max(np.abs(inc_ref))
        for j, inc in zip(sub, inc_ref):
            i = int(d["i"][j])
            a22, a13 = _fast(T, i, mu, c)
            z22, z13 = _fast(T, i, mu, c0)
            tol = 1e-3 if T.k[i] < 0.25 else 5e-3          # nfftlog = 128
            assert abs(a22 - d["ref_p22"][j]) < tol * loop_scale, (T.k[i], mu, "P22")
            assert abs(a13 - d["ref_p13"][j]) < tol * loop_scale, (T.k[i], mu, "P13")
            assert abs((a22 + a13 - z22 - z13) - inc) < 2 * tol * inc_scale, (T.k[i], mu, "increment")
    # the fixture itself stores the nfftlog=256 comparison: much tighter
    for comp in ("p22", "p13"):
        rel = np.abs(d[f"fast256_{comp}"] - d[f"ref_{comp}"])
        for mu in np.unique(d["mu"]):
            sub = d["mu"] == mu
            assert np.max(rel[sub]) < 2e-4 * np.max(np.abs(d["ref_p22"][sub] + d["ref_p13"][sub])), comp


@pytest.mark.slow
def test_recompute_reference_points(fixture, eds_setup):
    from folps.selection_reference import LinearSpectrum, direct_p13_ratio, direct_p22, exact_map_field_from_djs
    d, meta = fixture
    nl, T = eds_setup["nl"], eds_setup["tabs"].full
    P = LinearSpectrum(*nl.inputpkT)
    sel = SelectionBias(**meta["selection"])
    c = djs_coefficients(*folps_bias_to_djs(*meta["pars"][:4]), sel)
    F = exact_map_field_from_djs(c._asdict())
    for j in (2, len(d["k"]) // 2):
        k, mu = float(d["k"][j]), float(d["mu"][j])
        r22 = direct_p22(k, mu, P, F, float(T.f0), **meta["quadrature"]["p22"])
        assert abs(r22 - d["ref_p22"][j]) < 1e-9 * abs(r22)
        Z1 = c.b1 - c.b_eta * float(T.f0) * mu ** 2
        r13 = 2 * Z1 * float(P(k)) * direct_p13_ratio(k, mu, P, F, float(T.f0), **meta["quadrature"]["p13"])
        assert abs(r13 - d["ref_p13"][j]) < 1e-9 * abs(r13)


@pytest.mark.slow
def test_ir_insensitivity_of_complete_selected_loop(eds_setup, linear_input, generic_selection):
    """Adding power at q << k changes P22 and P13 by ~ k^2 sigma^2 P(k) each, but their sum
    only at O(q^2/k^2): checked with the fast templates for a generic selection."""
    from folps import NonLinearPowerSpectrumCalculator, build_selection_tables
    from folps.selection_templates import selection_loop
    from tests.conftest import COSMO
    k, pk = linear_input
    bump = 0.3 * pk * np.exp(-0.5 * (np.log(k / 2e-3) / 0.25) ** 2)
    out = []
    for p in (pk, pk + bump):
        nl = NonLinearPowerSpectrumCalculator(mmatrices=eds_setup["mm"], kernels="EdS", **COSMO)
        nl.calculate_loop_table(k=k, pklin=p, pknow=None, cosmo=None, **COSMO)
        T = build_selection_tables(nl).nowiggle
        c = djs_coefficients(1.6, -0.4, -0.2, 0.1, generic_selection)
        kk = T.k[85:110]
        out.append(selection_loop(kk, np.full_like(kk, 0.7), T, c))
    d22, d13 = out[1][0] - out[0][0], out[1][1] - out[0][1]
    assert np.max(np.abs(d22 + d13)) < 0.05 * np.max(np.abs(d22))
