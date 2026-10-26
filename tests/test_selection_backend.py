"""NumPy / JAX parity and JAX derivatives (each backend in its own process)."""
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest

from tests.conftest import ROOT

pytestmark = pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX not installed")


@pytest.fixture(scope="module")
def runs(tmp_path_factory, eds_setup):
    tmp = tmp_path_factory.mktemp("backend")
    from folps.tools import extrapolate_pklin, get_pknow
    k, pk = np.loadtxt(ROOT / "folps" / "inputpkT.txt", unpack=True)
    kk, pp = extrapolate_pklin(k, pk)
    knw, pnw = get_pknow(kk, pp, 0.6711)
    pknow_path = tmp / "pknow.npy"
    np.save(pknow_path, np.interp(k, knw, pnw))
    out = {}
    env = dict(os.environ)
    for backend in ("numpy", "jax"):
        path = tmp / f"{backend}.npz"
        r = subprocess.run([sys.executable, str(ROOT / "tests" / "backend_worker.py"), backend, str(path), str(pknow_path)],
                           cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=3000)
        assert r.returncode == 0, r.stderr[-3000:]
        out[backend] = np.load(path)
    return out


def _rel(a, b):
    return np.max(np.abs(a - b), axis=-1) / np.max(np.abs(b), axis=-1)


def test_contraction_parity(runs):
    a, b = runs["numpy"], runs["jax"]
    for key in ("contract_d22", "contract_d13"):
        assert np.max(np.abs(a[key] - b[key])) <= 1e-11 * np.max(np.abs(a[key]))


def test_end_to_end_parity(runs):
    """Stable predictions at the table nodes with an identical no-wiggle input.  The residual
    (~1e-6) comes from FolpsD's backend-specific interpolation of P_L onto the FFTLog grid
    (get_cm), shared by the legacy and selected predictions."""
    a, b = runs["numpy"], runs["jax"]
    assert np.all(_rel(b["legacy"], a["legacy"]) < 1e-5)
    assert np.all(_rel(b["selected"], a["selected"]) < 1e-5)
    inc_a, inc_b = a["selected"] - a["legacy"], b["selected"] - b["legacy"]
    assert np.all(_rel(inc_b, inc_a) < 5e-6)
    # off-node (AP != 1) evaluation: the legacy tables are interpolated with a SciPy cubic
    # spline (NumPy) or the backend interp (JAX); the same difference appears in legacy and
    # selected predictions (P4 up to ~7e-4), so compare like with like
    assert np.all(_rel(b["legacy_ap"], a["legacy_ap"]) < 2e-3)
    assert np.all(_rel(b["selected_ap"], a["selected_ap"]) < 2e-3)
    inc_a, inc_b = a["selected_ap"] - a["legacy_ap"], b["selected_ap"] - b["legacy_ap"]
    assert np.all(_rel(inc_b, inc_a) < 2e-3)


def test_jit_and_autodiff_vs_finite_differences(runs):
    b = runs["jax"]
    assert np.max(np.abs(b["jit"] - b["eager"])) < 1e-9 * np.max(np.abs(b["eager"]))
    jac = b["jac"]                       # (ell, k, theta): 9 selection coords, b1, qpar, qper
    assert np.all(np.isfinite(jac))
    errs = {h: np.max(np.abs(b[f"fd_{h:g}"] - jac), axis=(0, 1)) / np.max(np.abs(jac), axis=(0, 1))
            for h in (1e-2, 1e-3, 1e-4)}
    # selection coordinates and b1 (incl. the b_delta_eta = -b1 + s offset): the prediction is
    # polynomial (<= quadratic) in them, so central differences are exact up to round-off
    for h in errs:
        assert np.all(errs[h][:10] < 1e-8), (h, errs[h][:10])
    # AP (qpar, qper): O(h^2) convergence of central differences towards the autodiff value
    ap = np.array([errs[h][10:] for h in (1e-2, 1e-3, 1e-4)])
    assert np.all(ap[1] < ap[0] / 10) and np.all(ap[2] < 1e-5), ap
    rel = np.max(np.abs(b["dA_ad"] - b["dA_fd"])) / np.max(np.abs(b["dA_ad"]))
    assert rel < 1e-6
