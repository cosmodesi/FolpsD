"""Subprocess worker for backend tests: ``python tests/backend_worker.py {numpy|jax} OUT.npz``.

The FolpsD backend is fixed at import time, so NumPy and JAX predictions are produced in
separate processes from identical inputs (including an explicit no-wiggle spectrum, so the
backend-specific no-wiggle algorithms do not enter the comparison).
"""
import os
import sys
from pathlib import Path

backend, out = sys.argv[1], sys.argv[2]
os.environ["FOLPS_BACKEND"] = backend
if backend == "jax":
    import jax
    jax.config.update("jax_enable_x64", True)
import numpy as onp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.conftest import COSMO, PARS  # noqa: E402

import folps  # noqa: E402
from folps import (MatrixCalculator, NonLinearPowerSpectrumCalculator, RSDMultipolesPowerSpectrumCalculator,  # noqa: E402
                   SelectionBias, build_selection_tables, djs_coefficients)
from folps.selection_templates import _components  # noqa: E402

k, pk = onp.loadtxt(ROOT / "folps" / "inputpkT.txt", unpack=True)
pknow = onp.load(sys.argv[3]) if len(sys.argv) > 3 else None
mm = MatrixCalculator(A_full=True).get_mmatrices()
nl = NonLinearPowerSpectrumCalculator(mmatrices=mm, kernels="EdS", **COSMO)
table, table_now = nl.calculate_loop_table(k=k, pklin=pk, pknow=pknow, cosmo=None, **COSMO)
tabs = build_selection_tables(nl)
calc = RSDMultipolesPowerSpectrumCalculator(model="EFT")
calc._printed_model_damping_pk = True
sel = SelectionBias(0.21, -0.33, 0.27, 0.45, -0.52, 0.31, -0.23, 0.36, 0.19)
kobs = onp.asarray(table[0])[10:105]
res = {}
res["legacy"] = onp.asarray(calc.get_rsd_pkell(kobs, 1.0, 1.0, PARS, table, table_now, damping=None))
res["selected"] = onp.asarray(calc.get_rsd_pkell(kobs, 1.0, 1.0, PARS, table, table_now, damping=None,
                                                 selection=sel, selection_tables=tabs))
res["selected_ap"] = onp.asarray(calc.get_rsd_pkell(kobs, 1.02, 0.99, PARS, table, table_now, damping=None,
                                                    selection=sel, selection_tables=tabs))
res["legacy_ap"] = onp.asarray(calc.get_rsd_pkell(kobs, 1.02, 0.99, PARS, table, table_now, damping=None))
res["I22"] = onp.asarray(tabs.full.I22)
res["J13"] = onp.asarray(tabs.full.J13)

# contraction-level parity: identical host input arrays on both backends
ref = onp.load(ROOT / "tests" / "data" / "backend_contraction_input.npz") if (ROOT / "tests" / "data" / "backend_contraction_input.npz").exists() else None
if ref is not None:
    from folps.selection_templates import SelectionLoopTable
    bnp = folps.folps.np
    T = SelectionLoopTable(bnp.asarray(ref["k"]), bnp.asarray(ref["pk"]), bnp.asarray(ref["I22"]),
                           bnp.asarray(ref["J13"]), float(ref["s2"]), 0.0, 0.0, float(ref["f0"]), {})
    c = djs_coefficients(1.6, -0.4, -0.2, 0.1, sel)
    c0 = djs_coefficients(1.6, -0.4, -0.2, 0.1, SelectionBias())
    kk = bnp.asarray(ref["k"])
    mu = bnp.linspace(0.0, 1.0, kk.size)
    d22, d13 = _components(T.pk, T.I22, T.J13, T, c, c0, kk, mu)
    res["contract_d22"], res["contract_d13"] = onp.asarray(d22), onp.asarray(d13)

if backend == "jax":
    import jax.numpy as jnp
    kj = jnp.asarray(kobs[::8])

    def model(theta):
        s = SelectionBias(*[theta[i] for i in range(9)])
        p = list(PARS)
        p[0] = theta[9]
        out = calc.get_rsd_pkell(kj, theta[10], theta[11], p, table, table_now, damping=None,
                                 selection=s, selection_tables=tabs)
        return out
    theta0 = jnp.asarray(list(sel) + [PARS[0], 1.02, 0.99])
    res["theta0"] = onp.asarray(theta0)
    res["jac"] = onp.asarray(jax.jacfwd(model)(theta0))
    res["jit"] = onp.asarray(jax.jit(model)(theta0))
    res["eager"] = onp.asarray(model(theta0))
    fd = {}
    for h in (1e-2, 1e-3, 1e-4):
        cols = []
        for i in range(theta0.size):
            e = onp.zeros(theta0.size)
            e[i] = h
            cols.append((onp.asarray(model(theta0 + e)) - onp.asarray(model(theta0 - e))) / (2 * h))
        fd[h] = onp.stack(cols, -1)
    for h, v in fd.items():
        res[f"fd_{h:g}"] = v

    # derivative w.r.t. a linear-spectrum amplitude: P_L -> A P_L (table construction differentiated)
    def amp_model(A):
        n2 = NonLinearPowerSpectrumCalculator(mmatrices=mm, kernels="EdS", **COSMO)
        t2, t2n = n2.calculate_loop_table(k=k, pklin=A * pk, pknow=A * pknow, cosmo=None, **COSMO)
        tb = build_selection_tables(n2)
        return calc.get_rsd_pkell(kj, 1.0, 1.0, PARS, t2, t2n, damping=None, selection=sel, selection_tables=tb)
    res["dA_ad"] = onp.asarray(jax.jacfwd(amp_model)(1.0))
    hA = 1e-4
    res["dA_fd"] = (onp.asarray(amp_model(1.0 + hA)) - onp.asarray(amp_model(1.0 - hA))) / (2 * hA)
onp.savez(out, **res)
print("worker done", backend)
