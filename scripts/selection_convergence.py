#!/usr/bin/env python
"""FFTLog convergence and timing study of the selection templates (writes a JSON summary).

Varies the FFTLog resolution (nfftlog = 128, 256, 512), the local-class P22 tilt, and the
input-spectrum domain (truncating the tabulated input at k_max = 1 instead of 2 h/Mpc so that
FolpsD's power-law extrapolation takes over), for both the full-shape and no-wiggle tables.
Reference: nfftlog = 512 with the default tilt.  Residuals are reported separately for the
selected P22 and P13 as max |delta X| / max |X| over the k range at mu = 0.1, 0.6, 1.0 (the sum
P22 + P13 partly cancels, so it is not a stable denominator).

Usage::  python scripts/selection_convergence.py [--out docs/selection_1loop_convergence.json]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("FOLPS_BACKEND", "numpy")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import folps.selection_templates as st  # noqa: E402
from folps import (MatrixCalculator, NonLinearPowerSpectrumCalculator, RSDMultipolesPowerSpectrumCalculator,  # noqa: E402
                   SelectionBias, build_selection_tables, djs_coefficients, folps_bias_to_djs)
from folps import selection_matrices as smod  # noqa: E402
from tests.conftest import COSMO, PARS  # noqa: E402

SEL = SelectionBias(0.21, -0.33, 0.27, 0.45, -0.52, 0.31, -0.23, 0.36, 0.19)
MUS = (0.1, 0.6, 1.0)


def loops(N, kcut=None, tilt=None):
    k, pk = np.loadtxt(ROOT / "folps" / "inputpkT.txt", unpack=True)
    if kcut is not None:
        k, pk = k[k <= kcut], pk[k <= kcut]
    old = st.P22_TILT_LOCAL
    if tilt is not None:
        st.P22_TILT_LOCAL = tilt
    try:
        t0 = time.perf_counter()
        mm = MatrixCalculator(nfftlog=N, A_full=True).get_mmatrices()
        nl = NonLinearPowerSpectrumCalculator(mmatrices=mm, kernels="EdS", **COSMO)
        nl.calculate_loop_table(k=k, pklin=pk, cosmo=None, **COSMO)
        t1 = time.perf_counter()
        tabs = build_selection_tables(nl)
        t2 = time.perf_counter()
    finally:
        st.P22_TILT_LOCAL = old
    c = djs_coefficients(*folps_bias_to_djs(*PARS[:4]), SEL)
    out = {}
    for role, T in (("full", tabs.full), ("nowiggle", tabs.nowiggle)):
        for mu in MUS:
            p22, p13 = st.selection_loop(T.k, np.full_like(T.k, mu), T, c)
            out[(role, "P22", mu)] = p22
            out[(role, "P13", mu)] = p13
    return T.k, out, dict(legacy_tables_s=t1 - t0, selection_tables_s=t2 - t1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "selection_1loop_convergence.json"))
    args = ap.parse_args()
    K, ref, _ = loops(512)
    sel = (K > 0.005) & (K < 0.45)
    lowk = (K > 0.005) & (K < 0.25)
    summary, timings = {}, {}
    cases = {"N128": dict(N=128), "N256": dict(N=256), "N128_tilt-0.85": dict(N=128, tilt=-0.85),
             "N128_tilt-0.95": dict(N=128, tilt=-0.95), "N256_tilt-0.85": dict(N=256, tilt=-0.85),
             "N256_tilt-0.95": dict(N=256, tilt=-0.95), "N256_kin<=1": dict(N=256, kcut=1.0)}
    for name, kw in cases.items():
        smod._MEMORY_CACHE.clear()
        _, L, tm = loops(**kw)
        timings[name + "_cold"] = tm
        for (role, comp, mu), v in L.items():
            r = ref[(role, comp, mu)]
            summary[f"{name}|{role}|{comp}|mu={mu}|0.005<k<0.45"] = float(np.max(np.abs(v - r)[sel]) / np.max(np.abs(r)[sel]))
            summary[f"{name}|{role}|{comp}|mu={mu}|0.005<k<0.25"] = float(np.max(np.abs(v - r)[lowk]) / np.max(np.abs(r)[lowk]))
    # timing of the prediction itself (warm)
    k, pk = np.loadtxt(ROOT / "folps" / "inputpkT.txt", unpack=True)
    mm = MatrixCalculator(A_full=True).get_mmatrices()
    nl = NonLinearPowerSpectrumCalculator(mmatrices=mm, kernels="EdS", **COSMO)
    t, tn = nl.calculate_loop_table(k=k, pklin=pk, cosmo=None, **COSMO)
    tabs = build_selection_tables(nl)
    calc = RSDMultipolesPowerSpectrumCalculator(model="EFT")
    calc._printed_model_damping_pk = True
    kobs = np.geomspace(0.01, 0.3, 60)
    for label, kw in (("legacy_pkell_60k_ms", {}), ("selected_pkell_60k_ms", dict(selection=SEL, selection_tables=tabs))):
        t0 = time.perf_counter()
        for _ in range(30):
            calc.get_rsd_pkell(kobs, 1.0, 1.0, PARS, t, tn, damping=None, **kw)
        timings[label] = 1e3 * (time.perf_counter() - t0) / 30
    t0 = time.perf_counter()
    for _ in range(10):
        build_selection_tables(nl)
    timings["selection_tables_warm_ms"] = 1e3 * (time.perf_counter() - t0) / 10
    Path(args.out).write_text(json.dumps(dict(reference="nfftlog=512, tilt -0.9", summary=summary, timings=timings), indent=1))
    print(json.dumps(dict(summary=summary, timings=timings), indent=1))


if __name__ == "__main__":
    main()
