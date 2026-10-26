#!/usr/bin/env python
"""Independent validation of the selection one-loop templates (slow; not run by pytest).

* direct quadrature (``folps.selection_reference``: oracle kernels, full azimuthal
  integration, exact redshift-space map) of the selected P22 and bare P13 at FolpsD grid
  points for a generic nonzero selection (all nine coordinates on) and for zero selection;
* comparison with the FFTLog templates at ``nfftlog`` = 128 and 256;
* writes ``tests/data/selection_reference_fixture.npz`` (reference values + metadata) and
  ``docs/selection_1loop_validation_reference.json`` (residual summary).

Usage::

    python scripts/validate_selection_reference.py [--quick]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("FOLPS_BACKEND", "numpy")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from folps import (MatrixCalculator, NonLinearPowerSpectrumCalculator, SelectionBias,  # noqa: E402
                   build_selection_tables, djs_coefficients, folps_bias_to_djs)
from folps import selection_coefficients as sc  # noqa: E402
from folps.selection_reference import (LinearSpectrum, direct_p13_ratio, direct_p22,  # noqa: E402
                                       exact_map_field_from_djs)
from folps.selection_templates import _components  # noqa: E402
from tests.conftest import COSMO, PARS  # noqa: E402

SELECTION = SelectionBias(s_eta=0.21, s_delta_eta=-0.33, s_eta2=0.27, s_KK_par=0.45, s_Pi2_par=-0.52,
                          s_delta_Pi2_par=0.31, s_eta_Pi2_par=-0.23, s_KPi2_par=0.36, s_Pi3_par=0.19)
K_INDEX = (40, 60, 75, 85, 95, 103, 110, 115)
MUS = (0.15, 0.6, 0.95)
QUAD = dict(p22=dict(nq=300, nnu=40, nphi=24), p13=dict(nq=800, nnu=96, nphi=16))


def tables(nfftlog):
    k, pk = np.loadtxt(ROOT / "folps" / "inputpkT.txt", unpack=True)
    mm = MatrixCalculator(nfftlog=nfftlog, A_full=True).get_mmatrices()
    nl = NonLinearPowerSpectrumCalculator(mmatrices=mm, kernels="EdS", **COSMO)
    nl.calculate_loop_table(k=k, pklin=pk, cosmo=None, **COSMO)
    return nl, build_selection_tables(nl)


def fast_point(T, i, mu, c):
    ip = (T.pk[i:i + 1], T.I22[:, :, i:i + 1], T.J13[:, i:i + 1])
    p22, p13 = _components(*ip, T, c, None, np.array([T.k[i]]), np.array([mu]))
    return float(p22[0]), float(p13[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="two k values only")
    args = ap.parse_args()
    kidx = (60, 95) if args.quick else K_INDEX
    t0 = time.time()
    nl128, tabs128 = tables(128)
    nl256, tabs256 = tables(256)
    T128, T256 = tabs128.full, tabs256.full
    P = LinearSpectrum(*nl128.inputpkT)
    f = float(T128.f0)
    b1, b2, bK2, btd = folps_bias_to_djs(*PARS[:4])
    c = djs_coefficients(b1, b2, bK2, btd, SELECTION)
    c0 = djs_coefficients(b1, b2, bK2, btd, SelectionBias())
    FB, FB0 = exact_map_field_from_djs(c._asdict()), exact_map_field_from_djs(c0._asdict())
    rows = []
    for i in kidx:
        kk = float(T128.k[i])
        for mu in MUS:
            t1 = time.time()
            r22 = direct_p22(kk, mu, P, FB, f, **QUAD["p22"])
            r22_0 = direct_p22(kk, mu, P, FB0, f, **QUAD["p22"])
            Z1 = c.b1 - c.b_eta * f * mu ** 2
            Z10 = c0.b1 - c0.b_eta * f * mu ** 2
            r13 = 2 * Z1 * float(P(kk)) * direct_p13_ratio(kk, mu, P, FB, f, **QUAD["p13"])
            r13_0 = 2 * Z10 * float(P(kk)) * direct_p13_ratio(kk, mu, P, FB0, f, **QUAD["p13"])
            row = dict(i=i, k=kk, mu=mu, ref_p22=r22, ref_p22_0=r22_0, ref_p13=r13, ref_p13_0=r13_0,
                       seconds=round(time.time() - t1, 1))
            for N, T in ((128, T128), (256, T256)):
                a22, a13 = fast_point(T, i, mu, c)
                z22, z13 = fast_point(T, i, mu, c0)
                row.update({f"fast{N}_p22": a22, f"fast{N}_p13": a13, f"fast{N}_p22_0": z22, f"fast{N}_p13_0": z13})
            rows.append(row)
            print(f"k={kk:.4f} mu={mu}: P22 ref {r22:.4f} fast128 {row['fast128_p22']:.4f} fast256 {row['fast256_p22']:.4f} | "
                  f"P13 ref {r13:.4f} fast128 {row['fast128_p13']:.4f} fast256 {row['fast256_p13']:.4f} ({row['seconds']}s)",
                  flush=True)
    # residuals with a stable denominator: max over the k range of |loop| at the same mu
    summary = {}
    for N in (128, 256):
        for mu in MUS:
            sub = [r for r in rows if r["mu"] == mu]
            scale = max(abs(r["ref_p22"] + r["ref_p13"]) for r in sub)
            for comp in ("p22", "p13"):
                for tag, ref_key, fast_key in (("total", f"ref_{comp}", f"fast{N}_{comp}"),):
                    err = [abs(r[fast_key] - r[ref_key]) / scale for r in sub]
                    summary[f"N{N}_mu{mu}_{comp}_{tag}_max_over_loopscale"] = max(err)
                dref = [r[f"ref_{comp}"] - r[f"ref_{comp}_0"] for r in sub]
                dfast = [r[f"fast{N}_{comp}"] - r[f"fast{N}_{comp}_0"] for r in sub]
                dscale = max(abs(x) for x in dref)
                summary[f"N{N}_mu{mu}_{comp}_increment_max_over_incscale"] = max(abs(a - b) for a, b in zip(dfast, dref)) / dscale
    meta = dict(commit=subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                input_sha256=hashlib.sha256((ROOT / "folps" / "inputpkT.txt").read_bytes()).hexdigest(),
                coefficients_sha256=sc.COEFFICIENT_DATA_SHA256, cosmo=COSMO, pars=PARS, selection=SELECTION._asdict(),
                quadrature=QUAD, field="exact redshift-space map of the rest-frame density (oracle)",
                p13_scheme="bare; UV constant (sigma_0^2) of the angular-averaged kernel subtracted",
                p22_scheme="zero-lag constant subtracted", seconds=round(time.time() - t0, 1))
    if not args.quick:
        keys = sorted(rows[0])
        np.savez(ROOT / "tests" / "data" / "selection_reference_fixture.npz",
                 **{k: np.array([r[k] for r in rows]) for k in keys if k not in ("seconds",)},
                 metadata_json=json.dumps(meta))
    (ROOT / "docs" / ("selection_1loop_validation_reference.json" if not args.quick else "selection_1loop_validation_quick.json")
     ).write_text(json.dumps(dict(metadata=meta, summary=summary, rows=rows), indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
