"""Freeze reference FolpsD EFT predictions for regression tests.

This script was run on the *unmodified* upstream checkout (commit 6012d6d) before
any selection code was added, and its output ``tests/data/eft_baseline_reference.npz``
is the frozen baseline used by ``tests/test_selection_regression.py``.  Re-running it
on a modified checkout produces a new file; do not overwrite the committed fixture
unless the baseline is intentionally changed.

Usage::

    FOLPS_BACKEND=numpy python scripts/freeze_eft_baseline.py [--out PATH]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("FOLPS_BACKEND", "numpy")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from folps import (  # noqa: E402
    MatrixCalculator,
    NonLinearPowerSpectrumCalculator,
    RSDMultipolesPowerSpectrumCalculator,
)

COSMO = dict(z=0.3, h=0.6711, Omega_m=0.3211636237981114, f0=0.6880638641959066,
             fnu=0.004453689063655854)
# b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, PshotP, X_FoG_p
PARS = [1.645, -0.46, -4.0 / 7.0 * (1.645 - 1.0), 32.0 / 315.0 * (1.645 - 1.0),
        3.0, -28.9, 1.5, 0.0, 0.08, -8.1, 1.0 / 0.0002118763, 0.0]


def _git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "tests" / "data" / "eft_baseline_reference.npz"))
    args = parser.parse_args()

    k, pk = np.loadtxt(ROOT / "folps" / "inputpkT.txt", unpack=True)
    mm = MatrixCalculator(A_full=True, use_TNS_model=False).get_mmatrices()
    kobs = np.geomspace(0.005, 0.3, 40)
    mu = np.array([0.0, 0.3, 0.6, 0.9, 1.0])
    out = {"kobs": kobs, "mu": mu, "pars": np.array(PARS)}

    for kernels in ("EdS", "fk"):
        nl = NonLinearPowerSpectrumCalculator(mmatrices=mm, kernels=kernels, **COSMO)
        table, table_now = nl.calculate_loop_table(k=k, pklin=pk, cosmo=None, **COSMO)
        calc = RSDMultipolesPowerSpectrumCalculator(model="EFT")
        for ir in (True, False):
            for tag, (qpar, qper) in (("ap1", (1.0, 1.0)), ("ap2", (1.03, 0.98))):
                pl = calc.get_rsd_pkell(kobs, qpar, qper, PARS, table, table_now,
                                        bias_scheme="folps", damping=None, nmu=6,
                                        ells=(0, 2, 4), IR_resummation=ir)
                out[f"pell_{kernels}_ir{int(ir)}_{tag}"] = np.asarray(pl)
            kk, mm_ = np.meshgrid(kobs, mu, indexing="ij")
            out[f"pkmu_{kernels}_ir{int(ir)}"] = np.asarray(
                calc.get_rsd_pkmu(kk, mm_, PARS, table, table_now, IR_resummation=ir, damping=None))
        out[f"table0_{kernels}"] = np.array([np.asarray(c) for c in table[:-1] if np.ndim(c) == 1])
    meta = {"commit": _git_commit(), "numpy": np.__version__, "cosmo": COSMO,
            "input": "folps/inputpkT.txt", "model": "EFT", "A_full": True, "use_TNS_model": False}
    out["metadata_json"] = np.array(json.dumps(meta))
    np.savez_compressed(args.out, **out)
    print("wrote", args.out, "commit", meta["commit"])


if __name__ == "__main__":
    main()
