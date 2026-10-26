#!/usr/bin/env python
"""Minimal adapter between the FolpsD selection model and a Fisher / likelihood pipeline.

Three models on one parameter dictionary:

* ``"none"``     - legacy FolpsD EFT (``selection=None``);
* ``"tree_bA"``  - historical tree-only anisotropic bias (legacy ``bA`` argument; the tree
                   Kaiser factor only), projected with FolpsD's angular rule;
* ``"full"``     - complete one-loop selection (``SelectionBias``).

The adapter returns the multipole data vector in ell-major order and finite-difference
Jacobians.  **The demo below uses a diagonal toy covariance.** It is not a DESI forecast:
the LRG2 data vector, covariance, window and cuts must be supplied by the existing
pipeline, which also applies the window.  See docs/selection_1loop_integration.md.

Usage::

    python scripts/selection_fisher_adapter_example.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("FOLPS_BACKEND", "numpy")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.special import eval_legendre  # noqa: E402

from folps import (MatrixCalculator, NonLinearPowerSpectrumCalculator, RSDMultipolesPowerSpectrumCalculator,  # noqa: E402
                   SelectionBias, build_selection_tables)

LEGACY_ORDER = ("b1", "b2", "bs2", "b3nl", "alpha0", "alpha2", "alpha4", "ctilde", "alphashot0", "alphashot2",
                "PshotP", "X_FoG_p")


class SelectionPowerSpectrumModel:
    """Build tables once per cosmology; evaluate multipoles per parameter point."""

    def __init__(self, k_lin, pk_lin, cosmo, nfftlog=128, ells=(0, 2, 4), nmu=6, pknow=None):
        mm = MatrixCalculator(nfftlog=nfftlog, A_full=True, use_TNS_model=False).get_mmatrices()
        self.nl = NonLinearPowerSpectrumCalculator(mmatrices=mm, kernels="EdS", **cosmo)
        self.table, self.table_now = self.nl.calculate_loop_table(k=k_lin, pklin=pk_lin, pknow=pknow, cosmo=None, **cosmo)
        self.sel_tables = build_selection_tables(self.nl)
        self.calc = RSDMultipolesPowerSpectrumCalculator(model="EFT")
        self.calc._printed_model_damping_pk = True
        self.ells, self.nmu = tuple(ells), nmu

    def _pars(self, p):
        return [p[name] for name in LEGACY_ORDER]

    def multipoles(self, kobs, p, mode="full"):
        """``p``: dict with the legacy names, ``qpar``, ``qper``, the nine ``s_*`` coordinates
        (mode 'full') or ``bA`` (mode 'tree_bA').  Returns shape (len(ells), len(kobs))."""
        pars, qpar, qper = self._pars(p), p.get("qpar", 1.0), p.get("qper", 1.0)
        if mode == "none":
            return self.calc.get_rsd_pkell(kobs, qpar, qper, pars, self.table, self.table_now, damping=None,
                                           nmu=self.nmu, ells=self.ells)
        if mode == "full":
            sel = SelectionBias(**{f: p.get(f, 0.0) for f in SelectionBias._fields})
            return self.calc.get_rsd_pkell(kobs, qpar, qper, pars, self.table, self.table_now, damping=None,
                                           nmu=self.nmu, ells=self.ells, selection=sel, selection_tables=self.sel_tables)
        if mode == "tree_bA":
            # get_rsd_pkell does not expose bA: same symmetric Gauss rule as get_rsd_pkell
            x, w = np.polynomial.legendre.leggauss(2 * self.nmu)
            mu_obs, w = x[self.nmu:], (w[self.nmu:] + w[self.nmu - 1::-1]) / 2.0
            kk = kobs[:, None] * np.ones((1, mu_obs.size))
            kap, muap = self.calc.k_ap(kk, mu_obs[None, :], qpar, qper), self.calc.mu_ap(mu_obs, qpar, qper)[None, :] * np.ones_like(kk)
            pkmu = self.calc.get_rsd_pkmu(kap, muap, pars, self.table, self.table_now, damping=None, bA=p["bA"])
            return np.array([np.sum(pkmu * w * (2 * l + 1) * eval_legendre(l, mu_obs), axis=-1) / (qpar * qper ** 2)
                             for l in self.ells])
        raise ValueError(mode)

    def data_vector(self, kobs, p, mode="full"):
        return self.multipoles(kobs, p, mode).ravel()     # ell-major

    def jacobian(self, kobs, p, names, mode="full", step=1e-3):
        cols = []
        for n in names:
            pp, pm = dict(p), dict(p)
            h = step * max(1.0, abs(p.get(n, 0.0)))
            pp[n] = p.get(n, 0.0) + h
            pm[n] = p.get(n, 0.0) - h
            cols.append((self.data_vector(kobs, pp, mode) - self.data_vector(kobs, pm, mode)) / (2 * h))
        return np.stack(cols, -1)


def main():
    k, pk = np.loadtxt(ROOT / "folps" / "inputpkT.txt", unpack=True)
    cosmo = dict(z=0.3, h=0.6711, Omega_m=0.3211636237981114, f0=0.6880638641959066, fnu=0.004453689063655854)
    model = SelectionPowerSpectrumModel(k, pk, cosmo)
    p = dict(b1=1.645, b2=-0.46, bs2=-0.3686, b3nl=0.0655, alpha0=3.0, alpha2=-28.9, alpha4=1.5, ctilde=0.0,
             alphashot0=0.08, alphashot2=-8.1, PshotP=1 / 0.0002118763, X_FoG_p=0.0, qpar=1.0, qper=1.0, bA=0.2)
    kobs = np.arange(0.02, 0.20, 0.005)
    # TOY diagonal covariance (V = 1 (Gpc/h)^3, nbar = 2e-4, dk = 0.005): NOT a DESI covariance
    V, nbar, dk = 1e9, 2e-4, 0.005
    P0 = model.multipoles(kobs, p, "none")[0]
    sig = np.concatenate([(2 * l + 1) * (P0 + 1 / nbar) * np.sqrt(2 / (V * 4 * np.pi * kobs ** 2 * dk / (2 * np.pi) ** 3))
                          for l in model.ells])
    for mode, names in (("none", ["b1", "qpar", "qper"]), ("tree_bA", ["b1", "bA", "qpar", "qper"]),
                        ("full", ["b1", "s_eta", "s_delta_eta", "s_Pi2_par", "qpar", "qper"])):
        J = model.jacobian(kobs, p, names, mode)
        F = (J / sig[:, None] ** 2).T @ J
        w = np.linalg.eigvalsh(F)
        print(f"[{mode:8s}] params {names}: Fisher rank {int(np.sum(w > 1e-10 * w.max()))}/{len(names)}, "
              f"cond {w.max() / max(w.min(), 1e-300):.1e}, toy sigma(qpar) = {np.sqrt(np.linalg.inv(F)[names.index('qpar'), names.index('qpar')]):.4f}")


if __name__ == "__main__":
    main()
