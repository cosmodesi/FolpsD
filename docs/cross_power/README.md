# FOLPS Two-Tracer Cross-Power Spectrum

## Purpose

This directory documents the equal-time two-tracer redshift-space power spectrum implemented in FOLPS. The motivating use case is a cross-correlation such as LRG3 x ELG1 over an overlapping redshift interval, where two biased tracer samples respond to the same large-scale matter and velocity fields but carry different bias parameters, stochastic fields, and pair-level EFT nuisance parameters.

The central object is the redshift-space cross spectrum \(P^{AB}(k,\mu)\) and its even multipoles. It is not generally equivalent to \(\sqrt{P^{AA}P^{BB}}\). That geometric mean is exact only in a restrictive deterministic linear limit. At one loop, the density and velocity contractions are bilinear in the two tracer bias vectors, the \(A\), \(D\), and \(G\) redshift-space terms need endpoint-aware polarization, and the EFT/stochastic sector is a genuinely pair-level model.

The current public entry points remain the standard FOLPS power-spectrum methods. Cross mode is activated by passing `pars_b` and an explicit `cross_nuisance` vector. The implementation is intended as a clean first equal-time power-spectrum path for two tracers, not as a full DESI likelihood analysis.

The practical reason for this documentation package is to make the new capability reviewable without forcing collaborators to reconstruct the history from several audit files. A reader should be able to answer three questions from the top-level files: what physical quantity is implemented, how the public API should be called, and where the detailed formulas and provenance live. The archived notes remain available, but the normal entry points are this README, the consolidated technical reference, and the PDF note.

## Development status

This branch extends the FolpsD code on top of the repository `main` branch with equal-time two-tracer redshift-space cross-power spectra. The branch-preparation base is `origin/main` commit `94ee3bfbbf85e0c679be73e1038ccfd9f8523434`.

It is intentionally based on `main`, not on the separate `adematti-damping` development branch. Consequently, this branch does not implement or claim support for all damping prescriptions, flags, or modeling choices available in `adematti-damping`.

Cross damping is implemented only for damping functionality available in this `main`-based branch. Two cross prescriptions are supported:

- `cross_damping_mode="single"`: default, using the pair-level \(X_{\rm FoG}^{AB}\);
- `cross_damping_mode="geometric"`: using \(\sqrt{W_AW_B}\).

Before any future merge, damping functionality should be reviewed against the then-current target branch.

## Compatibility table

| Capability | Auto spectrum on this branch | Cross spectrum on this branch |
| --- | ---: | ---: |
| Standard EFT without phenomenological damping | Yes | Yes |
| Damping functions available on `main` | `exp`, `lor`, `vdg` | `exp`, `lor`, `vdg` |
| Single cross damping | N/A | Yes |
| Geometric cross damping | N/A | Yes |
| Full `adematti-damping` feature parity | No | No |
| Cross bispectrum | No | No |

## Current capabilities

The implementation supports the four spectra naturally needed for a two-tracer power-spectrum block:

```text
P^{AA}, P^{AB}, P^{BA}, P^{BB}
```

The same shared pair-level contraction is used for auto and cross spectra, so the auto spectrum is recovered by setting \(A=B\). Exchange symmetry \(P^{AB}=P^{BA}\) is tested directly.

Implemented physics components include:

- one-loop density-density, density-velocity, and velocity-velocity contributions;
- the full biased \(A\) term, including the `A_full=True` \(b_2\) and \(b_{s^2}\) rows;
- the \(D\) and \(G\) redshift-space terms;
- IR resummation using the existing FOLPS wiggle/no-wiggle tables;
- Alcock-Paczynski remapping and standard even-multipole quadrature;
- pair-level EFT counterterms and stochastic parameters;
- `priordoc` and canonical FOLPS bias conventions for both tracers;
- NumPy and JAX backend support through the existing backend selection;
- FolpsD cross damping for `model="FOLPSD"` with `damping="exp"`, `"lor"`, or `"vdg"`.

The implementation reuses the existing loop tables, no-wiggle tables, AP remapping, IR-resummation machinery, multipole integration, and matrix cache conventions. There is no new cross-specific matrix file, and there is no separate `folpsX.py` implementation. The non-marginalized auto path and cross path share one pair-level contraction, which is important for maintainability: the same algebra that evaluates \(P^{AB}\) also evaluates \(P^{AA}\) when the two endpoints are set equal.

The prior-document input convention uses:

```text
(b1, b2, bK2, btd, alpha0, alpha2, alpha4, ctilde,
 alphashot0, alphashot2, PshotP, X_FoG)
```

Internally, the bias entries are converted to canonical FOLPS parameters with

$$
b_{s^2}=2b_{K^2},
\qquad
b_{3\rm nl}
=-\frac{32}{21}\left(b_{K^2}+\frac25 b_{\rm td}\right).
$$

## Assumptions

The present implementation assumes:

- equal time;
- one common effective redshift for the pair;
- the plane-parallel approximation;
- common matter and velocity fields;
- no tracer velocity bias;
- even multipoles only;
- power spectrum only;
- no wide-angle or relativistic corrections;
- no cross bispectrum;
- no survey window convolution, covariance, or likelihood integration in this directory.

These assumptions are deliberate. They keep the first implementation focused on the one-loop cross-power calculation and on preserving the existing FOLPS table, interpolation, IR, AP, and backend infrastructure.

The limitations also describe the intended scientific interpretation. The examples in this directory use illustrative LRG-like and ELG-like parameters to exercise the code and generate readable figures. They should not be interpreted as DESI best fits, recommended priors, or a complete data-vector definition. In particular, realistic use still needs a model for the overlapping redshift range, tracer selection, covariance, and likelihood treatment.

## Damping modes

For `model="FOLPSD"`, cross spectra support two public prescriptions selected by `cross_damping_mode`. The default is:

```python
cross_damping_mode="single"
```

In this mode the cross damping factor is sourced by the pair-level FoG parameter:

$$
W_{AB}=W(X_{\rm FoG}^{AB}).
$$

The parameter \(X_{\rm FoG}^{AB}\) is the final entry of the explicit `cross_nuisance` vector.

The alternative is:

```python
cross_damping_mode="geometric"
```

with

$$
W_{AB}=\sqrt{W(X_{\rm FoG}^{A})W(X_{\rm FoG}^{B})}.
$$

Here \(X_{\rm FoG}^{A}\) and \(X_{\rm FoG}^{B}\) are read from the final entries of the two tracer parameter arrays after each array has been converted through its own bias convention. In `geometric` mode, the pair-level \(X_{\rm FoG}^{AB}\) remains present in the `cross_nuisance` vector for API compatibility, but it is not used by the damping factor.

Both modes use the same production damping kernels as the auto path. The damping factor multiplies the one-loop pair contraction in the current FolpsD convention. It does not multiply the IR-resummed linear Kaiser term, the standard counterterms, the NLO `ctilde` counterterm, or the stochastic term.

## Cross nuisance vector

The cross nuisance vector layout is unchanged:

```python
(
    alpha0_ab,
    alpha2_ab,
    alpha4_ab,
    ctilde_ab,
    alphashot0_ab,
    alphashot2_ab,
    PshotP_ab,
    X_FoG_ab,
)
```

These counterterm and stochastic parameters are pair-level parameters. They are not arithmetic averages, geometric averages, or otherwise inferred values from the two auto-spectrum nuisance vectors. This is especially important for stochasticity: for disjoint catalogs the naive Poisson overlap term is zero, but the renormalized cross-stochastic contribution need not vanish.

This convention keeps the public API honest about what is being modeled. A later likelihood layer may decide to tie auto and cross EFT coefficients together through a field-level parameterization, impose covariance-positivity constraints, or fix some cross-stochastic contribution to zero. Those are analysis choices above the current FOLPS calculation. The code path documented here only requires that the pair-level vector be explicit.

## Minimal API example

The example below uses the NumPy backend, the public input linear spectrum, the existing `A_full=True` matrix cache, prior-document bias inputs for both tracers, and single FolpsD cross damping. The numerical values are illustrative only.

```python
import os
import numpy as np

os.environ["FOLPS_BACKEND"] = "numpy"

from folps import (
    MatrixCalculator,
    NonLinearPowerSpectrumCalculator,
    RSDMultipolesPowerSpectrumCalculator,
)

k_input, p_input = np.loadtxt("folps/inputpkT.txt", unpack=True)

matrix_path = "folps/output_matrices/matrices_nfftlog128_Afull-True_use_TNS-False.npy"
MatrixCalculator(A_full=True, use_TNS_model=False, save_dir=None)
mmatrices = np.load(matrix_path, allow_pickle=True).item()

nonlinear = NonLinearPowerSpectrumCalculator(
    mmatrices=mmatrices,
    kernels="fk",
    z=0.3,
    h=0.6711,
    Omega_m=0.3211636237981114,
    f0=np.float64(0.6880638641959066),
    fnu=0.004453689063655854,
)
table, table_now = nonlinear.calculate_loop_table(
    k=k_input,
    pklin=p_input,
    z=0.3,
    h=0.6711,
    Omega_m=0.3211636237981114,
    f0=np.float64(0.6880638641959066),
    fnu=0.004453689063655854,
)

pars_a = [1.645, 0.20, -2.0 / 7.0 * (1.645 - 1.0), 23.0 / 42.0 * (1.645 - 1.0),
          0.7, -1.3, 0.2, 0.0, 0.015, -0.45, 4800.0, 0.30]
pars_b = [1.10, -0.23, -2.0 / 7.0 * (1.10 - 1.0), 23.0 / 42.0 * (1.10 - 1.0),
          0.2, -0.4, 0.1, 0.0, 0.0, 0.0, 3600.0, 6.00]
cross_nuisance_ab = [0.4, -0.9, 0.15, 0.0, 0.01, -0.2, 3600.0, 1.20]

kobs = np.linspace(0.01, 0.20, 80)
multipoles = RSDMultipolesPowerSpectrumCalculator(model="FOLPSD")
p0_ab, p2_ab, p4_ab = multipoles.get_rsd_pkell(
    kobs, 1.0, 1.0, pars_a, table, table_now,
    bias_scheme="priordoc",
    bias_scheme_b="priordoc",
    pars_b=pars_b,
    cross_nuisance=cross_nuisance_ab,
    damping="vdg",
    cross_damping_mode="single",
    IR_resummation=True,
    ells=(0, 2, 4),
)
```

The one-line change for geometric damping is:

```python
cross_damping_mode="geometric"
```

## Validation summary

The current cross-power test suite and figure-generation checks cover:

- exchange symmetry, \(P^{AB}=P^{BA}\);
- the auto limit, \(P^{AB}(A=B)=P^{AA}\);
- prior-document/canonical bias equivalence;
- individual loop-row tests for density, density-velocity, velocity-velocity, reduced \(A\), full \(A\), \(D\), and \(G\) rows;
- full-\(A\) tests with asymmetric tracer biases;
- IR on/off checks;
- AP-remapped multipole checks against direct quadrature;
- NumPy/JAX comparisons;
- FolpsD damping-mode tests for `exp`, `lor`, and `vdg`;
- unchanged auto-spectrum regression.

The figures and notes here are not a validation against DESI data. Further testing with realistic tracer parameters, mocks, covariances, and likelihood integration is still required.

Two notebooks complement the documentation. [notebooks/example_cross_power_numpy.ipynb](../../notebooks/example_cross_power_numpy.ipynb) demonstrates the basic equal-time two-tracer API. [notebooks/example_cross_power_damping_numpy.ipynb](../../notebooks/example_cross_power_damping_numpy.ipynb) focuses on the FolpsD damping modes and includes checks showing which FoG parameters each mode uses. The notebooks are demonstrations, not benchmark reports.

## Directory guide

- [README.md](README.md): this colleague-facing overview and quick API entry point.
- [CROSS_POWER_TECHNICAL.md](CROSS_POWER_TECHNICAL.md): consolidated technical reference with equations, implementation map, validation summary, and provenance.
- [folps_cross_power_notes.pdf](folps_cross_power_notes.pdf): current authoritative PDF note.
- [source/](source/): authoritative LaTeX source, bibliography, plotting script, and LaTeX figure snippets.
- [figures/](figures/): generated PDF/PNG figures used by the notes.
- [tables/](tables/): generated CSV tables and numerical summaries.
- [archive/](archive/): chronological development audits and earlier implementation reviews retained for provenance.

The authoritative PDF can be rebuilt from `docs/cross_power/` with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=. source/folps_cross_power_notes.tex
```

The figure and table set can be regenerated from anywhere inside the repository with:

```bash
/opt/anaconda3/envs/aaenv/bin/python docs/cross_power/source/make_cross_power_figures.py
```

The plotting script discovers the repository root and writes outputs to `docs/cross_power/figures/` and `docs/cross_power/tables/`.

The LaTeX source is intentionally under `source/`, while the final PDF remains at the top level. This keeps collaborator-facing files easy to find while preserving the reproducible source tree. The archived Markdown files are not deleted; they are retained under `archive/` so reviewers can inspect how the derivation and implementation checks evolved.

## Recommended reading order

1. [README.md](README.md)
2. [CROSS_POWER_TECHNICAL.md](CROSS_POWER_TECHNICAL.md)
3. [folps_cross_power_notes.pdf](folps_cross_power_notes.pdf)
4. [archive/](archive/) only for audit history

## Status

The implementation is intended for further testing with realistic tracer parameters, mocks, covariance modeling, and likelihood integration. It is a documented, tested equal-time two-tracer power-spectrum capability, but it should not be described as a final DESI data-analysis result.

Current status in one sentence: FOLPS can evaluate \(P^{AA}\), \(P^{AB}\), \(P^{BA}\), and \(P^{BB}\) for equal-time two-tracer redshift-space power spectra using the existing NumPy/JAX infrastructure, with explicit pair-level nuisance parameters and optional FolpsD cross damping, subject to the assumptions listed above.
