<!-- <p align="center">
    <img src="https://github.com/henoriega/FOLPSpipe/blob/main/folps_logo.png" width="600" height="300" alt="FOLPS logo">
</p> -->

# FOLPS

Python code for computing galaxy redshift-space power spectrum and bispectrum multipoles, with both NumPy and JAX backends. This is Folps v2.

[![arXiv](https://img.shields.io/badge/arXiv-2208.02791-red)](https://arxiv.org/abs/2208.02791)
[![arXiv](https://img.shields.io/badge/arXiv-2604.08895-red)](https://arxiv.org/abs/2604.08895)

## Requirements

The package relies on the following Python dependencies:

- numpy
- scipy

If jax is used:
- jax
- interpax

<!-- 
## Installation

Clone and install from source:

```bash
git clone https://github.com/henoriega/FOLPSpipe.git
cd FOLPSpipe_final_branch
python -m pip install -e .
```

You can replace `-e` with a standard installation:

```bash
python -m pip install .
```
 -->

## Backend Selection

Set the backend before importing the package:

```python
import os
os.environ["FOLPS_BACKEND"] = "numpy"  # or "jax"
```

## Quick Example

```python
import os
os.environ["FOLPS_BACKEND"] = "numpy"

import numpy as np
from folps import (
    BispectrumCalculator,
    MatrixCalculator,
    NonLinearPowerSpectrumCalculator,
    RSDMultipolesPowerSpectrumCalculator,
)

k, pk = np.loadtxt("folps/inputpkT.txt", unpack=True)
cosmo = dict(z=0.3, h=0.6711, Omega_m=0.3211636237981114, f0=0.6880638641959066, fnu=0.004453689063655854)

matrix = MatrixCalculator(A_full=True, save_dir="folps/output_matrices")
mmatrices = matrix.get_mmatrices()
nonlinear = NonLinearPowerSpectrumCalculator(mmatrices=mmatrices, kernels="fk", **cosmo)
table, table_now = nonlinear.calculate_loop_table(k=k, pklin=pk, cosmo=None, **cosmo)

# Pk parameters
PshotP = 1.0 / 0.0002118763
b1 = 1.645
b2 = -0.46
bs2 = -4.0 / 7.0 * (b1 - 1.0)
b3nl = 32.0 / 315.0 * (b1 - 1.0)
alpha0, alpha2, alpha4, ctilde = 3.0, -28.9, 0.0, 0.0
alphashot0 = 0.08
alphashot2 = -8.1
X_FoG_Pk = 0
pars = [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, PshotP, X_FoG_Pk]

multipoles = RSDMultipolesPowerSpectrumCalculator(model="FOLPSD")
P0, P2, P4 = multipoles.get_rsd_pkell(
    kobs=table[0],
    qpar=1.0,
    qper=1.0,
    pars=pars,
    table=table,
    table_now=table_now,
    bias_scheme="folps",
    damping="lor",
)

# Sugiyama bispectrum on diagonal configurations (k1 = k2)
k_ev = np.linspace(0.01, 0.2, num=40)
k1k2_pairs = np.vstack([k_ev, k_ev]).T
pars_bk = [b1, b2, bs2, 0.0, 0.0, 0.0, 0.0, 1.0]
k_pkl_pklnw = np.array([table[0], table[1], table_now[1]])
bispectrum = BispectrumCalculator(model="FOLPSD")

B000, B110, B220, B202, B022, B112 = bispectrum.Sugiyama_Bell(
    f=nonlinear.f0,
    bpars=pars_bk,
    k_pkl_pklnw=k_pkl_pklnw,
    k1k2pairs=k1k2_pairs,
    qpar=1.0,
    qper=1.0,
    precision=[10, 10, 10],
    damping="lor",
    multipoles=["B000", "B110", "B220", "B202", "B022", "B112"],
    renormalize=True,
    interpolation_method="linear",
    bias_scheme="folps",
)

print("Pk:", P0.shape, P2.shape, P4.shape)
print("Sugiyama Bk:", B000.shape, B202.shape)
```

## One-loop line-of-sight selection (EdS, `model="EFT"`)

The branch `feature/selection-1loop-eds` adds the complete renormalized one-loop galaxy
auto power spectrum with all nine independent line-of-sight (LOS) selection operators
through third order (P22 and P13, including their interference with ordinary bias and
RSD), for **EdS kernels with scale-independent growth** in the undamped EFT branch, with
BAO IR resummation and AP projection. Calls without `selection` are unchanged.

```python
import os
os.environ["FOLPS_BACKEND"] = "numpy"
import numpy as np
from folps import (MatrixCalculator, NonLinearPowerSpectrumCalculator, RSDMultipolesPowerSpectrumCalculator,
                   SelectionBias, build_selection_tables)

k, pk = np.loadtxt("folps/inputpkT.txt", unpack=True)
cosmo = dict(z=0.3, h=0.6711, Omega_m=0.3211636237981114, f0=0.6880638641959066, fnu=0.004453689063655854)
nonlinear = NonLinearPowerSpectrumCalculator(mmatrices=MatrixCalculator(A_full=True).get_mmatrices(),
                                             kernels="EdS", **cosmo)
table, table_now = nonlinear.calculate_loop_table(k=k, pklin=pk, cosmo=None, **cosmo)
sel_tables = build_selection_tables(nonlinear)            # named full + no-wiggle selection templates

pars = [1.645, -0.46, -0.3686, 0.0655, 3.0, -28.9, 1.5, 0.0, 0.08, -8.1, 1 / 0.0002118763, 0.0]
selection = SelectionBias(s_eta=0.15, s_delta_eta=-0.3, s_eta2=0.2, s_KK_par=0.4, s_Pi2_par=-0.5,
                          s_delta_Pi2_par=0.3, s_eta_Pi2_par=-0.2, s_KPi2_par=0.3, s_Pi3_par=0.2)
P0, P2, P4 = RSDMultipolesPowerSpectrumCalculator(model="EFT").get_rsd_pkell(
    np.geomspace(0.01, 0.3, 50), 1.0, 1.0, pars, table, table_now, damping=None,
    selection=selection, selection_tables=sel_tables)
```

`SelectionBias()` (all zero) reproduces the unselected prediction bit for bit. Optional
`EFTExtension(alpha6, alphashot2_iso, alphashot2_mu4)` and `PhysicalDerivativeBias` extend
the derivative and noise sectors. Selected cross spectra, `kernels="fk"` and FoG-damped
models raise errors in selection mode. The historical tree-only `bA` argument is kept as is.

* tutorial: [notebooks/selection_1loop_tutorial.ipynb](notebooks/selection_1loop_tutorial.ipynb)
* equations and conventions: [docs/selection_1loop_equations.md](docs/selection_1loop_equations.md)
* validation report: [docs/selection_1loop_validation.md](docs/selection_1loop_validation.md)
* sources and recorded corrections: [docs/selection_1loop_sources.md](docs/selection_1loop_sources.md)
* tests: `python -m pytest tests` (add `--runslow` for the direct-quadrature checks)

## Tests and Timing Benchmarks

The [folps](folps/) directory includes several `test_*.py` scripts that demonstrate end-to-end execution for both the power spectrum and bispectrum using NumPy and JAX.

Main scripts:

- [folps/test_folps_numpy.py](folps/test_folps_numpy.py)
- [folps/test_folps_jax.py](folps/test_folps_jax.py)
- [folps/test_compare_folps_numpy_vs_jax.py](folps/test_compare_folps_numpy_vs_jax.py)

## Notebooks

The [notebooks](notebooks/) directory contains worked examples showing how to run:

- Power spectrum calculations
- Bispectrum calculations in the Scoccimarro and Sugiyama bases
- Windowed bispectrum calculations including survey geometry effects

Main notebooks:

- [notebooks/example_folps_numpy.ipynb](notebooks/example_folps_numpy.ipynb)
- [notebooks/example_folps_jax.ipynb](notebooks/example_folps_jax.ipynb)
- [notebooks/B000_B202_windowing.ipynb](notebooks/B000_B202_windowing.ipynb)

## Developers

- [Hernan E. Noriega](mailto:henoriega@icf.unam.mx)
- [Alejandro Aviles](mailto:avilescervantes@gmail.com)

Arnaud de Mattia: support with JAX-related development

Prakhar Bansal: integration with desilike.

## References

FOLPS theory: [https://arxiv.org/abs/2007.06508](https://arxiv.org/abs/2007.06508), [https://arxiv.org/abs/2106.13771](https://arxiv.org/abs/2106.13771)  

Folps v1 original release: [https://arxiv.org/abs/2208.02791](https://arxiv.org/abs/2208.02791)   

Including bispectrum and JAX capabilities: [https://arxiv.org/abs/2604.08895](https://arxiv.org/abs/2604.08895)   


## Acknowledgements

We acknowledge financial support from grant DGAPA-PAPIIT IA101825 and SECIHITI grants CBF2023-2024-162 and CBF-2025-I-2795
