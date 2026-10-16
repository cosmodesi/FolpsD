# Cross-Power Stage 1 Audit

This is an audit-only report for extending FolpsD to the equal-time redshift-space
cross-power spectrum of two biased tracers. No physics source files or matrix
files were modified.

## 1. Repository and environment

- Working directory checked: the parent project directory.
- Repository cloned into: `<repo-root>`.
- Remote URL: `https://github.com/cosmodesi/FolpsD.git`.
- Branch: `feature/cross-power-spectrum`.
- Commit: `94ee3bfbbf85e0c679be73e1038ccfd9f8523434`.
- Commit summary: `2026-07-08 16:58:45 -0400 Enhance JAX gpu detection`.
- Working-tree status before this report: only the requested untracked
  `docs/cross_power/` files.
- Theory note copied to `docs/cross_power/folps_cross_power_theory_note.tex`.
- Theory PDF copied to `docs/cross_power/folps_cross_power_theory_note.pdf`.

Documented installation sources:

- `docs/Installation.rst:7-19` documents `python -m pip install -r requirements.txt`
  and editable `pip install -e .`.
- `requirements.txt:1-4` lists `numpy`, `jax`, `sphinx`, and `sphinx_rtd_theme`.
- `pyproject.toml:20-25` lists package dependencies `numpy`, `scipy`, `jax`,
  and `interpax`.

Environment used for the audit after user guidance:

- Python executable: `/opt/anaconda3/envs/aaenv/bin/python`.
- Python: `3.12.2`.
- NumPy: `2.1.3`.
- SciPy: `1.18.0`.
- JAX: `0.7.2`.
- jaxlib: `0.7.2`.
- interpax: `0.3.12`.
- Installed package: `folps 2.0.0`, editable location `<repo-root>`
  according to `pip show folps`.

Installation commands used:

```bash
git clone https://github.com/cosmodesi/FolpsD.git
git fetch --all --prune
git checkout main
git pull --ff-only
git checkout -b feature/cross-power-spectrum
mkdir -p docs/cross_power
cp <source-theory-note>.tex docs/cross_power/folps_cross_power_theory_note.tex
cp <source-theory-note>.pdf docs/cross_power/folps_cross_power_theory_note.pdf
/opt/anaconda3/envs/aaenv/bin/python -m pip install -r requirements.txt
/opt/anaconda3/envs/aaenv/bin/python -m pip install -e .
```

Setup notes:

- `git fetch --all --prune` and `git pull --ff-only` initially failed inside the
  sandbox with `Could not resolve host: github.com`; both succeeded after
  network approval.
- Before the user suggested `aaenv`, the default `/opt/anaconda3` Python was
  tried. Its dependency installation upgraded packages and emitted resolver
  conflicts in that base environment. The actual tests and this report use
  `aaenv`.
- The editable install refreshed tracked `folps.egg-info` metadata; those
  install-generated diffs were restored to `HEAD`.

Existing tests discovered:

- Script-style tests only: `folps/test_folps_numpy.py`,
  `folps/test_folps_jax.py`, and `folps/test_compare_folps_numpy_vs_jax.py`.
- No `def test_`, `pytest`, or `unittest` tests were found outside notebook
  checkpoints.

Test commands and results:

```bash
/opt/anaconda3/envs/aaenv/bin/python test_folps_numpy.py
```

- Run from `<repo-root>/folps`.
- Result: PASS, exit code 0.
- Output: loaded `output_matrices/matrices_nfftlog128_Afull-True_use_TNS-False.npy`;
  wrote finite NumPy result files.
- Timings: matrix build 0.003 s, loop table 0.050 s, power spectrum 0.006 s,
  bispectrum 0.011 s.
- Warnings: Matplotlib and fontconfig cache directories under the user home were
  not writable, so Matplotlib created a temporary cache directory.

```bash
/opt/anaconda3/envs/aaenv/bin/python test_folps_jax.py
```

- Run from `<repo-root>/folps`.
- Result: PASS, exit code 0.
- Output: JAX used CPU only; loaded the same A_full matrix file; wrote finite
  JAX result files.
- Timings: Pnow precompute 2.410 s, matrix build 0.003 s, loop table setup
  2.206 s, PK first JIT 2.063 s, PK cached 0.006 s, BK first JIT 0.404 s,
  BK cached 0.001 s.
- Warnings: the same Matplotlib/fontconfig cache warnings as the NumPy run.

```bash
/opt/anaconda3/envs/aaenv/bin/python test_compare_folps_numpy_vs_jax.py --skip-run
```

- Run from `<repo-root>/folps`.
- Result: PASS, exit code 0.
- Default tolerances: `atol=5.00e-02`, `rtol=5.00e-03`.
- Max differences:
  - `p0`: abs `1.023e+00`, rel `2.919e-04`, PASS.
  - `p2`: abs `4.823e-01`, rel `1.793e-03`, PASS.
  - `p4`: abs `3.209e-02`, rel `4.537e-04`, PASS.
  - `p0_marg`: abs `1.023e+00`, rel `2.918e-04`, PASS.
  - `p2_marg`: abs `4.823e-01`, rel `1.793e-03`, PASS.
  - `p4_marg`: abs `3.209e-02`, rel `4.538e-04`, PASS.
  - Bispectrum quantities `b000`, `b110`, `b220`, `b202`, `b022`, `b112`: all PASS.

The test runs rewrote tracked output artifacts and `.pyc` files; those generated
changes were restored to `HEAD`.

## 2. Current power-spectrum execution path

Public package exports:

- `folps/__init__.py:4-6` exports `folps.folps`, `folps.cosmo_class`, and
  `folps.tools`.
- `folps/__init__.py:8-14` exposes `tools_jax` only when `FOLPS_BACKEND=jax`.

Backend selection:

- `BackendManager` in `folps/folps.py:29-144` selects NumPy or JAX at import
  time from `FOLPS_BACKEND`.
- `folps/folps.py:147-178` binds global `np`, `interp`, `simpson`, `legendre`,
  `extrapolate_pklin`, and related helpers to the selected backend.
- The power-spectrum formulas are shared by NumPy and JAX through these globals,
  not duplicated into separate physics source files.

Loop-table and matrix setup before multipoles:

- `MatrixCalculator.__init__`, `folps/folps.py:521-543`.
  - Inputs: `nfftlog`, `A_full`, `use_TNS_model`, optional `save_dir`.
  - Outputs: object state, global `A_full_status` and `use_TNS_model_status`,
    optional matrix filename.
  - Shared: NumPy-only matrix construction; the resulting arrays are consumed by
    both backends.
  - Role: control and matrix-file naming.
- `MatrixCalculator.M22`, `folps/folps.py:549-666`.
  - Inputs: FFTLog exponents `nu1`, `nu2`.
  - Outputs: M22 kernels for matter loops, A, D, optional full-A biased channels.
  - Role: physics kernels.
- `MatrixCalculator.M22bias`, `folps/folps.py:670-728`.
  - Inputs: FFTLog exponents `nu1`, `nu2`.
  - Outputs: bias kernels plus optional full-A biased channels.
  - Role: physics kernels.
- `MatrixCalculator.M13`, `folps/folps.py:732-765`.
  - Inputs: FFTLog exponent `nu1`.
  - Outputs: M13 matter and A kernels.
  - Role: physics kernels.
- `MatrixCalculator.M13bias`, `folps/folps.py:769-774`.
  - Inputs: FFTLog exponent `nu1`.
  - Outputs: `Msigma23`.
  - Role: physics kernel for `sigma23`.
- `MatrixCalculator.M22type`, `folps/folps.py:777-791`.
  - Inputs: FFTLog bias `b_nu`, matrix-constructor function.
  - Outputs: matrix stack evaluated on `nuT_y, nuT_x`.
  - Role: matrix assembly; no source spectra are contracted here.
- `MatrixCalculator.M13type`, `folps/folps.py:793-804`.
  - Inputs: FFTLog bias `b_nu`, vector-constructor function.
  - Outputs: vector stack.
  - Role: matrix/vector assembly.
- `MatrixCalculator.calculate_matrices`, `folps/folps.py:806-822`.
  - Inputs: calculator state.
  - Outputs: dict with `M22matrices` and `M13vectors`.
  - Role: concatenates common and bias matrices, optionally saves matrix file.
- `MatrixCalculator.get_mmatrices`, `folps/folps.py:824-836`.
  - Inputs: optional filename state.
  - Outputs: loaded or calculated matrix dict.
  - Role: control logic and matrix-file loading.

Power-spectrum table construction:

- `NonLinearPowerSpectrumCalculator.__init__`, `folps/folps.py:856-887`.
  - Inputs: `mmatrices`, `kernels`, `rbao`, cosmology kwargs.
  - Outputs: object state, output grid `kTout`, matrix stacks.
  - Shared: yes; uses backend `np` for arrays.
  - Role: control and table configuration.
- `_get_f0`, `folps/folps.py:890-931`.
  - Inputs: optional `cosmo`, optional `k`, kwargs.
  - Outputs: scalar `f0`.
  - Shared: yes; selects JAX-optimized EH helper when available.
  - Role: control/cosmology growth.
- `_initialize_factors`, `folps/folps.py:933-970`.
  - Inputs: optional `cosmo`, `k`.
  - Outputs: `inputfkT`, `f0`, `Fkoverf0`.
  - Shared: yes.
  - Role: growth-factor interpolation/control.
- `_initialize_nonwiggle_power_spectrum`, `folps/folps.py:973-991`.
  - Inputs: extrapolated linear spectrum, optional `pknow`, optional `cosmo`.
  - Outputs: `inputpkT_NW`.
  - Shared: yes, calls backend non-wiggle helper.
  - Role: no-wiggle preparation for IR.
- `_initialize_liner_power_spectra`, `folps/folps.py:994-1009`.
  - Inputs: `inputpkT`.
  - Outputs: density, density-velocity, and velocity linear spectra and no-wiggle
    versions.
  - Shared: yes.
  - Role: linear-spectrum bookkeeping.
- `get_cm`, `folps/folps.py:454-490`.
  - Inputs: FFTLog range, `N`, bias `b_nu`, input spectrum.
  - Outputs: FFTLog coefficient vector `c_m`.
  - Shared: yes through backend `np`.
  - Role: FFTLog coefficient construction.
- `_initialize_fftlog_terms`, `folps/folps.py:1012-1050`.
  - Inputs: object spectra and FFTLog settings.
  - Outputs: `cmT`, `cmTf`, `cmTff`, biased-coefficient variants, powers of `K`.
  - Shared: yes.
  - Role: coefficient preparation.
- `P22type`, `folps/folps.py:1053-1150`.
  - Inputs: linear spectra/coefficient vectors and M22 matrices.
  - Outputs: P22 tuple of loop, bias, A, D, and optional A_full biased arrays.
  - Shared: yes after matrix arrays exist.
  - Role: physics contractions.
- `P13type`, `folps/folps.py:1153-1187`.
  - Inputs: linear spectra/coefficient vectors and M13 vectors.
  - Outputs: P13-over-linear tuple and A-term pieces.
  - Shared: yes.
  - Role: physics contractions and zero-lag sigma terms.
- `calculate_P22`, `folps/folps.py:1190-1193`, and `calculate_P13`,
  `folps/folps.py:1195-1198`.
  - Inputs: initialized object state.
  - Outputs: wiggle and no-wiggle P22/P13 pieces.
  - Shared: yes.
  - Role: control wrappers.
- `calculate_loop_table`, `folps/folps.py:1201-1311`.
  - Inputs: `k`, `pklin`, optional `pknow`, optional `cosmo`, cosmology kwargs.
  - Outputs: `(table, table_now)`.
  - Shared: yes.
  - Role: constructs `table` and `table_now`, including IR scalars.

Public multipole path:

- `RSDMultipolesPowerSpectrumCalculator.__init__`, `folps/folps.py:1349-1359`.
  - Inputs: model name (`EFT`, `TNS`, `FOLPSD`, or other).
  - Outputs: calculator state.
  - Shared: yes.
  - Role: control.
- `set_bias_scheme`, `folps/folps.py:1361-1400`.
  - Inputs: positional `pars`, `bias_scheme`.
  - Outputs: canonical Folps parameter ordering.
  - Shared: yes.
  - Role: parameter translation only.
- `interp_table`, `folps/folps.py:1403-1442`.
  - Inputs: query `k`, `table`, global `A_full_status`.
  - Outputs: interpolated table tuple without the original k row.
  - Shared: yes; JAX uses backend `interp`, NumPy uses SciPy `CubicSpline`.
  - Role: interpolation only.
- `k_ap`, `folps/folps.py:1444-1447`, and `mu_ap`,
  `folps/folps.py:1449-1452`.
  - Inputs: observed `k`, observed `mu`, `qpar`, `qper`.
  - Outputs: AP-remapped true `k` and `mu`.
  - Shared: yes.
  - Role: AP remapping.
- `get_eft_pkmu`, `folps/folps.py:1454-1618`.
  - Inputs: true `kev`, true `mu`, canonical `pars`, interpolated table,
    `damping`.
  - Outputs: one-loop redshift-space `P(k,mu)` contribution before external IR
    mixing.
  - Shared: yes.
  - Role: main physics contractions, EFT/stochastic terms, FolpsD damping.
- `get_rsd_pkmu`, `folps/folps.py:1620-1637`.
  - Inputs: true `k`, true `mu`, `pars`, `table`, `table_now`, IR flag,
    `damping`.
  - Outputs: IR-resummed `P(k,mu)`.
  - Shared: yes.
  - Role: table interpolation, IR resummation, and pkmu assembly.
- `get_rsd_pkell`, `folps/folps.py:1639-1674`.
  - Inputs: observed `kobs`, AP parameters, `pars`, `table`, `table_now`,
    `bias_scheme`, `damping`, `nmu`, `ells`, IR flag.
  - Outputs: multipole array with leading ell dimension.
  - Shared: yes; Gauss-Legendre nodes come from host NumPy, then backend arrays
    are used in the formula.
  - Role: public multipole interface, AP remapping, quadrature, multipole
    integration.

Additional public power-spectrum helpers:

- `get_rsd_pkell_marg_const`, `folps/folps.py:1680-1773`, duplicates the
  pkmu and pkell control path with EFT/stochastic parameters set to zero.
- `PEFTs_derivatives`, `folps/folps.py:1777-1800`, returns derivatives with
  respect to `alpha0`, `alpha2`, `alpha4`, `alphashot0`, `alphashot2`.
- `get_rsd_pkell_marg_derivatives`, `folps/folps.py:1804-1870`, duplicates the
  AP, interpolation, IR, and multipole-integration path for analytical
  marginalization derivatives.

Code locations for requested physical pieces:

- AP remapping: `k_ap` and `mu_ap`, `folps/folps.py:1444-1452`; applied in
  `get_rsd_pkell`, `folps/folps.py:1671-1673`.
- Table interpolation: `interp_table`, `folps/folps.py:1403-1442`; used in
  `get_rsd_pkmu`, `folps/folps.py:1622-1623`.
- `P(k,mu)` construction: `get_eft_pkmu`, `folps/folps.py:1454-1618`; IR
  wrapper in `get_rsd_pkmu`, `folps/folps.py:1620-1637`.
- IR resummation: loop-table sigmas in `calculate_loop_table`,
  `folps/folps.py:1220-1226`; pkmu IR mixing in `get_rsd_pkmu`,
  `folps/folps.py:1629-1636`.
- A term: matrix kernels `folps/folps.py:564-581`, M13 A kernels
  `folps/folps.py:744-765`, contractions `folps/folps.py:1097-1103` and
  `folps/folps.py:1182-1185`, combination `folps/folps.py:1265-1269`,
  use in `get_eft_pkmu`, `folps/folps.py:1488-1507`.
- D term: kernels `folps/folps.py:599-647`, contractions
  `folps/folps.py:1104-1122`, table mapping `folps/folps.py:1271-1278`,
  use in `get_eft_pkmu`, `folps/folps.py:1498-1510`.
- G term: `GTNS`, `folps/folps.py:1512-1516`; uses `sigma2w`, `Pdt_L`, and
  `Ptt_L`.
- EFT counterterms: `PctNLOs` and `Pcts`, `folps/folps.py:1535-1539`, added at
  `folps/folps.py:1618`.
- Stochastic terms: `Pshot`, `folps/folps.py:1541-1542`, added at
  `folps/folps.py:1613`.
- FolpsD damping: `Winfty`, `Wexp`, `Wlorentz`, model dispatch
  `folps/folps.py:1544-1611`; damping multiplies `PloopSPTs` at
  `folps/folps.py:1613`.
- Multipole integration: `get_rsd_pkell`, `folps/folps.py:1661-1674`.

## 3. Parameter ordering

Canonical power-spectrum parameter order after `set_bias_scheme`:

```text
(b1, b2, bs2, b3nl,
 alpha0, alpha2, alpha4, ctilde,
 alphashot0, alphashot2, PshotP, X_FoG_p)
```

Power-spectrum bias schemes:

- `folps`, `pat`, `mcdonald`, `folps/folps.py:1366-1370`.
  - Input order is already canonical.
- `assassi`, `classpt`, `folps/folps.py:1371-1382`.
  - Input:
    `(b1_classPT, b2_classPT, bG2_classPT, bGamma3_classPT,
    alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, PshotP, X_FoG_p)`.
  - Transform:
    `b1 = b1_classPT`;
    `b2 = b2_classPT - 4/3 * bG2_classPT`;
    `bs2 = 2 * bG2_classPT`;
    `b3nl = -32/21 * (bG2_classPT + 2/5 * bGamma3_classPT)`.
- `desi`, `priordocument`, `dr2`, `priordoc`, `folps/folps.py:1384-1395`.
  - Input:
    `(b1_priordoc, b2_priordoc, bK2_priordoc, btd_priordoc,
    alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, PshotP, X_FoG_p)`.
  - Transform:
    `b1 = b1_priordoc`;
    `b2 = b2_priordoc`;
    `bs2 = 2. * bK2_priordoc`;
    `b3nl = -32/21 * (bK2_priordoc + 2/5 * btd_priordoc)`.

Model and damping options do not change parameter ordering:

- `model="EFT"` ignores damping and sets `W=1`, `folps/folps.py:1573-1574`.
- `model="TNS"` requires `use_TNS_model=True`, `folps/folps.py:1575-1578`;
  damping can be `None`, `exp`, `lor`, or `vdg`, `folps/folps.py:1579-1588`.
- `model="FOLPSD"` uses `exp`, `lor`, or `vdg`; `None` is changed to `lor`,
  `folps/folps.py:1589-1600`.
- `X_FoG_p` is read by the damping windows, `folps/folps.py:1544-1561`.

Positional unpacking locations:

- Canonicalization: `set_bias_scheme`, `folps/folps.py:1361-1400`.
- Direct canonical unpack in `get_eft_pkmu`, `folps/folps.py:1456`.
- Direct `b1 = pars[0]` in `get_rsd_pkmu`, `folps/folps.py:1624`.
- Marginalized constant helper unpack, `folps/folps.py:1693-1695`.
- Marginalized derivative helper unpack, `folps/folps.py:1816-1818`.

There is not one central unpacking function. `set_bias_scheme` is the public
canonicalization point for `get_rsd_pkell`, but direct calls to `get_rsd_pkmu`
or `get_eft_pkmu` assume canonical ordering already. The marginalization
helpers also repeat their own unpacking.

## 4. Loop-table inventory

`calculate_loop_table` returns `(table, table_now)`, `folps/folps.py:1308-1311`.
The table schema is built in `combine_loop_terms`, `folps/folps.py:1241-1306`.

Original `table` rows before interpolation:

| Row A_full=False | Row A_full=True | Code variable | Physical quantity | Category | Present when A_full=False | Present when A_full=True | Index changes with flag |
|---:|---:|---|---|---|---|---|---|
| 0 | 0 | `self.kTout` | output k grid | other | yes | yes | no |
| 1 | 1 | `pk_l` | linear wiggle spectrum | linear | yes | yes | no |
| 2 | 2 | `self.Fkoverf0` | `f(k)/f0` | linear | yes | yes | no |
| 3 | 3 | `Ploop_dd` | one-loop matter density-density | loop | yes | yes | no |
| 4 | 4 | `Ploop_dt` | one-loop matter density-velocity | loop | yes | yes | no |
| 5 | 5 | `Ploop_tt` | one-loop velocity-velocity | loop | yes | yes | no |
| 6 | 6 | `Pb1b2` | `b1 b2` density bias loop | bias | yes | yes | no |
| 7 | 7 | `Pb1bs2` | `b1 bs2` density bias loop | bias | yes | yes | no |
| 8 | 8 | `Pb22` | `b2^2` density bias loop, zero-lag removed | bias | yes | yes | no |
| 9 | 9 | `Pb2bs2` | `b2 bs2` density bias loop, zero-lag removed | bias | yes | yes | no |
| 10 | 10 | `Pb2s2` | `bs2^2` density bias loop, zero-lag removed | bias | yes | yes | no |
| 11 | 11 | `sigma23pkl` | `b3nl`/`sigma23` contribution times linear spectrum | bias | yes | yes | no |
| 12 | 12 | `Pb2t` | `b2 theta` loop | bias | yes | yes | no |
| 13 | 13 | `Pbs2t` | `bs2 theta` loop | bias | yes | yes | no |
| 14 | 14 | `I1udd_1` | combined A `I1udd_1` | A | yes | yes | no |
| 15 | 15 | `I2uud_1` | combined A `I2uud_1` | A | yes | yes | no |
| 16 | 16 | `I2uud_2` | combined A `I2uud_2` | A | yes | yes | no |
| 17 | 17 | `I3uuu_2` | combined A `I3uuu_2` | A | yes | yes | no |
| 18 | 18 | `I3uuu_3` | combined A `I3uuu_3` | A | yes | yes | no |
| 19 | 19 | `I2uudd_1D` | D `I2uudd_1D` | D | yes | yes | no |
| 20 | 20 | `I2uudd_2D` | D `I2uudd_2D` | D | yes | yes | no |
| 21 | 21 | `I3uuud_2D` | D `I3uuud_2D` | D | yes | yes | no |
| 22 | 22 | `I3uuud_3D` | D `I3uuud_3D` | D | yes | yes | no |
| 23 | 23 | `I4uuuu_2D` | D `I4uuuu_2D` | D | yes | yes | no |
| 24 | 24 | `I4uuuu_3D` | D `I4uuuu_3D` | D | yes | yes | no |
| 25 | 25 | `I4uuuu_4D` | D `I4uuuu_4D` | D | yes | yes | no |
| 26 | 26 | `I3uuud_1_B` | TNS/D auxiliary row, zero unless `use_TNS_model=True` | D | yes | yes | no |
| 27 | 27 | `I4uuuu_1_B` | TNS/D auxiliary row, zero unless `use_TNS_model=True` | D | yes | yes | no |
| - | 28 | `I1udd_1_b2` | full-A `b2` channel 1 | A | no | yes | yes |
| - | 29 | `I2uud_1_b2` | full-A `b2` channel 2 | A | no | yes | yes |
| - | 30 | `I2uud_2_b2` | full-A `b2` channel 3 | A | no | yes | yes |
| - | 31 | `I1udd_1_bs2` | full-A `bs2` channel 1 | A | no | yes | yes |
| - | 32 | `I2uud_1_bs2` | full-A `bs2` channel 2 | A | no | yes | yes |
| - | 33 | `I2uud_2_bs2` | full-A `bs2` channel 3 | A | no | yes | yes |
| 28 | 34 | `sigma2w` | velocity-dispersion scalar used by G and damping | G/other | yes | yes | yes |
| 29 | 35 | `self.f0` | large-scale growth rate | linear/control | yes | yes | yes |

`table_now` has the same physical rows, built from no-wiggle inputs, but when
`extra_NW=True` it appends two IR scalars before `f0`:

| Row A_full=False | Row A_full=True | Code variable | Physical quantity | Category | Present when A_full=False | Present when A_full=True | Index changes with flag |
|---:|---:|---|---|---|---|---|---|
| 0-27 | 0-27 | same as `table` rows 0-27 | no-wiggle versions of common rows | see above | yes | yes | no |
| - | 28-33 | same as `table` rows 28-33 | no-wiggle full-A biased rows | A | no | yes | yes |
| 28 | 34 | `sigma2w_NW` | no-wiggle velocity-dispersion scalar | G/other | yes | yes | yes |
| 29 | 35 | `self.sigma2_NW` | IR damping sigma | IR | yes | yes | yes |
| 30 | 36 | `self.delta_sigma2_NW` | IR angular correction sigma | IR | yes | yes | yes |
| 31 | 37 | `self.f0` | large-scale growth rate | linear/control | yes | yes | yes |

After `interp_table`, the k row is removed. For A_full=False, interpolated rows
0-26 correspond to original rows 1-27; for A_full=True, interpolated rows 0-32
correspond to original rows 1-33. The scalar tail starts at interpolated row 27
for A_full=False and row 33 for A_full=True.

The six additional A_full=True arrays are exactly:

- `I1udd_1_b2`, `I2uud_1_b2`, `I2uud_2_b2`.
- `I1udd_1_bs2`, `I2uud_1_bs2`, `I2uud_2_bs2`.

Hard-coded row-number assumptions:

- M22 matrix unpacking changes with `A_full_status`, `folps/folps.py:1058-1074`.
- P22 output row use is hard-coded in `combine_loop_terms`,
  `folps/folps.py:1242-1297`.
- `interp_table` hard-codes `extra = 6 if A_full_status else 0`,
  `cols_to_interp = table[1:28 + extra]`, and `table[28 + extra:]`,
  `folps/folps.py:1411-1427` and `folps/folps.py:1436-1442`.
- `get_eft_pkmu` has separate A_full and non-A_full table unpacking,
  `folps/folps.py:1460-1472`.
- `get_rsd_pkmu` assumes interpolated `table[1]` is `Fkoverf0`,
  `table[0]` is `pkl`, and `table_now[-3:-1]` are the IR sigmas,
  `folps/folps.py:1624-1628`.

## 5. Exact current auto-spectrum polynomial

This section rewrites the implemented auto-spectrum with code variable names
from `get_eft_pkmu`, `folps/folps.py:1454-1618`. Define:

```text
fk = Fkoverf0 * f0
Pdt_L = pkl * Fkoverf0
Ptt_L = pkl * Fkoverf0**2
```

The density-density loop polynomial is:

```text
PddXloop =
    b1**2 * Ploop_dd
  + 2 * b1 * b2 * Pb1b2
  + 2 * b1 * bs2 * Pb1bs2
  + b2**2 * Pb22
  + 2 * b2 * bs2 * Pb2bs2
  + bs2**2 * Pb2s2
  + 2 * b1 * b3nl * sigma23pkl
```

The density-velocity piece entering redshift space is:

```text
2 * f0 * mu**2 * PdtXloop

PdtXloop =
    b1 * Ploop_dt
  + b2 * Pb2t
  + bs2 * Pbs2t
  + b3nl * Fkoverf0 * sigma23pkl
```

The velocity-velocity piece is:

```text
f0**2 * mu**4 * Ploop_tt
```

The A term with `A_full=False` is:

```text
ATNS =
    b1**2 * f0 * mu**2 * I1udd_1
  + b1 * f0**2 * (mu**2 * I2uud_1 + mu**4 * I2uud_2)
  + f0**3 * (mu**4 * I3uuu_2 + mu**6 * I3uuu_3)
```

The extra A terms with `A_full=True` are:

```text
ATNS_b2_bs2 =
    (b1 * b2 * f0 / 2) * mu**2 * I1udd_1_b2
  + (b2 * f0**2 / 2) * (mu**2 * I2uud_1_b2 + mu**4 * I2uud_2_b2)
  + (b1 * bs2 * f0 / 2) * mu**2 * I1udd_1_bs2
  + (bs2 * f0**2 / 2) * (mu**2 * I2uud_1_bs2 + mu**4 * I2uud_2_bs2)
```

The D term is:

```text
DRSD =
    b1**2 * f0**2 * (mu**2 * I2uudd_1D + mu**4 * I2uudd_2D)
  + b1 * f0**3 * (mu**2 * I3uuud_1B + mu**4 * I3uuud_2D + mu**6 * I3uuud_3D)
  + f0**4 * (
        mu**2 * I4uuuu_1B
      + mu**4 * I4uuuu_2D
      + mu**6 * I4uuuu_3D
      + mu**8 * I4uuuu_4D
    )
```

The G term is:

```text
GTNS =
  - (kev * mu * f0)**2 * sigma2w
    * (b1**2 * pkl + 2 * b1 * f0 * mu**2 * Pdt_L + f0**2 * mu**4 * Ptt_L)
```

when `use_TNS_model_status` is false. When `use_TNS_model_status` is true,
`GTNS` returns zero.

The full SPT loop part is:

```text
PloopSPTs =
    PddXloop
  + 2 * f0 * mu**2 * PdtXloop
  + f0**2 * mu**4 * Ploop_tt
  + ATNS
  + DRSD
  + GTNS
  + ATNS_b2_bs2   # only if A_full_status is true
```

EFT and stochastic terms are:

```text
PKaiserLs = (b1 + mu**2 * fk)**2 * pkl

PctNLOs = ctilde * (mu * kev * f0)**4 * sigma2w**2 * PKaiserLs

Pcts = (alpha0 + alpha2 * mu**2 + alpha4 * mu**4) * kev**2 * pkl

Pshot = PshotP * (alphashot0 + alphashot2 * (kev * mu)**2)
```

FolpsD/TNS damping windows are:

```text
Winfty:
  c2 = (f0 * kev * mu)**2
  X2 = X_FoG_p**2
  exp = - c2 * sigma2w / (1 + c2 * X2)
  W = exp(exp) / sqrt(1 + c2 * X2)

Wexp:
  l2 = (f0 * kev * mu * X_FoG_p)**2
  W = exp(-l2 * sigma2w)

Wlorentz:
  l2 = (f0 * kev * mu * X_FoG_p)**2
  W = 1 / (1 + l2 * sigma2w)
```

`get_eft_pkmu` returns:

```text
PEFT = W * PloopSPTs + Pshot + Pcts + PctNLOs
```

because `Winfty_all` is hard-coded false and resets `W = 1.0` before adding
`Pcts + PctNLOs`. Thus damping multiplies `PloopSPTs`, but not stochastic or
EFT counterterms in the current implementation.

The IR-resummed wiggle/no-wiggle assembly in `get_rsd_pkmu` is:

```text
sigma2t =
  (1 + f0 * mu**2 * (2 + f0)) * sigma2
  + (f0 * mu)**2 * (mu**2 - 1) * delta_sigma2
```

if `IR_resummation=True`, otherwise `sigma2t = 0`. With
`E = exp(-k**2 * sigma2t)`, the returned value is:

```text
pkmu =
    (b1 + fk * mu**2)**2
    * (pkl_now + E * (pkl - pkl_now) * (1 + k**2 * sigma2t))
  + E * get_eft_pkmu(k, mu, pars, table, damping)
  + (1 - E) * get_eft_pkmu(k, mu, pars, table_now, damping)
```

This is a transcription of the code, not a replacement by a textbook formula.

## 6. Matrix sufficiency for A_full=True

Relevant matrix construction:

- M22 A kernels: `MtAfp_11`, `MtAfkmpfp_12`, `MtAfkmpfp_22`,
  `MtAfpfp_22`, `MtAfkmpfpfp_23`, `MtAfkmpfpfp_33`,
  `folps/folps.py:564-581`.
- D kernels and optional TNS substitutions: `folps/folps.py:599-647`.
- A_full M22 kernels proportional to `b2` and `bs2`:
  `MtAfkmpfp_22_b2`, `MtAfkmpfp_22_bs2`, `folps/folps.py:648-664`.
- A_full M22-bias kernels proportional to `b2` and `bs2`:
  `MtAfp_11_b2`, `MtAfp_11_bs2`, `MtAfkmpfp_12_b2`,
  `MtAfkmpfp_12_bs2`, `folps/folps.py:705-726`.
- M13 A kernels: `Mafk_11`, `Mafp_11`, `Mafkfp_12`, `Mafpfp_12`,
  `Mafkfkfp_33`, `Mafkfpfp_33`, `folps/folps.py:744-765`.

Matrix assembly and contractions:

- `M22type` builds `nuT_x, nuT_y = np.meshgrid(nuT, nuT)` and evaluates
  `M22(nuT_y, nuT_x)`, `folps/folps.py:777-791`.
- `calculate_matrices` concatenates `M22T` with `M22biasT` and `M13T` with
  `M13biasT`, `folps/folps.py:806-814`.
- No explicit transpose or symmetrization appears in the M22/M13 matrix
  construction path. A search found only the meshgrid and concatenations in the
  matrix path; transpose operations occur in unrelated likelihood/bispectrum
  utilities.
- Matter and A contractions are of the form `np.sum(left @ M * right, axis=-1)`,
  for example `folps/folps.py:1084-1102`.
- Full-A biased contractions are:
  - `I1udd_1b_b2 = vecf_b @ MtAfp_11_b2 * vec_b`,
    `folps/folps.py:1134`.
  - `I2uud_1b_b2 = vecf_b @ MtAfkmpfp_12_b2 * vecf_b`,
    `folps/folps.py:1135`.
  - `I2uud_2b_b2 = vecf @ MtAfkmpfp_22_b2 * vecf`,
    `folps/folps.py:1136`.
  - `I1udd_1b_bs2 = vecf_b @ MtAfp_11_bs2 * vec_b`,
    `folps/folps.py:1138`.
  - `I2uud_1b_bs2 = vecf_b @ MtAfkmpfp_12_bs2 * vecf_b`,
    `folps/folps.py:1139`.
  - `I2uud_2b_bs2 = vecf @ MtAfkmpfp_22_bs2 * vecf`,
    `folps/folps.py:1140`.

Findings:

- If endpoint exchange only transposes the FFTLog kernel and the same coefficient
  vector appears on both sides, then the scalar contraction is invariant:
  `c^T M c = c^T M^T c`.
- This invariant applies directly to full-A contractions with identical vectors
  on both sides, such as `vecf_b`/`vecf_b` or `vecf`/`vecf`.
- It does not follow from the code for contractions with different left and right
  vectors, notably `vecf_b @ MtAfp_11_b2 * vec_b` and
  `vecf_b @ MtAfp_11_bs2 * vec_b`, unless either the EdS limit makes
  `cmTf_b == cmT_b` or an external analytic derivation shows that the endpoint
  exchange also swaps the coefficient vectors in a way already represented by
  the stored scalar.
- The current code stores six full-A biased arrays, but it does not label them as
  ordered endpoint contributions, endpoint sums, or one endpoint with an implicit
  auto-spectrum factor. The auto code then multiplies them by `b2/(2*b1)` and
  `bs2/(2*b1)`, `folps/folps.py:1506-1507`.

Conclusion: C. The code inspection is inconclusive.

The exact missing information is an endpoint-level derivation, tied to the
definitions of the six full-A biased kernels, that proves one of the following:

- `O@A` and `O@B` are equal for `O in {b2, bs2}` with `fk` kernels and
  scale-dependent growth, including the `vecf_b`/`vec_b` contractions; or
- the stored arrays are already endpoint sums with a known normalization; or
- the stored arrays are one endpoint ordering and the missing ordered partner is
  recoverable by transposition and coefficient-vector swapping.

Until that derivation exists, Stage 2 should not treat the existing A_full=True
arrays as sufficient for independent `b2_a`, `b2_b`, `bs2_a`, and `bs2_b`
coefficients. The ordered channels at risk are:

- `I1udd_1_b2^{O@A}` and `I1udd_1_b2^{O@B}`.
- `I2uud_1_b2^{O@A}` and `I2uud_1_b2^{O@B}`.
- `I2uud_2_b2^{O@A}` and `I2uud_2_b2^{O@B}`.
- The analogous three `bs2` channel pairs.

## 7. Recommended API

Design A: add separate explicit methods `get_rsd_pkmu_cross(...)` and
`get_rsd_pkell_cross(...)`.

- Strength: clear user intent and no ambiguity about one-tracer vs two-tracer
  parameter arrays.
- Weakness: likely duplicates AP remapping, interpolation, IR resummation, and
  multipole integration unless those internals are factored first.
- Weakness: auto-limit tests must compare two independent physics formulas if
  the auto path remains separate.

Design B: generalize the existing methods to accept two tracer parameter sets.

- Strength: preserves backward compatibility if `pars_b=None` means auto mode.
- Strength: lets current auto-spectrum call the same generalized pair
  contraction with `A=B`, making auto-limit testing exact and ongoing.
- Strength: less duplication in AP, IR, interpolation, damping, and multipole
  integration.
- Strength: better for desilike integration, because the public multipole method
  remains the stable entry point.
- Risk: a long flat positional array would be unsafe for JAX and users.

Recommendation: Design B, but with a small internal pair-parameter structure and
optional explicit cross wrappers later. Keep the existing signature valid:

```text
get_rsd_pkmu(k, mu, pars, table, table_now, ..., pars_b=None, cross_nuisance=None)
get_rsd_pkell(kobs, qpar, qper, pars, table, table_now, ..., pars_b=None, cross_nuisance=None)
```

Parameter organization:

```text
bias_a = (b1_a, b2_a, bs2_a, b3nl_a)
bias_b = (b1_b, b2_b, bs2_b, b3nl_b)

cross_nuisance = (
    alpha0_ab, alpha2_ab, alpha4_ab, ctilde_ab,
    alphashot0_ab, alphashot2_ab, PshotP_ab, X_FoG_ab
)
```

For `pars_b=None`, set `bias_b = bias_a` and use the existing auto nuisance
parameters. For cross spectra, canonicalize each tracer through the existing
bias-scheme transformations before entering the JAX-traceable pair contraction.

## 8. Minimal Stage 2 implementation plan

Smallest source files/functions likely needing changes:

- `folps/folps.py`
  - Add a canonical power-parameter helper near `set_bias_scheme` so
    canonicalization and unpacking are not repeated.
  - Add an internal pair contraction, for example `_get_eft_pkmu_pair`, that
    takes `bias_a`, `bias_b`, and `cross_nuisance`.
  - Update `get_eft_pkmu` so current auto mode calls the pair contraction with
    `bias_a == bias_b`.
  - Update `get_rsd_pkmu` and `get_rsd_pkell` to accept optional `pars_b` and
    `cross_nuisance` while preserving existing calls.
  - Update marginalized helpers only after the non-marginalized cross path is
    validated.
- `folps/test_cross_power_spectrum.py`
  - Add script-style tests following the current repository pattern.
- Optional later documentation:
  - `README.md` or docs examples only after the API is stable.

Implementation invariants:

- Build each cross coefficient symmetrically so `P_AB = P_BA` by construction.
- Use the same generalized pair contraction for the auto path so
  `P_AB(A=B)` reproduces the current auto-spectrum.
- In a synthetic linear-only table with all loop rows set to zero and IR off,
  return `(b1_a + f mu**2) * (b1_b + f mu**2) * P_L`.
- Implement `A_full=False` first.
- Keep `A_full=True` blocked or explicitly unsupported for cross mode until the
  endpoint proof in section 6 is resolved.

The current auto method can eventually become:

```text
canonical = canonicalize_power_pars(pars, bias_scheme)
return _get_rsd_pkmu_pair(..., bias_a=canonical.bias, bias_b=canonical.bias,
                          cross_nuisance=canonical.nuisance)
```

That prevents maintaining two independent physics formulas.

## 9. Test plan

Existing framework: script-style tests in `folps/test_*.py`, executable with the
chosen Python interpreter. New tests should follow that pattern unless the
project adopts pytest later.

Proposed test file: `folps/test_cross_power_spectrum.py`.

Use the same baseline cosmology and input spectrum as the existing tests:

```text
k, pk = np.loadtxt("inputpkT.txt", unpack=True)
z = 0.3
h = 0.6711
Omega_m = 0.3211636237981114
f0 = 0.6880638641959066
fnu = 0.004453689063655854
```

Representative tracer parameters:

```text
LRG-like: b1=1.645, b2=-0.46,
          bs2=-4/7*(b1-1), b3nl=32/315*(b1-1)
ELG-like: b1=1.1, b2=0.2,
          bs2=-4/7*(b1-1), b3nl=32/315*(b1-1)
```

Tests:

- Auto-limit recovery with `A_full=False`.
  - Matrix: `MatrixCalculator(A_full=False, use_TNS_model=False)`.
  - Expected invariant: cross with `A=B` equals current auto multipoles.
  - Tolerance: NumPy `rtol=1e-10`, `atol=1e-6`, because this should be the same
    algebra and interpolation path.
- Auto-limit recovery with `A_full=True`.
  - Matrix: `MatrixCalculator(A_full=True, use_TNS_model=False)`.
  - Expected invariant: cross with `A=B` equals current auto multipoles.
  - Tolerance: NumPy `rtol=1e-10`, `atol=1e-6`.
  - Status: should be xfailed or blocked until section 6 is resolved.
- `A <-> B` exchange symmetry.
  - Parameter point: LRG-like A, ELG-like B, `A_full=False`.
  - Expected invariant: `P_AB == P_BA` for `P0`, `P2`, `P4`.
  - Tolerance: `rtol=1e-11`, `atol=1e-7`.
- Linear Kaiser limit.
  - Parameter point: synthetic table with all loop, A, D, G, EFT, stochastic,
    and IR rows zero; `pkl_now=pkl`, `IR_resummation=False`.
  - Expected invariant:
    `(b1_a + f mu**2) * (b1_b + f mu**2) * P_L`.
  - Tolerance: `rtol=1e-12`, `atol=1e-12`, because no numerical integration
    beyond the controlled multipole quadrature is needed.
- Isolated `b2_a` variation.
  - Parameter point: set `b2_a=0.3`, `b2_b=0`, other nonlinear biases zero.
  - Expected invariant: swapping A/B gives the same final spectrum after
    swapping labels; auto limit still holds when `b2_b=b2_a`.
  - Tolerance: `rtol=1e-10`, `atol=1e-6`.
- Isolated `b2_b` variation.
  - Same as above with `b2_a=0`, `b2_b=0.3`.
  - Expected invariant: paired with previous test under exchange.
  - Tolerance: `rtol=1e-10`, `atol=1e-6`.
- Isolated `bs2_a` variation.
  - Parameter point: `bs2_a=-0.2`, `bs2_b=0`, other nonlinear biases zero.
  - Expected invariant: exchange symmetry and auto-limit behavior.
  - Tolerance: `rtol=1e-10`, `atol=1e-6`.
- Isolated `bs2_b` variation.
  - Same as above with `bs2_a=0`, `bs2_b=-0.2`.
  - Expected invariant: paired with previous test under exchange.
  - Tolerance: `rtol=1e-10`, `atol=1e-6`.
- NumPy/JAX agreement.
  - Parameter point: LRG-like A, ELG-like B, `A_full=False`, `model="FOLPSD"`.
  - Expected invariant: NumPy and JAX cross multipoles agree.
  - Tolerance: start with existing backend-comparison defaults
    `rtol=5e-3`, `atol=5e-2`, then tighten after interpolation differences are
    understood.
- IR on/off.
  - Parameter point: LRG-like A, ELG-like B, `qpar=qper=1`.
  - Expected invariant: `IR_resummation=False` equals the direct non-IR pair
    formula; `IR_resummation=True` remains symmetric and auto-limited.
  - Tolerance: auto-limit `rtol=1e-10`, cross symmetry `rtol=1e-11`.
- AP remapping.
  - Parameter point: `qpar=1.02`, `qper=0.98`.
  - Expected invariant: cross `A=B` equals current AP-remapped auto multipoles;
    `P_AB=P_BA`.
  - Tolerance: `rtol=1e-10`, `atol=1e-6`.
- Standard EFT without phenomenological damping.
  - Parameter point: `model="EFT"`, `damping=None`, nonzero `alpha0/2/4`,
    `ctilde=0`, stochastic off.
  - Expected invariant: no FoG window applied, cross auto-limit holds.
  - Tolerance: `rtol=1e-10`, `atol=1e-6`.
- Regression of current auto-spectrum at representative points.
  - Parameter point: existing `test_folps_numpy.py` power-spectrum parameters,
    `A_full=True`, `model="FOLPSD"`, `damping="lor"`.
  - Expected invariant: current auto multipoles unchanged at stored k values.
  - Tolerance: use current NumPy output as reference with `rtol=1e-10`,
    `atol=1e-6` if same backend; use existing compare tolerances for JAX.

## 10. Risks and unresolved issues

- Parameter conventions are fragile: the power-spectrum API uses long positional
  arrays, and canonicalization is not centralized.
- Direct `get_rsd_pkmu` and `get_eft_pkmu` calls can bypass `set_bias_scheme`;
  cross support would amplify this risk.
- Physics is duplicated in marginalization helpers, especially
  `get_rsd_pkell_marg_const` and `get_rsd_pkell_marg_derivatives`.
- `A_full_status` and `use_TNS_model_status` are globals set by the last
  `MatrixCalculator`, so mixed matrix/table states can be hard to reason about.
- JAX tracing risk: model strings, damping strings, `A_full_status`, and optional
  `pars_b=None` branches should remain static outside JIT-compiled kernels.
- Shape risk: `interp_table` removes the k row and relies on numeric slices; the
  scalar tail changes between `table` and `table_now` and shifts with A_full.
- IR resummation risk: `get_rsd_pkmu` uses `table_now[-3:-1]` for IR sigmas;
  any cross-table schema change must preserve that tail contract or replace it
  with a named structure.
- Interpolation risk: NumPy uses SciPy `CubicSpline`, JAX uses backend `interp`;
  current backend tolerances are loose enough to allow small interpolation
  differences.
- A_full=True should block Stage 2 cross production until the ordered-endpoint
  matrix question is resolved.
- Existing tests are scripts that rewrite tracked output files and `.pyc` files;
  a new cross test should avoid changing tracked artifacts or should write to a
  temporary directory.

Final Stage 1 conclusion:

- Repository setup and existing tests are complete in `aaenv`.
- Existing NumPy, JAX, and NumPy/JAX comparison scripts pass.
- The code path for the current auto-spectrum is documented above.
- Existing loop tables are sufficient for the density, density-velocity,
  velocity-velocity, D, G, EFT, stochastic, AP, IR, and multipole infrastructure.
- Matrix sufficiency for full biased A (`A_full=True`) is inconclusive from the
  source alone and should block a production cross implementation until proven.
