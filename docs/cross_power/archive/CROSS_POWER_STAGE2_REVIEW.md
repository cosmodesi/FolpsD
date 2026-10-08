# Cross-Power Stage 2 Review

This is the Stage 3 review of the uncommitted Stage 2 equal-time two-tracer
redshift-space power-spectrum implementation on branch
`feature/cross-power-spectrum`.

No commit or push was performed.

Update: this review records the pre-damping Stage 2 state. The current branch
now adds FolpsD cross damping through `cross_damping_mode="single"` and
`"geometric"`; see `CROSS_POWER_IMPLEMENTATION.md` for the active API.

## Initial Repository Snapshot

Commands requested before editing:

```bash
git branch --show-current
```

```text
feature/cross-power-spectrum
```

```bash
git rev-parse HEAD
```

```text
94ee3bfbbf85e0c679be73e1038ccfd9f8523434
```

```bash
git status --short
```

```text
 M .DS_Store
 M docs/.DS_Store
 M folps/folps.py
?? docs/cross_power/
?? folps/test_cross_power_spectrum.py
```

```bash
git diff --stat
```

```text
 .DS_Store      | Bin 10244 -> 12292 bytes
 docs/.DS_Store | Bin 6148 -> 8196 bytes
 folps/folps.py | 211 +++++++++++++++++++++++++++++++++++++++++----------------
 3 files changed, 154 insertions(+), 57 deletions(-)
```

```bash
git diff --check
```

```text
(no output)
```

I inspected the complete `folps/folps.py` diff, the untracked
`folps/test_cross_power_spectrum.py`, and the requested cross-power documents
before editing.

## File-by-File Diff Review

`folps/folps.py`

- Adds `_PowerSpectrumBias`, `_PowerSpectrumNuisance`, and
  `_PowerSpectrumParameters` `NamedTuple` containers. These are JAX-compatible
  pytrees at the field level because they contain scalars/arrays and no custom
  object mutation.
- Adds `_split_power_pars`, `_split_cross_nuisance`,
  `_resolve_pair_parameters`, `_validate_cross_spectrum_model`, and
  `_get_eft_pkmu_pair`.
- Replaces the old auto-only EFT contraction with one pair-level contraction.
  The auto path calls it with `bias_b = bias_a` and the current auto nuisance
  tuple when `pars_b is None`.
- Extends `get_eft_pkmu`, `get_rsd_pkmu`, and `get_rsd_pkell` for cross mode.
  In the final reviewed version the new public arguments are keyword-only, so
  old positional calls retain exactly their previous meaning.
- Leaves interpolation, AP remapping, Jacobian multiplication, and multipole
  quadrature structure unchanged.
- Does not touch marginalized helper implementations. Those helpers still have
  their own auto-spectrum polynomial copies, but they are outside the current
  non-marginalized cross path.

`folps/test_cross_power_spectrum.py`

- Adds backend-isolated NumPy/JAX tests for cross spectra.
- Adds exact residual printing for successful checks.
- Adds explicit `ValueError` tests for missing cross nuisance in cross mode,
  invalid tracer parameter lengths, invalid `cross_nuisance` lengths, and
  the retained 12-value `cross_nuisance` extraction convention.
- Adds independent synthetic-table checks with one nonzero row at a time for
  density loops, density-velocity terms, velocity-velocity terms, the five
  reduced-A rows, all six full-A rows, D rows, and G.
- Keeps auto-limit, exchange symmetry, AP-remapped multipoles, direct quadrature
  equivalence, current auto-output regression, and NumPy/JAX cross agreement.

`docs/cross_power/CROSS_POWER_IMPLEMENTATION.md`

- Updated the API description so it no longer claims that omitted cross
  nuisance defaults to tracer A.
- Documents the final behavior: `pars_b` with omitted `cross_nuisance` raises
  `ValueError`, and pair-level EFT/stochastic parameters must be explicit.
- Keeps `PshotP_ab` documented only as a pair-level stochastic normalization;
  it is not described as an automatic Poisson cross-shot-noise prediction.

`docs/cross_power/CROSS_POWER_STAGE2_REVIEW.md`

- This review report.

`.DS_Store`, `docs/.DS_Store`

- Both were modified in the initial worktree. They were restored before final
  status checks so no `.DS_Store` file remains modified.

## API Review

Changed public signatures:

```python
get_eft_pkmu(self, kev, mu, pars, table, damping='lor', *, pars_b=None, cross_nuisance=None)
get_rsd_pkmu(self, k, mu, pars, table, table_now, IR_resummation=True, damping='lor', *, pars_b=None, cross_nuisance=None)
get_rsd_pkell(self, kobs, qpar, qper, pars, table, table_now, bias_scheme="folps",
              damping='lor', nmu=6, ells=(0, 2, 4), IR_resummation=True,
              *, pars_b=None, cross_nuisance=None, bias_scheme_b=None)
```

Changed private/internal signatures:

```python
_split_power_pars(self, pars)
_split_cross_nuisance(self, cross_nuisance)
_resolve_pair_parameters(self, pars, pars_b=None, cross_nuisance=None)
_validate_cross_spectrum_model(self, pars_b)
_get_eft_pkmu_pair(self, kev, mu, bias_a, bias_b, nuisance, table, damping='lor')
```

The old public positional API remains valid: all old positional parameters are
before `*`, and the new cross-spectrum controls must be passed by keyword.
Direct `get_rsd_pkmu` and `get_eft_pkmu` calls still require canonical FOLPS
ordering because those methods do not accept `bias_scheme`. `get_rsd_pkell`
canonicalizes tracer A with `bias_scheme` and tracer B with `bias_scheme_b` if
given, otherwise with `bias_scheme`.

The required nuisance correction was made. If `pars_b is not None` and
`cross_nuisance is None`, `_resolve_pair_parameters` raises:

```text
cross_nuisance must be supplied when pars_b is supplied; pair-level EFT and stochastic parameters must be specified explicitly for a cross-spectrum.
```

Auto mode with `pars_b=None` preserves the prior behavior and uses tracer A's
nuisance tuple. The code retains documented support for 8-value
`cross_nuisance` and 12-value FOLPS arrays, and rejects all other lengths with a
clear `ValueError`. It does not silently truncate arbitrary arrays.

## Physics Coefficient Review

Linear term: `get_rsd_pkmu` uses the cross-Kaiser polynomial
`(b1_a + f(k) mu**2) * (b1_b + f(k) mu**2) * P_L` in the IR-resummed linear
piece. This matches the Stage 1b formula.

Density-density loops: `_get_eft_pkmu_pair` polarizes the current auto
coefficients:

```text
b1_a b1_b P_dd
+ (b1_a b2_b + b2_a b1_b) P_b1b2
+ (b1_a bs2_b + bs2_a b1_b) P_b1bs2
+ b2_a b2_b P_b2b2
+ (b2_a bs2_b + bs2_a b2_b) P_b2bs2
+ bs2_a bs2_b P_s2s2
+ (b1_a b3nl_b + b3nl_a b1_b) sigma3^2 P_L
```

This reduces exactly to the previous auto convention with the factors of 2 in
the mixed monomials.

Density-velocity and velocity-velocity sectors: the code computes
`Pdt_a` and `Pdt_b` with independent tracer biases and enters them only through
`f0 * mu**2 * (Pdt_a + Pdt_b)`. The common velocity-velocity loop remains
`f0**2 * mu**4 * Ploop_tt`.

A term: for `A_full=False`, the implementation matches
`b1_a b1_b f0 mu^2 I1udd_1`, `0.5 (b1_a + b1_b) f0^2` multiplying the
`I2uud_1/I2uud_2` pair, and the pure velocity `f0^3` rows
`I3uuu_2/I3uuu_3`. For `A_full=True`, the six full-A rows carry the required
`0.25` endpoint-polarized factors for both `b2` and `bs2`. No averaged
effective bias is passed into the old single-tracer A helper; the pair
contraction writes the cross polynomial directly.

D and G: the D term is implemented as
`b1_a b1_b f0^2 D2 + 0.5 (b1_a + b1_b) f0^3 D3 + f0^4 D4`. The G term uses the
complete cross-Kaiser polynomial inside the existing FolpsD normalization.

EFT and stochastic terms: the pair-level nuisance tuple enters with the current
FOLPS normalization:

```text
P_ct = (alpha0 + alpha2 mu^2 + alpha4 mu^4) k^2 P_L
P_ct,NLO = ctilde (k mu f0)^4 sigma2w^2 P_K,AB
P_stoch = PshotP * (alphashot0 + alphashot2 (k mu)^2)
```

`PshotP_ab` is treated only as a pair-level stochastic normalization.

## Shared Auto/Cross Implementation

There is one non-marginalized pair-level physics contraction,
`_get_eft_pkmu_pair`. The auto path resolves `params_b = params_a` and uses
the current auto nuisance tuple when `pars_b is None`. Cross mode uses the
same contraction with independent `bias_a`, `bias_b`, and explicit pair
nuisance.

I did not find a remaining duplicate non-marginalized auto-spectrum polynomial
that could drift from the cross implementation. The marginalized helper
methods still duplicate auto logic by design and were not refactored because
the current task scoped only the non-marginalized cross implementation.

## IR, AP, and Interpolation Review

- The cross-Kaiser polynomial is used in the resummed linear term.
- The wiggle and no-wiggle EFT contractions are both evaluated through
  `_get_eft_pkmu_pair` with the same pair biases and nuisance.
- `interp_table` row indexing is unchanged: rows `table[1:28 + extra]` are
  interpolated and scalar tails remain in `table[28 + extra:]`.
- The scalar IR tail remains `sigma2_NW, delta_sigma2_NW` from `table_now[-3:-1]`.
- AP remapping still occurs in `get_rsd_pkell` before evaluating `get_rsd_pkmu`.
- The Jacobian remains `(qpar * qper**2)**(-1)`.
- The Gauss-Legendre multipole integration weights and summation are unchanged.

## JAX Review

The `NamedTuple` containers are compatible with JAX arrays. All optional
branching on `pars_b`, `cross_nuisance`, and model strings happens before the
numerical contraction. The model and damping strings remain Python-static. No
NumPy-only conversion was introduced inside `_get_eft_pkmu_pair`; the pair
kernel uses the module-selected backend `np`. Auto and cross return the same
array shapes for matching `k` and `mu` inputs. The retained 8-value and 12-value
cross nuisance handling is trace-safe in the current design because the length
branch is outside JIT-sensitive kernels.

## Test Independence Review

The original cross test could partially pass by using grouped expected
expressions for grouped production sectors. The final test file removes that
weakness for the requested sectors:

- Synthetic linear Kaiser uses a synthetic raw table and an analytic polynomial.
- Density rows `Ploop_dd`, `Pb1b2`, `Pb1bs2`, `Pb22`, `Pb2bs2`, `Pb2s2`, and
  `sigma23pkl` are checked one row at a time.
- Density-velocity rows `Ploop_dt`, `Pb2t`, `Pbs2t`, and velocity row
  `Ploop_tt` are checked one row at a time.
- A rows `I1udd_1`, `I2uud_1`, `I2uud_2`, `I3uuu_2`, and `I3uuu_3` are checked
  one row at a time.
- Each of the six full-A rows is checked one row at a time with asymmetric
  tracer biases, so the `1/4` factors are tested away from the auto limit.
- D rows and the G cross-Kaiser polynomial are checked analytically.
- Exchange symmetry uses the same explicit `cross_nuisance` in both tracer
  orders.
- Auto-limit and exchange-symmetry checks use tight same-backend tolerances,
  not the looser NumPy/JAX tolerance.

## Test Results

All required commands were run from `<repo-root>/folps`.

```bash
/opt/anaconda3/envs/aaenv/bin/python test_cross_power_spectrum.py
```

Exit status: 0.

Passed checks included both backends, `A_full=False`, `A_full=True`, IR off,
IR on, AP-remapped multipoles, explicit missing-nuisance rejection, invalid
parameter-length rejection, 12-value cross-nuisance extraction, synthetic
coefficient checks, current auto regressions, and NumPy/JAX cross agreement.

Key residuals from the final run:

```text
same-backend auto-limit and exchange checks: max_abs=0.000000e+00, max_rel=0.000000e+00
AP manual multipole integration, NumPy: max_abs<=7.275958e-12, max_rel<=3.169844e-15
AP manual multipole integration, JAX: max_abs<=5.456968e-12, max_rel<=4.184117e-15
synthetic linear Kaiser: max_abs=5.684342e-14, max_rel<=2.948896e-16, rtol=1.0e-12, atol=1.0e-10
synthetic one-row coefficient checks: max_abs<=2.220446e-16, max_rel<=9.172047e-16, rtol=1.0e-10, atol=1.0e-06
full-A one-row checks: max_abs<=8.673617e-19, max_rel<=2.075639e-16, rtol=1.0e-10, atol=1.0e-06
NumPy current auto regression: max_abs=9.356427e-10, max_rel=5.160729e-12
JAX current auto regression: max_abs=2.384186e-05, max_rel=8.438051e-11
NumPy/JAX cross multipoles: max_abs=1.214908e+00, max_rel=1.866533e-03, rtol=5.0e-03, atol=5.0e-02
NumPy/JAX cross pkmu: max_abs=1.273484e+00, max_rel=1.550098e-04, rtol=5.0e-03, atol=5.0e-02
```

Warnings/notices: NumPy CPU notice, JAX CPU notice. No cross test artifacts
were left in the repository; parent-mode temporary `.npz` files were created
under a temporary directory and removed by `TemporaryDirectory`.

```bash
/opt/anaconda3/envs/aaenv/bin/python test_folps_numpy.py
```

Exit status: 0.

Output artifacts were generated and then restored:

```text
folps/test_outputs_numpy/results_numpy.npz
folps/test_outputs_numpy/power_spectrum_numpy.png
folps/test_outputs_numpy/bispectrum_numpy.png
```

Timing:

```text
Matrix build:              0.002 s
Loop table:                0.071 s
Power spectrum multipoles: 0.012 s
Bispectrum multipoles:     0.009 s
```

Warnings: Matplotlib could not write to the default user cache and used a
temporary cache directory; Fontconfig reported no writable cache directories.
These warnings did not affect the exit status.

```bash
/opt/anaconda3/envs/aaenv/bin/python test_folps_jax.py
```

Exit status: 0.

Output artifacts were generated and then restored:

```text
folps/test_outputs_jax/results_jax.npz
folps/test_outputs_jax/power_spectrum_jax.png
folps/test_outputs_jax/bispectrum_jax.png
```

Timing:

```text
Pnow precompute:                 2.443 s
Matrix build:                    0.001 s
Loop table setup:                2.338 s
PK JIT first run (compile+exec): 2.228 s
PK JIT cached run:               0.007 s
PK speedup:                      340.65x
PK improvement:                  99.7%
BK JIT first run (compile+exec): 0.416 s
BK JIT cached run:               0.001 s
BK speedup:                      291.18x
BK improvement:                  99.7%
```

Warnings: Matplotlib and Fontconfig cache warnings as above, plus JAX CPU notice.

```bash
/opt/anaconda3/envs/aaenv/bin/python test_compare_folps_numpy_vs_jax.py --skip-run
```

Exit status: 0.

```text
Tolerances: atol=5.00e-02, rtol=5.00e-03
p0:      max_abs=1.023e+00, max_rel=2.919e-04, PASS
p2:      max_abs=4.823e-01, max_rel=1.793e-03, PASS
p4:      max_abs=3.209e-02, max_rel=4.537e-04, PASS
p0_marg: max_abs=1.023e+00, max_rel=2.918e-04, PASS
p2_marg: max_abs=4.823e-01, max_rel=1.793e-03, PASS
p4_marg: max_abs=3.209e-02, max_rel=4.538e-04, PASS
b000:    max_abs=1.318e+04, max_rel=1.463e-05, PASS
b110:    max_abs=1.285e+04, max_rel=4.997e-04, PASS
b220:    max_abs=6.412e+03, max_rel=1.987e-05, PASS
b202:    max_abs=1.338e+04, max_rel=2.518e-05, PASS
b022:    max_abs=1.338e+04, max_rel=2.518e-05, PASS
b112:    max_abs=1.590e+04, max_rel=2.844e-05, PASS
Overall status: PASS
```

## Fixes Made

- Required correction: cross mode now raises `ValueError` when `pars_b` is
  supplied without explicit `cross_nuisance`.
- New cross API arguments are keyword-only.
- Invalid 12-value tracer parameter lengths are rejected clearly.
- Invalid `cross_nuisance` lengths are rejected clearly; accepted lengths are
  exactly 8 and 12.
- Cross implementation documentation now matches the corrected nuisance API.
- Cross tests now include independent one-row synthetic checks for all required
  physics rows and print residuals for successful comparisons.
- Tracked generated test artifacts were restored after recording results.

## Remaining Limitations

- Historical Stage 2 limitation: cross spectra were limited to `model="EFT"`
  with no cross-FoG damping model. This is superseded by the current FolpsD
  cross damping implementation.
- Direct `get_rsd_pkmu` and `get_eft_pkmu` cross calls require canonical FOLPS
  parameter ordering; only `get_rsd_pkell` applies `bias_scheme` and
  `bias_scheme_b`.
- The retained 12-value `cross_nuisance` form intentionally uses the last
  8 values by convention. This is documented and tested, but the 8-value tuple
  is the clearer pair-level API.
- Marginalized helper methods are still auto-only and were intentionally left
  out of scope.
- No window convolution, odd multipoles, redshift evolution, wide-angle terms,
  velocity bias, bispectrum cross-correlation, DESI data loading, or production
  matrix change is included.
