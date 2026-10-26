# Validation report: EdS one-loop LOS selection in FolpsD

Branch `feature/selection-1loop-eds`, base `6012d6d`. Measured on a NERSC Perlmutter login
node (AMD EPYC, CPU only), Python 3.12.12, numpy 2.5.4, scipy 1.18.1, jax 0.12.0.1 (x64),
sympy 1.14.0. All numbers below were produced by code in this branch and can be
regenerated with the commands given in each section. Equations and conventions:
[`selection_1loop_equations.md`](selection_1loop_equations.md).

## 0. Gate summary

| gate (brief) | status | evidence |
|---|---|---|
| A. kernel identities and non-trivial cubic checks, without using the implementation as expected output | **passed** | sections 2–3 |
| A. F.7 eta discrepancy resolved before importing matrices | **passed**: corrected row derived, nothing imported | section 3 |
| B. standard comparison explained and validated (zero selection) | **passed**: complete loop = FolpsD to 1.6e-5 (N=256) | section 5 |
| B. generic nonzero selection = independent oracle after the same subtractions | **passed** | section 6 |
| B. renormalization bridge derived (no fitted correction) | **passed**: closed form | section 4 |
| C. fast vs independent within declared tolerances; FFTLog convergence | **passed** (with the FFTLog-resolution caveat at N=128) | sections 6–7 |
| D. legacy calls unchanged; IR on/off, AP unity/non-unity; invalid modes fail | **passed** | sections 8–9 |
| D. marginalization helpers gated and checked against numerical derivatives | **passed** | section 10 |
| backend parity and derivatives | **passed** | section 11 |
| E. tutorial: fresh-kernel execution, real outputs | **passed**: 22 passed, 0 failed, 1 skipped (slow) | section 13 |
| F. Fisher / NERSC runs | **not run** (pipelines not available) | `selection_1loop_integration.md` |
| `fk` kernels with selection | **not implemented** (rejected) | section 14 |

## 1. How to run

```bash
python -m pytest tests -q                    # 54 tests (4 slow skipped), ~2 min; JAX tests in subprocesses
python -m pytest tests -q --runslow          # all 58, ~3 min
python scripts/validate_selection_reference.py          # direct quadrature fixture, ~25 min
python scripts/selection_convergence.py                 # FFTLog resolution / tilt / domain, ~10 min
python scripts/generate_selection_coefficients.py --check   # regenerate the coefficient data and diff, ~50 min
python -I dev/selection_audit/audit_checks.py           # reconstructed audit checks, <1 min
python -I dev/selection_audit/compare_djs_supplement.py PATH/pkgs_supplement.nb   # external comparison
cd folps && python test_folps_numpy.py && python test_cross_power_spectrum.py     # upstream scripts
```

## 2. Kernel algebra (oracle `folps/selection_kernels.py`)

200–300 random configurations, general LOS, unequal sides, \(f=0.73\) (`tests/test_selection_kernels.py`):

| identity | max abs deviation |
|---|---|
| \(F_2,G_2\) recursion vs closed form | 2.7e-15 |
| \(\Pi^{[2](2)}\) recursion vs explicit tensor | 1.8e-15 |
| \(\operatorname{tr}\Pi^{[2]}=\vartheta-\delta+\delta\vartheta\), 2nd and 3rd order (continuity) | 1.0e-14 |
| \(\operatorname{tr}\Pi^{[3](3)}\) vs Euler + continuity identity (derived here) | 2.1e-14 |
| mapped \(Z_2\) vs hand-written ten-kernel table (E5) | 1.1e-14 |
| zero-selection \(Z_2\) vs standard redshift-space \(Z_2\) | 1.1e-14 |
| exact rest-frame map \(Z_1,Z_2\) vs mapped field | 4e-15 |
| \(Z_2,Z_3\) permutation + LOS reversal | 7e-15 |
| growth powers \(f^{n_f}\) of 10 quadratic and 16 cubic kernels | 1e-12 relative |
| full spatial vs LOS-only advection (must differ) | differ at O(1) |
| collapsed limit \(Z_3(\mathbf k,\mathbf q,-\mathbf q+\epsilon)\to\) exact evaluation | error decreases as \(\epsilon^2\), < 1e-8 |
| tree \(b_A\) identity; physical-derivative conversion and \(\mu^6\) relation | 4e-16; 1e-13 |

**IR soft limit (kernel level).**
\(q^2[4\langle Z_2(\mathbf q,\mathbf k-\mathbf q)^2\rangle+6Z_1\langle Z_3(\mathbf k,\mathbf q,-\mathbf q)\rangle]\) at \(q/k=10^{-2},10^{-3},10^{-4}\)
with all nine selection coordinates nonzero:

| field | \(\mu=0.2\) | \(\mu=0.7\) | \(\mu=1\) |
|---|---|---|---|
| exact map (B) | 6.0e-5, 6.0e-7, 6.0e-9 | 4.3e-4, 4.3e-6, 4.3e-8 | 1.1e-3, 1.1e-5, 1.1e-7 |
| reduced mapped field (A) | 1.7e-2 (constant) | 0.27 (constant) | 0.71 (constant) |

The exact map cancels as \((q/k)^2\). The mapped field does not, because it lacks the
double displacement. This is the reason for the bare bridge in section 4.

## 3. Coefficient generation and cubic matrices

`scripts/generate_selection_coefficients.py` (2947 s, one core); report
`docs/selection_1loop_generation_report.json`; data sha256 `b8db1b3b…be42537`.

* P22: exact SymPy reduction; the generated set equals the 23 DJS functions (asserted). The
  pair classification (no UV constant with a soft-singular kernel; no UV \(k^2\) for
  singular × singular) holds in exact arithmetic. The tensor reproduces the oracle's
  azimuthally averaged \(Z_2^2\) at random points to 3e-13.
* Cubic: pointwise identification of 16 operators, fit residual ≤ 4.0e-13, \(\mu\)-polynomial
  residual ≤ 5.6e-12, rationalization error ≤ 6.3e-11 (denominators ≤ 735).
* Bridge: (A) per DJS coefficient and (B) per rest-frame coefficient, 6 × 7 \((f,\mu)\) grid,
  degree-3 fits: residual ≤ 4.0e-13 / 8.7e-12. The J parts of (A) and (B) equal the reduced
  list to 5.3e-9 / 4.6e-9 (fit noise). The bare bridge equals \(-\tfrac16[1+f\mu^2(2+f)]Z_1\) to 8.5e-14.
* **Eta row:** adopted \((-2,\tfrac32,\tfrac12,0,0)/7\); printed DJS F.7 \((2,\tfrac32,-\tfrac12,0,0)/7\) rejected.
  Reconstructed audit at \(k=0.4\): \(\eta/(f\mu^2)\) response from integrated \(G_3\)
  \(3.8289945\times10^{-4}\) (audit: \(3.82899421\times10^{-4}\)). The adopted row gives the same
  value to 1e-15. The printed row gives \(6.17878\times10^{-4}\) (audit: \(6.17892\times10^{-4}\)).
  Collapsed \(F_3,G_3\) identities hold to 2.8e-17 on the toy spectrum.

**External comparisons.**

| source | result |
|---|---|
| GeneralBiasPk `M13.jl` (commit `d4d5bee`) | all 16 cubic matrices identical |
| DJS supplement `Anmp` (commit `2ecd100`), 315 polynomial entries parsed symbolically | identical to 4.5e-14 **iff** the \(u_\parallel\partial_\parallel\delta\) kernel sign is flipped. All ten other single-kernel sign hypotheses fail at O(1). Our sign reproduces the standard \(Z_2\) and FolpsD's P22; see the sources document. |
| FolpsD closed forms | \(J\) basis reproduces `M13_dd` and `M13_tt+M13_dt` (EdS) to 1e-16; the UV bookkeeping reproduces \(-\tfrac{61}{105}\) and \(-\tfrac95\,k^2\sigma_\psi^2\) exactly; P22 tensor + matrices reproduce `P22dd` to 2e-14; FolpsD's \(b_2,b_{s^2}\) terms (tilt −1.51 with zero-lag extrapolation) agree with our local class to <1e-5 at N=256 (6e-4 at N=128). This also confirms \(b_{s^2}=2b_{K^2}\). |
| GeneralBiasPk run | **not run** (Julia packages not installed; the matrices were transcribed and compared instead) |

## 4. Renormalization bridge

Implemented P13 = bare exact-map scheme (equations section 7):

\[
L_{\rm implemented}-L_{\rm DJS\ reduced}=-k^2\sigma_\psi^2[1+f\mu^2(2+f)]Z_1^2P_L ,
\]

derived (no fit), of derivative-counterterm shape and IR-relevant. The constant
\(\sigma_0^2\) pieces renormalize \(b_1,b_\eta\) and are dropped, as in FolpsD.

**Spectrum-level IR insensitivity** (`test_ir_insensitivity_of_complete_selected_loop`):
adding a log-normal bump of 30 % of \(P_L\) at \(q_0\), the selected P22 and P13 each change by
\(\sim k^2\delta\sigma^2P(k)\), while their sum changes by a fraction \(2\times10^{-3}\) of that
(\(q_0=0.002\)) or \(1.1\times10^{-2}\) (\(q_0=0.005\)), at \(0.085<k<0.3\), \(\mu=0.7\). The ratio scales as \(q_0^2\).

## 5. Zero-selection comparison with FolpsD (Stage B gate)

The complete template loop at \(\mathbf s=0\) (P22 + bare P13, all built from the selection
templates and kernels) against FolpsD's complete one-loop EdS EFT loop (Kaiser loops +
\(A\) + \(D\) + GTNS, `A_full=True`), IR off, AP = 1, at all output-grid points
\(0.005<k<0.4\), relative to the loop scale:

| \(\mu\) | 0 | 0.3 | 0.6 | 0.9 | 1 |
|---|---|---|---|---|---|
| nfftlog = 128 | 8.6e-4 | 6.9e-4 | 3.9e-4 | 1.7e-4 | 1.3e-4 |
| nfftlog = 256 | 1.6e-5 | 1.3e-5 | 6.8e-6 | 3.1e-6 | 2.3e-6 |

The residual converges with FFTLog resolution, so no scheme difference remains. FolpsD's
loop, including its \(b_2,b_{s^2}\) RSD terms, is in the same scheme (zero-lag constant only),
and the \(b_{3\rm nl}\leftrightarrow b_{\rm td}\) adapter matches the \(\sigma_3^2\) normalization.

## 6. Generic nonzero selection vs direct quadrature

`scripts/validate_selection_reference.py` → `tests/data/selection_reference_fixture.npz`
(1371 s). All nine coordinates nonzero (`s = (0.21, −0.33, 0.27, 0.45, −0.52, 0.31, −0.23,
0.36, 0.19)`), \(b_1=1.645\), FolpsD's fixture spectrum, 8 grid \(k\) × 3 \(\mu\). Reference: oracle
kernels of the exact map, direct radial/polar/azimuthal quadrature (P22: 300×40×24 with
region splitting; P13: 800 radial nodes with pole-adapted polar rule, 96×16). It was
itself checked against FolpsD's `P13dd`/`P22dd` (9e-7/1e-7) and for convergence (1e-9; the
P13 radial rule was increased from 300 to 800 nodes after the \(\int P\,dq\) check showed
6e-5 under-resolution at low \(k\)). Residuals are relative to each component's max over \(k\)
at that \(\mu\):

| quantity | N=128, k < 0.25 | N=128, k > 0.25 | N=256, all k |
|---|---|---|---|
| selected P22 | 5.7e-4 | 3.7e-3 | 1.7e-5 |
| selected bare P13 | 1.1e-5 | 3.5e-5 | 2.2e-7 |
| increment \(L(\mathbf s)-L(0)\) | 4.5e-4 | 3.6e-3 | 3.9e-5 |

Declared tolerances (tests): N=128: P22 and increment 1e-3 (k < 0.25) and 5e-3 (k > 0.25),
P13 1e-4. N=256: P22 5e-5, P13 1e-6, increment 1e-4.

## 7. FFTLog convergence (`scripts/selection_convergence.py`)

Reference nfftlog = 512, tilt −0.9; max relative deviation of the selected P22 / P13 (own
scale), \(\mu\in\{0.1,0.6,1\}\):

| case | P22 full, k<0.25 / <0.45 | P22 no-wiggle | P13 full | P13 no-wiggle |
|---|---|---|---|---|
| N = 128 | 1.1e-3 / 4.7e-3 | 1.3e-4 / 1.5e-4 | 1.4e-5 / 3.6e-5 | 4.9e-7 / 1.1e-6 |
| N = 256 | 1.4e-5 / 2.8e-5 | 9.4e-6 / 2.4e-5 | 2.4e-7 / 3.9e-7 | 2.6e-7 / 4.4e-7 |
| N = 256, local tilt −0.85 / −0.95 | as N = 256 (P22 changes ≤ 1.6e-7) | as N = 256 | unchanged | unchanged |
| N = 256, input truncated at k ≤ 1 | 1.5e-5 / 2.6e-5 | 8.5e-5 | 2.3e-3 | 2.4e-3 |

* The N = 128 P22 error is FolpsD's default FFTLog resolution (the same 3e-3 appears in
  FolpsD's own `P22dd` vs converged quadrature at k = 0.28, 0.41). It is larger for the
  full-shape table (BAO wiggles) than for the no-wiggle table.
* Within the strip, the local tilt does not matter. Its edges do: at −0.6, −0.7, −0.8 the
  \(k^2\) piece of the \(K^2\times K^2\) pair is off by −4151, −64, −1.0 \(k^2\), matching the
  periodic-image estimate \((k_{\max}/k_{\min})^{1+2b}\). Hence −0.9.
* The input-domain test changes \(\sigma_{-1}^2\): the bare P13 is UV-sensitive through the
  \(k^2\sigma^2P_L\) term only, a counterterm shape. P22 is not affected.

## 8. Legacy behavior, explicit zero, supported modes

* Legacy calls (`selection=None`) reproduce the frozen pre-edit reference
  (`tests/data/eft_baseline_reference.npz`, produced on the unmodified commit) **bit for
  bit**: EdS and `fk`, IR on/off, AP unity and \((1.03,0.98)\).
* An explicit `SelectionBias()` runs the selection machinery (increments computed, exactly
  zero) and returns arrays bit-identical to the legacy call, for all IR/AP cases and for
  the `desi` bias scheme.
* Upstream scripts: `folps/test_folps_numpy.py` outputs are bit-identical to the run before
  the edits; `folps/test_cross_power_spectrum.py` passes (321 checks); `folps/test_folps_jax.py`
  outputs are bit-identical between base commit and branch.
* Rejected with informative errors: selection without tables; selection + `bA`; selected
  cross spectra; `model != "EFT"`; `kernels="fk"` tables or calculators; `A_full=False` or
  `use_TNS_model=True`; physical-derivative mode with nonzero `alpha`'s; incompatible
  matrix caches; mismatched full/no-wiggle tables.
* Historical tree-only `bA` path is unchanged and still works.

## 9. IR, AP and angular quadrature

* One-loop re-expansion: scaling \(\Sigma^2,\delta\Sigma^2\) by \(\epsilon^2\) in the table, the selected
  \(P_{\rm IR}(\epsilon)-P_{\rm IR\ off}\) has relative size 5.5e-4, 1.5e-4, 3.8e-5, 9.6e-6 for \(\epsilon=0.2,0.1,
  0.05,0.025\) (ratios 3.70, 3.92, 3.98 → 4). The resummation changes the selected prediction only
  beyond one loop, exactly as for the legacy model (5.3e-4, …, 9.1e-6).
* With IR on, selected − unselected = the IR-mixed tree and loop increments (to 1e-10).
  \(\Sigma_s\) is computed from the physical \(f\) only. Components sum to the total (1e-12).
* \(\max|L[P_L]-L[P_{\rm nw}]-L[P_w]|/\max|L[P_L]-L[P_{\rm nw}]|=1.0016\) at \(\mu=0.6\): the
  loop of the wiggles alone is negligible, and the smooth×wiggle cross terms carry the whole
  wiggle loop (tutorial section 13).
* AP: an explicit evaluation of \(P(k(k_{\rm obs},\mu_{\rm obs}),\mu(\mu_{\rm obs}))/(q_\parallel q_\perp^2)\)
  projected with a 64-node rule reproduces `get_rsd_pkell` for the selected model to ≤ 9.2e-7 at
  \((q_\parallel,q_\perp)=(1.04,0.97)\). True \(k\) stays within the template domain \([0.001,0.5]\).
  The NumPy route's AP derivatives converge as \(h^2\).
* \(\ell=6,8\) diagnostics (projected from `get_rsd_pkmu`, since FolpsD's `legendre` helper stops at
  \(\ell=4\)): \(\max|P_6|/\max|P_0|=8.8\times10^{-4}\), \(\max|P_8|/\max|P_0|=4.8\times10^{-5}\) without IR at
  unity AP. With IR and AP \((1.04,0.97)\) they are \(2.6\times10^{-3}\) and \(1.8\times10^{-4}\), with a
  12-node (default) angular error of \(2.5\times10^{-5}\) and \(7.3\times10^{-4}\) relative to their own size.
* Angular quadrature (tutorial section 16, IR on, AP ≠ 1), relative to `nmu=96`:
  `nmu=3`: (4e-7, 2e-5, 3.2e-3) for (P0, P2, P4); default `nmu=6` (12 full-range nodes):
  (5.8e-9, 3.6e-8, 9.2e-7); `nmu=12`: ≤ 4e-8.

## 10. Derivative / noise sectors and marginalization

* `alpha6`, `alphashot2_iso`, `alphashot2_mu4` have exactly their declared shapes (tests).
  Physical-derivative mode equals the converted polynomial mode to 1e-12.
* Rank (tutorial section 14): the normalized \((\alpha_0,\alpha_2,\alpha_4,\alpha_6)\) templates in
  \((P_0,P_2,P_4)\), \(0.01<k<0.3\), have singular values (1.73, 0.93, 0.37, 2.9e-13) without IR,
  so \(\alpha_6\) is exactly degenerate, and (1.73, 0.93, 0.37, 1.1e-4) with IR.
* Toy Fisher (tutorial section 22, diagonal toy covariance, *not* DESI): the 12 parameters
  (9 selection coordinates, \(b_1\), \(q_\parallel\), \(q_\perp\)) give Fisher rank 11 at relative
  tolerance 1e-10, condition number 8.5e10. Not all selection directions are independent in
  \((P_0,P_2,P_4)\) at fixed nuisance parameters.
* Model comparison (tutorial section 11): at matched linear kernel, the full one-loop model
  with zero nonlinear selection coefficients differs from the historical tree-only \(b_A\)
  model (\(b_A=0.25\)) by 0.4 %, 1.9 % and 7.3 % of \(\max|P_\ell|\) in \(P_0,P_2,P_4\) for
  \(0.01<k<0.3\). This is the effect of the selected loops alone; any nonzero nonlinear
  coefficients add to it.
* Gated `get_rsd_pkell_marg_const` / `_derivatives`: the design vectors reconstruct the full
  selected prediction (AP ≠ 1) to 1e-10 and equal central finite differences of the
  complete model to 1e-9. At zero selection they agree with the legacy helpers to round-off,
  once the legacy derivative helper is given matching angular nodes. **Pre-existing legacy
  inconsistency (not changed):** the legacy derivative helper uses `nmu` full-range nodes
  while the model uses `2 nmu` symmetric ones, giving a 3e-4 ℓ = 4 mismatch with IR on.

## 11. Backends and derivatives

* Contraction-level parity NumPy vs JAX (identical arrays): ≤ 1e-11 relative.
* End to end at table nodes with an identical no-wiggle input: legacy and selected
  predictions agree to ≲ 2e-6, the selection increment to 6e-7. The residual comes from
  FolpsD's backend-specific interpolation of \(P_L\) onto the FFTLog grid (`get_cm`), shared by
  both. Off-node (AP ≠ 1) the legacy cubic spline vs JAX `interp` difference (≲ 7e-4 in
  \(P_4\)) appears identically in legacy and selected predictions.
* `jax.jit` = eager to 2e-11. `jax.jacfwd` vs central differences: the nine selection
  coordinates and `b1` (including its \(b_{\delta\eta}\) offset) agree to ≤ 2e-10 at every step
  (the model is polynomial in them); `qpar`, `qper` converge as \(h^2\) to 1e-8 / 3e-7. The
  linear-spectrum amplitude derivative through `calculate_loop_table` and
  `build_selection_tables` agrees to 1.2e-11.
* Not differentiable: the cosmology-independent matrix generation (host NumPy, cached),
  and in the NumPy route the SciPy spline interpolation.

## 12. Timings (measured; cold = matrices not cached)

| step | time |
|---|---|
| selection FFTLog matrices, 2 tilts, nfftlog = 128 (cold, once per configuration) | 12.3 s |
| same, nfftlog = 256 | 39 s |
| legacy loop tables (`calculate_loop_table`) | 0.06–0.8 s |
| selection templates (`build_selection_tables`, warm matrices) | 25–50 ms |
| legacy multipoles, 60 k, nmu = 6 | 13–16 ms |
| selected multipoles, 60 k, nmu = 6 | 46–55 ms |

Ranges reflect load from concurrent jobs on a shared login node. No speedup relative to
any other code is claimed.

## 13. Tutorial execution

`notebooks/selection_1loop_tutorial.ipynb`, executed headless in a fresh kernel:

```bash
cd notebooks && python -m jupyter nbconvert --to notebook --execute selection_1loop_tutorial.ipynb \
    --output selection_1loop_tutorial.ipynb --ExecutePreprocessor.timeout=1800
```

Execution date 2026-10-26, wall time 1 min 57 s, 13 figures, no errors or stderr output.
Validation summary in the notebook: 22 passed, 0 failed, 1 skipped (recomputation of a
direct-quadrature point; enabled with `FOLPS_RUN_SLOW=1`). The JAX parity section ran (JAX
installed) in subprocesses.

## 14. Remaining limitations and open issues

1. **`fk` / massive-neutrino kernels:** not implemented (requires rederiving velocity
   kernels, convective time derivatives of \(\Pi^{[a]}\) with non-separable growth, the EdS
   operator equivalences and all templates). Rejected in selection mode.
2. **Cross spectra with selection:** not implemented (rejected).
3. **FFTLog resolution:** at FolpsD's default `nfftlog=128`, P22 (baseline and selection)
   has ~1e-3 (k < 0.25) to ~5e-3 (k ≤ 0.45) errors. Use 256 when that matters.
4. **Stochastic basis (theory question):** the UV-sensitive order-\(k^2\) part of the selected
   P22 contains a \(k^2\mu^4\) shape for every nonzero selection coordinate. It is absent
   without selection, and \(\mu^6,\mu^8\) are absent always (checked both from the generated
   tensor and directly from oracle kernels). The DJS/GeneralBiasPk stochastic basis
   (\(P_\epsilon^{(0)},P_\epsilon^{(2)},P^{(2)}_{\epsilon\epsilon_\eta}\)) has no such term. The model keeps the full
   finite P22, so predictions are unaffected, but an EFT-complete fit may need
   `alphashot2_mu4` (provided, default 0). Needs theory confirmation.
5. **DJS supplement sign:** the supplement's `Anmp` corresponds to the opposite sign of the
   \(u_\parallel\partial_\parallel\delta\) kernel; we keep the sign validated against standard RSD and
   FolpsD. Not author-confirmed.
6. **Legacy derivative-helper angular rule** (section 10): documented, unchanged.
7. **GeneralBiasPk** was compared through its matrices only; the Julia code was not run.
8. **No DESI Fisher or NERSC fits were run.** The interfaces are documented in
   `selection_1loop_integration.md`. The tutorial's Fisher example uses a labeled toy
   covariance.
9. The shipped linear spectrum includes massive neutrinos but is used with EdS kernels
   and constant \(f\): a mathematical benchmark, not a self-consistent forecast.
