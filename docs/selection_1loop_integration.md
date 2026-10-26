# Integrating the selection model with the DESI Fisher and NERSC simulation pipelines

This document lists the interfaces the existing pipelines need. **None of those pipelines
was available or run during this implementation.** No DESI data vector, covariance,
window or simulation was used, and no Fisher or NERSC fit result is claimed. The toy Fisher
example in the tutorial (section 22) uses a labeled diagonal toy covariance only.

## 1. Per-cosmology setup (once per linear spectrum)

```python
from folps import MatrixCalculator, NonLinearPowerSpectrumCalculator, build_selection_tables
mm = MatrixCalculator(nfftlog=128, A_full=True, use_TNS_model=False).get_mmatrices()   # cached on disk if save_dir
nl = NonLinearPowerSpectrumCalculator(mmatrices=mm, kernels="EdS", **cosmo)              # EdS required
table, table_now = nl.calculate_loop_table(k=k, pklin=pk_lin, pknow=pk_nw_or_None, cosmo=None, **cosmo)
sel_tables = build_selection_tables(nl, cache_dir="/path/to/matrix_cache")               # ~0.1 s warm
```

* Requirements enforced by the code: `kernels="EdS"`, `A_full=True`, `use_TNS_model=False`,
  `model="EFT"`, auto spectrum. The legacy table and the selection tables must come from
  the **same** calculator call (same `f0`, same no-wiggle spectrum); a table-pairing check
  and an `f0` check run whenever the inputs are concrete.
* Matrix cache: `get_selection_matrices(..., cache_dir=...)` stores two `.npz` files
  (≈0.5 MB each at `nfftlog=128`) keyed by generator version, coefficient checksum, grid,
  tilt and dtype; incompatible files are rejected.
* Accuracy: `nfftlog=128` (FolpsD default) gives ~1e-3 on the loop for k ≲ 0.25 h/Mpc and
  up to ~3e-3 above, for the baseline and the selection alike; `nfftlog=256` gives ~1e-5
  (validation report). Use 256 for validation runs or if the k range extends past 0.25.

## 2. Per-parameter-point call

```python
P_ell = calc.get_rsd_pkell(kobs, qpar, qper, pars, table, table_now, bias_scheme="folps",
                           damping=None, nmu=6, ells=(0, 2, 4),
                           selection=SelectionBias(...), selection_tables=sel_tables,
                           eft_extension=EFTExtension(alpha6=..., alphashot2_iso=..., alphashot2_mu4=...))
```

* Output shape `(len(ells), len(kobs))`, units (Mpc/h)^3, AP Jacobian included; window
  convolution and scale cuts stay in the pipeline (apply them to these multipoles as for
  the legacy model).
* Parameters (all differentiable with JAX; no branching on values):
  * legacy 12-vector `pars` (any `bias_scheme`, converted as before);
  * the nine selection coordinates (`SelectionBias`, zero = no selection);
  * optional `EFTExtension` linear terms (zero = legacy);
  * optional `PhysicalDerivativeBias` (replaces `alpha0..alpha4` and `alpha6`; those must
    then be zero).
* Three models to compare:
  1. **no selection**: `selection=None` (identical to the legacy model);
  2. **historical tree-only `bA`**: legacy argument `bA=...` of `get_rsd_pkmu` (not exposed
     by `get_rsd_pkell`; project with the same Gauss rule if needed, as in the tutorial,
     section 11);
  3. **full selection**: `selection=SelectionBias(...)`, with the linear coordinates
     matched to model 2 by `tree_bA_to_linear_djs(b1, bA, f)` if desired. State the
     convention used for the eight nonlinear coordinates explicitly (e.g. all zero, or a
     named coevolution relation, which is not provided as a default).

A minimal adapter implementing the three models on one parameter dictionary, with
ell-major data vectors and finite-difference Jacobians, is
`scripts/selection_fisher_adapter_example.py` (`SelectionPowerSpectrumModel`). Its `__main__`
runs a demo with a **toy** diagonal covariance only.

## 3. Analytical marginalization

`get_rsd_pkell_marg_const(..., selection=..., selection_tables=..., eft_extension=...)` and
`get_rsd_pkell_marg_derivatives(...)` return the constant part and the design vectors of
the **genuinely linear** parameters (`alpha0, alpha2, alpha4, alphashot0, alphashot2` and,
with `eft_extension`, `alpha6, alphashot2_iso, alphashot2_mu4`), as exact differences of the
complete selected model with AP and IR. The selection coordinates are **not** linear (P22 is
quadratic and `s_eta` also enters `Z1`), so they must be sampled, never marginalized
analytically. The legacy derivative helper (without `selection`) is unchanged; note that it
integrates over μ with `nmu` full-range nodes while the model uses `2 nmu` symmetric nodes
(an ℓ=4 inconsistency of ~3e-4 with IR on). The gated helpers do not have this issue.

## 4. Fisher calculations

* Data vector ordering: the pipeline's own; the tutorial uses ℓ-major `[(ℓ, k)]`.
* Jacobians: `jax.jacfwd` through `get_rsd_pkell` with `FOLPS_BACKEND=jax` and
  `jax_enable_x64` (validated against central finite differences: exact for bias and
  selection directions, O(h²) for AP, 1e-11 for a P_L amplitude). Template construction is
  traceable, so cosmology derivatives through `pklin` are possible. Matrix generation is
  not traced (host NumPy, cached).
* Report rank, conditioning and prior assumptions. Nine selection coordinates do not imply
  nine measured directions: several responses are close to degenerate in (P0, P2, P4)
  (tutorial sections 12 and 22). Never use a pseudo-inverse without reporting this.
* For the LRG2 Fisher: reuse the existing data vector, covariance, window and cuts *when
  they are supplied*; do not substitute the tutorial's toy covariance.

## 5. Returning to the NERSC simulations

Use the same pipeline and environment that produced the previous successful fit:
reproduce that fit first with `selection=None` (it must agree with the earlier result),
then switch to `selection=SelectionBias(...)` with the selection coordinates free (and
`eft_extension` terms as decided). No new simulations need to be downloaded, and nothing
needs to be rebuilt outside that pipeline. Install this branch in that environment with
`pip install -e .` on a checkout of `feature/selection-1loop-eds`.

## 6. Limitations relevant for inference

* EdS kernels with constant `f`: the massive-neutrino `fk` baseline cannot be combined with
  full selection (rejected). An "EdS selection increment on an `fk` baseline" is
  deliberately not offered.
* Selected cross spectra are rejected.
* The `k^2 mu^4` UV content of the selected P22 (validation report, section 6) suggests that
  a complete stochastic basis for selected tracers includes a `k^2 mu^4` noise shape
  (`alphashot2_mu4`). Whether to free it, and with which prior, is a modeling decision for
  the inference.
