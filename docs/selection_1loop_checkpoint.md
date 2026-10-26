# Selection one-loop development checkpoint

Running log for the `feature/selection-1loop-eds` branch. Newest entries at the bottom
of each section. Commands are run from the repository root with the workspace virtual
environment (`../.venv`, see "Environment").

## Starting state (2026-10-26)

- Clone: `https://github.com/cosmodesi/FolpsD` into `Folps_IA/FolpsD` (fresh clone; no
  pre-existing checkout to preserve).
- Base commit: `6012d6d52791cba46b5c762dc4020f7544625ee3` ("Merge cross-ruiyang into main").
- `folps/folps.py` file blob: `246a8077f650325cb08888474654457851916227` — identical to the
  blob inspected by the pre-implementation audit.
- Branch created: `feature/selection-1loop-eds` (name was free; no remote branch of that name).
- Repository instructions: no `AGENTS.md`, `CLAUDE.md` or `CONTRIBUTING` file exists.
  Existing tests are standalone scripts `folps/test_*.py` (run from inside `folps/`).
- Pre-existing observation: `folps/test_outputs_numpy/results_numpy.npz` (committed) does not
  match HEAD (`p0` differs by 0.4 %, `p4` by 10 %; the marginalized arrays agree to 1e-14),
  i.e. the committed figure/data outputs are stale with respect to later damping commits.
  Running `test_folps_numpy.py` overwrites them; they were restored with `git checkout`.

## Environment

- NERSC Perlmutter login node, `module load python/3.12-26.1.0`, venv `Folps_IA/.venv`.
- numpy 2.5.4, scipy 1.18.1, jax/jaxlib 0.12.0.1 (CPU), interpax 0.3.16, matplotlib 3.11.2,
  sympy 1.14.0, mpmath 1.3.0, pytest 9.1.1, nbconvert 7.17.2, ipykernel 7.4.0.
- `pip install -e .` (editable) inside the venv.
- Julia available as a module (`julia/1.12.1` default); not used by the core implementation.

## External sources fetched (data only, not executed)

- DJS supplement `https://github.com/djeong98/pkgs_supplement` @ `2ecd100b2ad68ec7e1e19c6d56aa45bd01dcf682`
  (Mathematica notebooks).
- GeneralBiasPk `https://github.com/JosephTomlinson/GeneralBiasPk` @ `d4d5bee2477d6efd719a4c7193fef0261d88334f`
  (Julia). `P13/M13.jl` contains all 16 cubic matrices; its `η3` row is `(-2, 3/2, 1/2)/7`,
  i.e. the audit's corrected row, not the printed DJS F.7 row.

## Completed milestones

1. Frozen pre-edit EFT reference: `scripts/freeze_eft_baseline.py` →
   `tests/data/eft_baseline_reference.npz` (EdS and fk tables, IR on/off, AP unity and
   `qpar=1.03, qper=0.98`, `P_ell` and `P(k,mu)`), produced on the unmodified commit.

2. Kernel oracle `folps/selection_kernels.py` (field algebra: EdS F/G recursion, Pi hierarchy
   recursion, normalized symmetrized products, LOS advection, exact redshift-space map).
   Checks (300 random configurations, general LOS, f=0.73):
   F2/G2 vs closed form 2.7e-15; Pi2 tensor identity 1.8e-15; tr Pi2 at 2nd/3rd order vs
   continuity 1e-14; tr Pi3 at 3rd order vs an Euler/continuity identity derived independently
   2e-14; Z1, Z2 of the DJS mapped field vs the hand-written E5 table 1e-14 and vs the
   standard Z2 1e-14; Z2 of the exact rest-frame map (with the derived rest-frame map) equals
   the mapped Z2 to 4e-15; Z3 real, permutation- and LOS-reversal-symmetric.
3. Cubic matrices: all 16 reduced operators decompose *pointwise in x=q/k* in
   {j1..j5, 1/x^2, 1} with residual ~1e-13 (complete basis). Rationalized matrices agree
   exactly with GeneralBiasPk `M13.jl`. eta row: mu^2 (-2/7, 3/14, 1/14) = (-2, 3/2, 1/2)/7,
   confirming the audit's correction of the printed F.7 row (2, 3/2, -1/2)/7.
   Bare 1/x^2 coefficients: delta^(3): -1/6, eta^(3): +1/6 mu^2, u d eta^(2): -1/3 mu^4.
4. P22 tensor (SymPy, exact rationals): generated set = exactly the 23 DJS functions;
   reproduces the oracle's azimuthally averaged Z2^2 to 3e-13.
5. FFTLog (`folps/selection_matrices.py`): J basis reproduces FolpsD's independent closed
   forms M13_dd and M13_tt+M13_dt (EdS) to 1e-16; the P22 tensor + matrices reproduce
   FolpsD P22dd to 2e-14; the UV k^2 sigma^2 bookkeeping reproduces FolpsD's -61/105 and
   -9/5 sigma_psi^2 coefficients exactly.
6. Direct-quadrature reference `folps/selection_reference.py`; converged to ~1e-9 and equal to
   FolpsD P13dd (9e-7) and P22dd (1e-7) at k=0.02-0.1 grid points.

7. Bridge (prototype run, 5x6 (f, mu) grid): (A) and (B) J parts equal the reduced list
   (fit noise <= 1e-7); exact-map bare k^2 sigma_-1^2 coefficient = -1/6 [1 + f mu^2 (2+f)] Z1,
   zero for all quadratic/cubic selection coefficients. Kernel-level IR soft limit cancels as
   (q/k)^2 for (B) with all nine coordinates on; (A) leaves O(1) residual.
8. P22 two-class FFTLog vs direct quadrature, generic selection: 1e-5..1e-4 for
   0.02 < k < 0.19 at N=128; 4e-6 at k=0.41 with N=256.
9. Zero-selection gate: complete template loop vs FolpsD complete EdS EFT loop, IR off,
   AP=1, grid points: <=8.6e-4 of loop scale at N=128, <=1.6e-5 at N=256 (all mu).
10. API: legacy = frozen baseline bit for bit; explicit SelectionBias() = legacy bit for bit;
    invalid modes raise; JAX jit/jacfwd work; tests in tests/ (pytest).
11. Reconstructed audit script dev/selection_audit/audit_checks.py reproduces the audit
    numbers (eta response at k=0.4: 3.82899e-4 vs printed-F.7 6.1788e-4).

## Decisions


- D1 (P22 numerics). Evaluating UV-constant kernels at the matter tilt (-0.1) and restoring
  the dropped k^2 piece analytically is correct (restored coefficient matches direct
  quadrature to 5e-4) but ill-conditioned (restored piece ~5x the answer). Adopted instead:
  Galilean-safe advection kernels S'8 = S9+S5, S'9 = S10+S6 (UV-suppressed as (k/q)^2), with
  coefficients that are the rest-frame ones; soft-singular x soft-singular pairs at tilt
  -0.1 (UV-convergent), all other pairs at tilt -0.8 (continued integral drops exactly the
  zero-lag constant and keeps k^2 pieces). Generator asserts the pair classification exactly.
  No restoration terms; the 'continued' scheme option was removed from the API.
- D2 (comparisons): always compare at FolpsD output-grid points (linear interpolation of the
  tables produced spurious 5e-4..5e-3 "errors" early on).
- D3 (local tilt): -0.8 left a ghost-image k^2 error (-0.98 k^2 for K^2xK^2); measured tilt
  scan matched the (kmax/kmin)^(1+2b) estimate; adopted P22_TILT_LOCAL = -0.9.
- D4 (bare P13): production P13 uses the exact-map bare bridge (FolpsD-compatible, IR-safe),
  not the reduced (A) convention; sigma_0^2 constants dropped (b1, b_eta renormalization).
- D5 (noise basis): added optional EFTExtension.alphashot2_mu4 after finding k^2 mu^4 UV
  content in the selected P22 (zero without selection); flagged for theory review.
- D6 (legacy marginalization): gated helpers compute exact templates from the full model;
  legacy derivative helper's nmu full-range nodes vs model's 2*nmu symmetric nodes noted
  (3e-4 in ell=4 with IR), legacy behaviour left unchanged.

## Unresolved issues

- U1. FolpsD's default `nfftlog=128` gives ~3e-3 oscillating errors at k >~ 0.25 in P22
  (FolpsD's own P22dd vs converged direct quadrature: -3.2e-3 at k=0.28, +2.8e-3 at k=0.41);
  `nfftlog=256` reduces this to 1e-5 for both baseline and selection. Pre-existing baseline
  limitation; documented, not changed.
- U2. k^2 mu^4 UV-sensitive content of selected P22 (all selection coordinates): not in the
  DJS / GeneralBiasPk stochastic basis (P0eps, P2eps, P2eps_eta). Needs theory confirmation.
