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

## Decisions

## Unresolved issues
