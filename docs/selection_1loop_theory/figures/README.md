# Figures from the executed one-loop LOS-selection notebook

These PNG files are exported **unchanged** from the embedded outputs in `notebooks/selection_1loop_tutorial.ipynb` at Git blob `040e9e2829355fc5b1e11a79e8afeadb46ed18f3` on branch `feature/selection-1loop-eds`. They are **illustrative validation figures**, not DESI measurements. Reproduce by executing the notebook in its supported environment, then run `python scripts/export_selection_notebook_figures.py`.

1. `01_linear_spectrum.png` — Linear input spectrum (notebook cell 7).
2. `02_baseline_multipoles.png` — Standard FolpsD EFT multipoles (notebook cell 10).
3. `03_linear_selection.png` — Linear selection and tree bA (notebook cell 15).
4. `04_quadratic_kernel_check.png` — Quadratic kernel checks (notebook cell 18).
5. `05_selected_multipoles.png` — Selected versus unselected multipoles (notebook cell 23).
6. `06_power_spectrum_components.png` — Tree, loop, EFT and stochastic components (notebook cell 26).
7. `07_matched_tree_vs_one_loop.png` — Matched linear kernel: tree-only bA versus full one-loop (notebook cell 29).
8. `08_selection_responses.png` — Nine selection-operator responses (notebook cell 32).
9. `09_ir_resummation.png` — IR-resummed and unresummed multipoles (notebook cell 35).
10. `10_ap_projection.png` — Alcock-Paczynski projection (notebook cell 41).
11. `11_angular_convergence.png` — Multipole quadrature convergence (notebook cell 44).
12. `12_zero_selection_regression.png` — Zero-selection regression (notebook cell 46).
13. `13_jax_finite_difference.png` — JAX Jacobian finite-difference check (notebook cell 55).

The theoretical note uses selected figures from this directory. The notebook itself remains the authoritative record of plotting choices and numerical inputs.
