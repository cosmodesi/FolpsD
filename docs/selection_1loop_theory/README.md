# Full one-loop LOS selection: theory note

- [Read the LaTeX source](selection_1loop_theory.tex).
- [Read the PDF containing nine original notebook figures](selection_1loop_theory.pdf).
- [Browse all 13 exported notebook figures](figures/).
- [Read the derivation and implementation equations](../selection_1loop_equations.md).

The PDF is compiled with LaTeX in the GitHub Actions workflow
\`.github/workflows/selection-theory-pdf.yml\` whenever the source changes.
The build will fail if any figure is missing. The workflow embeds nine
original plots and commits the resulting PDF to this branch. It also
retains a downloadable build artifact.

**Scope:** theoretical and numerical documentation for the implemented
EdS, single-fluid, auto-power-spectrum one-loop line-of-sight-selection
extension to FolpsD EFT, with perturbative GTNS on and phenomenological
FoG damping off. The figures are generated from the demonstration
notebook, **not DESI observational measurements**.
