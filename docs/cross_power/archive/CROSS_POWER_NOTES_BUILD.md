# Cross-Power Notes Build

## Compilation

Run from `docs/cross_power/`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error folps_cross_power_notes.tex
```

The final PDF is:

```text
docs/cross_power/folps_cross_power_notes.pdf
```

Page count from `pdfinfo`: 14.

## Figures Included

- `figures/cross_vs_geometric_mean_pkmu.pdf`
- `figures/cross_geometric_mean_bias_dependence.pdf`
- `figures/cross_ir_resummation.pdf`
- `figures/cross_ap_remapping.pdf`
- `figures/cross_implementation_residuals.pdf`
- `figures/cross_power_multipoles.pdf`
- `figures/cross_nonlinear_bias_response.pdf`
- `figures/cross_nuisance_response.pdf`

The nuisance-response figure was revised to use separate rows for standard EFT
counterterms, the NLO `ctilde` response, and stochastic terms. The filename was
kept stable. The corresponding CSV now includes a `group` column.

## References Included

- Taruya, Nishimichi, and Saito, arXiv:1006.0699.
- Aviles, Banerjee, Niz, and Slepian, arXiv:2106.13771.
- Noriega, Aviles, Fromenteau, and Vargas-Magana, arXiv:2208.02791.
- Bansal et al., arXiv:2604.08895.

The bibliography is stored in `docs/cross_power/references.bib`.

## Warnings

The final `latexmk` build completed without errors. Before cleaning auxiliary
files, this log check returned no matches:

```bash
rg -n "Warning|Overfull|Underfull|undefined|Citation" folps_cross_power_notes.log
```

No missing citations, unresolved references, missing figures, overfull boxes, or
underfull boxes remained in the final build log.

## Visual Validation

Rendered with:

```bash
pdftoppm -png -r 140 folps_cross_power_notes.pdf <temporary-render-dir>/page
```

Inspected pages: title/TOC, equation-heavy pages, the A-row mapping table, all
figure pages, bibliography, and final page. No clipped equations, unreadable
tables, oversized figures, broken references, empty pages, or detached captions
were found.

## Files Changed

- Created `docs/cross_power/folps_cross_power_notes.tex`.
- Created `docs/cross_power/folps_cross_power_notes.pdf`.
- Created `docs/cross_power/references.bib`.
- Created `docs/cross_power/CROSS_POWER_NOTES_BUILD.md`.
- Updated `docs/cross_power/make_cross_power_figures.py`.
- Updated `docs/cross_power/CROSS_POWER_FIGURES.md`.
- Updated `docs/cross_power/CROSS_POWER_LATEX_FIGURE_SNIPPETS.tex`.
- Regenerated the figure and CSV table set under `docs/cross_power/figures/`
  and `docs/cross_power/tables/`.

## Unresolved Scientific Questions

- Effective-redshift treatment and redshift averaging over a realistic overlap
  interval.
- Realistic LRG3 x ELG1 parameter choices and data-vector definitions.
- Cross stochasticity, positivity, and possible covariance-matrix
  parametrizations.
- Field-level counterterm parametrizations relating auto and cross EFT terms.
- Window convolution, covariance, and likelihood/desilike integration.
- Velocity bias, odd multipoles, wide-angle terms, and relativistic corrections.
- FolpsD-compatible phenomenological cross damping.
- Bispectrum cross-correlations and joint two-tracer power/bispectrum analyses.
