# Local fNL in FolpsD — status

Branch `adematti-fnl`.  The user-facing description of the model and of the API is
`docs/api/PNG.rst`; this note records what was decided, what is deliberately left out, and what
still has to be checked.

## Implemented

- **`folps/png.py`** — `alpha_png` (both the ratio-of-spectra and the transfer-function routes,
  the same code as `desilike.theories.galaxy_clustering.png`), `primordial_pk` (the `cosmoprimo`
  convention), `bphi_universality`, `bfnl_loc`.
- **`alpha(k)` transported through the loop table**, as a new interpolated column carrying
  `alpha k^2` — flat at low `k`, where `alpha` itself diverges and where the signal is.  It is
  built in `NonLinearPowerSpectrumCalculator._initialize_png` from `alpha_png`, `pk_prim`, or
  `A_s`/`n_s` in the kwargs, and is zero when none of these is given.  It sits at the *end* of
  the interpolated block, so every existing index keeps its meaning (`table[1]` the linear power
  spectrum, `table[2]` `f(k)/f0`, `table[-2]` `sigma2w`); `table_alphak2` reads it back.
- **The bispectrum takes it as an `alphak2` keyword**, not as an extra row of `k_pkl_pklnw`.
  A row would have been free to plumb, but desilike already passes four rows
  (`[k, P_L, P_L^nw, f(k)]`) to `BispectrumCalculator`, which reads only three — the fourth
  would have been silently read as `alpha k^2` the moment `fnl` was non-zero.
- **Power spectrum** — `b1 -> b1 + b_phi f_NL alpha(k)` in the tree-level Kaiser term and in what
  multiplies it (`GTNS`, the NLO counterterm).  Direct path, monomial path, and both
  analytic-marginalisation paths.
- **Bispectrum** — the same substitution in the three legs, the `b_phi_delta` term in `Z2`, and
  the primordial bispectrum `Z1 Z1 Z1 x 2 f_NL [alpha_1 alpha_2 / alpha_3 P_1 P_2 + 2 perms]`,
  inside the same FoG bracket and the same AP prefactor as the gravitational terms.  Both
  bispectrum classes (`BispectrumCalculator`, `BispectrumCalculator_fk`), the Sugiyama and
  Scoccimarro multipoles, the monomial path, and the window convolution, which needed nothing
  beyond passing the keyword along.
- **`folps/test_png.py`** — `alpha` from the two routes against each other; `fNL = 0` bitwise
  identical to the Gaussian model in every entry point; the scale-dependent bias against the
  analytic tree level; the primordial term against the local template and against its squeezed
  limit `B/(P_1 P_3) -> 4 f_NL alpha(k_3)`; the monomial decompositions exact with PNG on; and,
  under JAX, `jit` and gradients with respect to `f_NL`.

Also on this branch: `RSDMultipolesPowerSpectrumCalculator.get_rsd_pkell`, deleted by accident in
d659a63 ("implementing monomials"), is restored.

## Conventions and choices

- `phi = phi_G + f_NL (phi_G^2 - <phi_G^2>)` with `phi = (3/5) zeta`, and
  `alpha = 1/M = sqrt(P_phi / P_L)`, exactly as in desilike, so `f_NL` means the same thing in a
  FOLPS fit and in a Kaiser one.
- The PNG parameters are appended to the bias vector *before* `X_FoG`, which stays last: `(fnl,
  bphi)` for the power spectrum, `(fnl, bphi, bphid)` for the bispectrum.  `split_png_pars`
  reads a vector of either length, so every existing call keeps working unchanged.  `fnl` and
  `bphi` are carried separately rather than as their product because the primordial bispectrum
  is proportional to `f_NL` alone.
- **The 1-loop bracket is left Gaussian.**  Its `b1` sits inside convolution integrals that
  `alpha(k)` cannot be pulled out of; doing it properly needs new FFTLog kernels with the `M(k)`
  weight, and is the standard omission.
- The `b1` multiplying `Bshot` in the bispectrum stochastic term keeps its Gaussian value; the
  legs, through `Z1eft`, do not.
- `A_s`/`n_s` inputs are converted with `k_pivot` in 1/Mpc while `k` is in h/Mpc, and the `h^3`
  of the `cosmoprimo` normalisation cancels against the one in `alpha_png`.  This is the single
  likeliest place for a constant `f_NL` bias, which is why `test_alpha` checks `primordial_pk`
  against `cosmoprimo` directly.

## Not done

- **Equilateral and orthogonal templates.**  Only the local one; they need a different `M(k)`
  weighting in the power spectrum and a non-separable shape in the bispectrum.
- **PNG stochastic terms** (a `<eps eps_phi>` piece going as `1/alpha`).
- **Cross-code validation.**  The internal checks pin the normalisation and the squeezed limit,
  but nothing has been compared against PBJ or Class-PT-PNG yet.  That is the main thing left
  before using this for a fit.
- **Squeezed-bin quadrature.**  `B_prim` grows like `1/alpha(k_3) ~ k_3^-2`, so the angular and
  `k` quadratures of the windowed bispectrum should be re-checked in the low-`k` rows rather
  than assumed from the Gaussian case.
- **The monomial basis grows** — 20 to 26 for the power spectrum, 45 to 116 for the bispectrum
  (44 with `c1`, `c2` fixed).  If that matters for an emulator, `fixed_bias` is the lever.
