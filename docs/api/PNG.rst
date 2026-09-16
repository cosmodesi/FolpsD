Primordial non-Gaussianity
==========================

``folps.png`` holds the local :math:`f_\mathrm{NL}` model: the scale-dependent bias of the power
spectrum and the primordial bispectrum.  Conventions follow
``desilike.theories.galaxy_clustering.png``, so :math:`f_\mathrm{NL}` means the same thing in a
FOLPS fit and in a Kaiser one.

Model
-----

With :math:`\phi = \phi_G + f_\mathrm{NL}(\phi_G^2 - \langle\phi_G^2\rangle)` the potential in the
matter-dominated era and :math:`\delta_L = M(k, z)\phi`, everything is written in terms of
:math:`\alpha = 1/M = \sqrt{P_\phi / P_L}`:

- the linear response of the tracer becomes :math:`b_1 + b_\phi f_\mathrm{NL}\, \alpha(k)`, which
  enters the tree-level Kaiser term and the three bispectrum legs (the 1-loop bracket is left
  Gaussian: its :math:`b_1` sits inside convolution integrals that :math:`\alpha(k)` cannot be
  pulled out of);
- the second-order bias adds :math:`b_{\phi\delta} f_\mathrm{NL} (\alpha_i + \alpha_j)/2` to
  :math:`Z_2`;
- the primordial bispectrum adds
  :math:`Z_1(k_1) Z_1(k_2) Z_1(k_3) \times 2 f_\mathrm{NL} [\alpha_1\alpha_2/\alpha_3 P(k_1)P(k_2)
  + 2\ \mathrm{perms}]`.

Usage
-----

Pass the primordial spectrum (or ``A_s`` and ``n_s``) when building the loop table, then extend
the bias vector with ``(fnl, bphi)`` for the power spectrum and ``(fnl, bphi, bphid)`` for the
bispectrum, inserted just before ``X_FoG``:

.. code-block:: python

   from folps import (NonLinearPowerSpectrumCalculator, RSDMultipolesPowerSpectrumCalculator,
                      BispectrumCalculator, table_alphak2, bphi_universality)

   kwargs = dict(z=0.5, Omega_m=0.31, h=0.6777, fnu=0., A_s=2.1e-9, n_s=0.9649)
   nonlinear = NonLinearPowerSpectrumCalculator(mmatrices=mmatrices, kernels='fk', **kwargs)
   table, table_now = nonlinear.calculate_loop_table(k=k, pklin=pklin, **kwargs)

   bphi = bphi_universality(b1, p=1.)
   pars = [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, sn0, sn2, PshotP, fnl, bphi, X_FoG]
   pkell = RSDMultipolesPowerSpectrumCalculator(model='FOLPSD').get_rsd_pkell(
       kobs, qpar, qper, pars, table, table_now, damping='lor')

   bpars = [b1, b2, bs, c1, c2, Bshot, Pshot, fnl, bphi, bphid, X_FoG]
   bkell = BispectrumCalculator(model='FOLPSD').Sugiyama_Bell(
       f=nonlinear.f0, bpars=bpars, k_pkl_pklnw=[table[0], table[1], table_now[1]],
       alphak2=table_alphak2(table),
       k1k2pairs=k1k2pairs, qpar=qpar, qper=qper, damping='lor')

Bias vectors without the PNG entries keep working and give the Gaussian model bitwise.  The
bispectrum entry points switch their PNG terms on from the ``alphak2`` keyword -- a keyword
rather than an extra row of ``k_pkl_pklnw``, so that a caller passing the four rows
``[k, P_L, P_L^{nw}, f(k)]`` to :class:`BispectrumCalculator` (which reads only the first three)
cannot have the fourth mistaken for :math:`\alpha k^2`.  :func:`table_alphak2` reads the column
out of a loop table, interpolated or not.

The transfer :math:`\alpha(k)` is read from the ``kwargs`` of the loop table, in order of
decreasing directness: ``alpha_png`` (an array on the input ``k``), ``pk_prim`` (likewise, e.g.
``cosmoprimo``'s ``primordial.pk_k``), or ``A_s`` and ``n_s`` (with ``k_pivot`` in
:math:`1/\mathrm{Mpc}` and ``h``).  ``png_method='transfer'`` selects the transfer-function route
of eq. 2.3 of arXiv:1904.08859 instead of the default ratio of spectra; the two agree to the 1%
radiation correction of the matter-dominated normalisation at :math:`z = 10`.

Emulation
---------

Both monomial decompositions carry the PNG parameters symbolically, so
``get_rsd_pkmu_monomials_tables`` and ``Sugiyama_Bell_monomials`` remain exact.  The basis grows
-- 20 to 26 monomials for the power spectrum, 45 to 116 for the bispectrum, where the
counterterms dominate the growth (44 with ``c1`` and ``c2`` fixed).  Pass
``fixed_bias={'fnl': 0.}`` to collapse the PNG monomials for a Gaussian run.

Functions
---------

.. py:function:: alpha_png(k, pk_dd, pk_prim, h, method='prim', Omega0_m=None, growth_factor_z=None, growth_factor_znorm=None, znorm=10.)

   :math:`\alpha(k, z) = \sqrt{P_\phi(k)/P_L(k, z)}`, from the ratio of spectra (``'prim'``) or
   from the transfer function (``'transfer'``).

.. py:function:: primordial_pk(k, A_s=2.1e-9, n_s=0.96, k_pivot=0.05, h=0.6777, alpha_s=0., beta_s=0.)

   Primordial spectrum of curvature perturbations, in the ``cosmoprimo`` convention.

.. py:function:: bphi_universality(b1, p=1.)

   :math:`b_\phi = 2\delta_c (b_1 - p)`, the universal-mass-function relation.

.. py:function:: bfnl_loc(fnl, b1=None, p=1., bphi=None)

   :math:`b_\phi f_\mathrm{NL}`, from ``bphi`` or from ``b1`` and ``p``.

.. py:function:: table_alphak2(table)

   The :math:`\alpha(k) k^2` column of a loop table (``folps.folps``), to be passed as the
   ``alphak2`` keyword of the bispectrum entry points.  It is the last entry of the interpolated
   block, so adding it left every index before it -- ``table[1]`` the linear power spectrum,
   ``table[2]`` :math:`f(k)/f_0` -- untouched.
