# ============================================================================================ #
#                     Local primordial non-Gaussianity (PNG) for FOLPS                         #
# ============================================================================================ #
r"""Scale-dependent bias and primordial bispectrum for local :math:`f_\mathrm{NL}`.

The primordial potential is taken as

.. math::  \phi = \phi_G + f_\mathrm{NL} (\phi_G^2 - \langle \phi_G^2 \rangle),

with :math:`\phi = \frac{3}{5}\zeta` the potential in the matter-dominated era, and the linear
density field is :math:`\delta_L(k, z) = M(k, z) \phi(k)`.  Everything here is written in terms of

.. math::  \alpha(k, z) = 1 / M(k, z) = \sqrt{P_\phi(k) / P_L(k, z)},

so that the local PNG scale-dependent bias is simply

.. math::  b_1 \rightarrow b_1 + b_\phi f_\mathrm{NL} \alpha(k, z).

Conventions follow ``desilike.theories.galaxy_clustering.png``, so that a fit can move between
the Kaiser and the FOLPS models without a rescaling of :math:`f_\mathrm{NL}`.
"""

try:
    from .folps import np
except ImportError:  # running from inside folps/, as the test scripts do
    from folps import np


delta_c = 1.686
"""Linear collapse threshold, for the universal-mass-function :math:`b_\\phi`."""

C_KMS = 299792.458
"""Speed of light in km/s."""


def primordial_pk(k, A_s=2.1e-9, n_s=0.96, k_pivot=0.05, h=0.6777, alpha_s=0., beta_s=0.):
    r"""Primordial spectrum of curvature perturbations :math:`\mathcal{P_R}(k)`.

    Same convention as :meth:`cosmoprimo.Primordial.pk_k`, so that its output can be passed
    straight to :func:`alpha_png` as *pk_prim*.

    Parameters
    ----------
    k : array
        Wavenumbers in :math:`h/\mathrm{Mpc}`.
    A_s : float
        Scalar amplitude at *k_pivot*.
    n_s : float
        Scalar tilt.
    k_pivot : float
        Pivot scale in :math:`1/\mathrm{Mpc}` (note the units: *k* is in :math:`h/\mathrm{Mpc}`).
    h : float
        Reduced Hubble rate, which sets both the pivot conversion and the :math:`(\mathrm{Mpc}/h)^3`
        normalisation.
    alpha_s, beta_s : float
        Running and running of the running.

    Returns
    -------
    pk_prim : array, in :math:`(\mathrm{Mpc}/h)^3`.
    """
    kp = k_pivot / h
    lnkkp = np.log(k / kp)
    return h**3 * A_s * (k / kp)**(n_s - 1. + alpha_s * lnkkp / 2. + beta_s * lnkkp**2 / 6.)


def alpha_png(k, pk_dd, pk_prim, h, method='prim', Omega0_m=None,
              growth_factor_z=None, growth_factor_znorm=None, znorm=10.):
    r"""Return :math:`\alpha(k, z) = \sqrt{P_\phi(k) / P_L(k, z)}`, the inverse of the
    potential-to-density transfer :math:`M(k, z)`.

    Parameters
    ----------
    k : array, shape (nk,)
        Wavenumbers in :math:`h/\mathrm{Mpc}`.  For ``method = 'transfer'``, ``k[0]`` must be
        small enough (:math:`\sim 10^{-4}\, h/\mathrm{Mpc}`) that :math:`T(k_0) \simeq 1`.
    pk_dd : array, shape (nk,)
        Linear power spectrum at the redshift of interest, in :math:`(\mathrm{Mpc}/h)^3`.
    pk_prim : array, shape (nk,)
        Primordial spectrum of curvature perturbations on the same grid, e.g. from
        :func:`primordial_pk`.
    h : float
        Reduced Hubble rate.
    method : str
        ``'prim'``, the ratio above, which needs nothing but *pk_dd* and *pk_prim* and is
        therefore free of any growth-normalisation convention; or ``'transfer'``,
        :math:`\alpha = 3 \Omega_m H_0^2 / (2 c^2 k^2 T(k) D(z))` with :math:`T` normalised to 1
        at ``k[0]`` and :math:`D` normalised in the matter-dominated era (eq. 2.3 of
        arXiv:1904.08859).  The two agree when the inputs are mutually consistent; 'transfer' is
        mostly useful as a cross-check.
    Omega0_m : float
        Matter density parameter today; ``'transfer'`` only.
    growth_factor_z, growth_factor_znorm : float
        Linear growth factors at the redshift of *pk_dd* and at *znorm*, in any common
        normalisation; ``'transfer'`` only.
    znorm : float
        Redshift at which the growth factor is matched to matter domination, :math:`D = 1/(1+z)`.

    Returns
    -------
    alpha : array, shape (nk,)
    """
    if method == 'prim':
        pk_phi = 9. / 25. * 2. * np.pi**2 / k**3 * pk_prim / h**3
        return np.sqrt(pk_phi / pk_dd)
    if method == 'transfer':
        if Omega0_m is None or growth_factor_z is None or growth_factor_znorm is None:
            raise ValueError("method='transfer' requires Omega0_m, growth_factor_z and growth_factor_znorm")
        growth_ratio = growth_factor_z / (growth_factor_znorm * (1. + znorm))
        tk = np.sqrt(pk_dd / pk_prim / k / (pk_dd[0] / pk_prim[0] / k[0]))
        return 3. * Omega0_m * 100.**2 / (2. * C_KMS**2 * k**2 * tk * growth_ratio)
    raise ValueError(f"method must be 'prim' or 'transfer'; got {method!r}")


def bphi_universality(b1, p=1.):
    r"""Return :math:`b_\phi = 2 \delta_c (b_1 - p)`, the universal-mass-function relation.

    *p* is 1 for a tracer selected on halo mass alone, and closer to 1.6 for a merger-selected
    one; it is an assumption, not a definition, which is why the FOLPS entry points take
    :math:`b_\phi f_\mathrm{NL}` itself.
    """
    return 2. * delta_c * (b1 - p)


def bfnl_loc(fnl, b1=None, p=1., bphi=None):
    r"""Return :math:`b_{f_\mathrm{NL}} = b_\phi f_\mathrm{NL}`, the combination the power
    spectrum depends on.

    Either *bphi* is given, or it is built from *b1* and *p* with :func:`bphi_universality`.
    """
    if bphi is None:
        if b1 is None:
            raise ValueError("provide either bphi, or b1 (and optionally p)")
        bphi = bphi_universality(b1, p=p)
    return bphi * fnl
