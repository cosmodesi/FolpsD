"""Tests for the local-PNG (fNL) terms of the power spectrum and bispectrum.

Run with either backend::

    python folps/test_png.py                     # numpy
    FOLPS_BACKEND=jax python folps/test_png.py   # jax (adds the jit / grad checks)
"""

import os

backend = os.environ.get('FOLPS_BACKEND', 'numpy').lower()
if backend == 'jax':   # before folps imports jax.numpy
    import jax
    jax.config.update('jax_enable_x64', True)

import numpy as np

from folps import (BispectrumCalculator, MatrixCalculator, NonLinearPowerSpectrumCalculator,
                   RSDMultipolesPowerSpectrumCalculator, table_alphak2)
try:
    from folps.png import alpha_png, bphi_universality, primordial_pk
except ModuleNotFoundError:  # script run from inside folps/, as the other test scripts are
    from png import alpha_png, bphi_universality, primordial_pk

kwargs = dict(z=0.5, Omega_m=0.31, h=0.6777, fnu=0., A_s=2.1e-9, n_s=0.9649)
bphi = bphi_universality(2.)
# [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, PshotP] and
# [b1, b2, bs, c1, c2, Bshot, Pshot]; the PNG entries and X_FoG are appended by each test.
pars_pk = [2., 0.5, 0.3, 0.1, 3., -28.9, 0., 0., 0.08, -8.1, 1e4]
pars_bk = [2., 0.5, 0.3, 0.01, 0.01, 1., 0.5]
kobs = np.linspace(0.01, 0.2, 20)
kev = np.linspace(0.02, 0.2, 10)
k1k2pairs = np.vstack([kev, kev]).T

_cache = {}


def setup():
    """Loop tables for the reference cosmology, computed once."""
    if 'tables' not in _cache:
        path = os.path.join(os.path.dirname(__file__), 'inputpkT.txt')
        k, pk = np.loadtxt(path, unpack=True)
        matrices = MatrixCalculator(A_full=True, save_dir=os.path.join(os.path.dirname(__file__), 'output_matrices'))
        nonlinear = NonLinearPowerSpectrumCalculator(mmatrices=matrices.get_mmatrices(), kernels='fk', **kwargs)
        table, table_now = nonlinear.calculate_loop_table(k=np.asarray(k), pklin=np.asarray(pk), **kwargs)
        _cache['tables'] = (nonlinear, table, table_now)
    return _cache['tables']


def pkell(pars, table, table_now, **kw):
    # Returned in the backend's own array type, not as numpy: these helpers are called from
    # inside jit / grad in test_jax, where np.asarray on a tracer raises.
    multipoles = RSDMultipolesPowerSpectrumCalculator(model='FOLPSD')
    return multipoles.get_rsd_pkell(kobs, 1., 1., pars, table, table_now, damping='lor', **kw)


def k_pkl_pklnw(table, table_now):
    return np.array([np.asarray(table[0]), np.asarray(table[1]), np.asarray(table_now[1])])


def bkell(bpars, alphak2=None, monomials=False, **kw):
    nonlinear, table, table_now = setup()
    bispectrum = BispectrumCalculator(model='FOLPSD')
    options = dict(k_pkl_pklnw=k_pkl_pklnw(table, table_now), k1k2pairs=k1k2pairs,
                   qpar=1., qper=1., alphak2=alphak2,
                   precision=[6, 8, 8], multipoles=['B000', 'B202'], renormalize=True,
                   interpolation_method='linear', **kw)
    if monomials:
        tables = bispectrum.Sugiyama_Bell_monomials(f=nonlinear.f0, **options)
        return bispectrum.Sugiyama_Bell_from_monomials(tables, bpars, damping='lor')
    return bispectrum.Sugiyama_Bell(f=nonlinear.f0, bpars=bpars, damping='lor',
                                    bias_scheme='folps', **options)


def test_alpha():
    """alpha(k) from the primordial spectrum against the transfer-function route."""
    from cosmoprimo.fiducial import DESI

    cosmo, z = DESI(), 1.
    k = np.geomspace(1e-4, 1., 200)
    pk_dd = cosmo.get_fourier().pk_interpolator(of='delta_m')(k, z=z)
    pk_prim = cosmo.get_primordial().pk_k(k)
    assert np.allclose(pk_prim, primordial_pk(k, A_s=cosmo['A_s'], n_s=cosmo['n_s'],
                                              k_pivot=cosmo['k_pivot'], h=cosmo['h']), rtol=1e-10)
    alpha = alpha_png(k, pk_dd, pk_prim, cosmo['h'], method='prim')
    alpha_transfer = alpha_png(k, pk_dd, pk_prim, cosmo['h'], method='transfer',
                               Omega0_m=cosmo['Omega_m'], growth_factor_z=cosmo.growth_factor(z),
                               growth_factor_znorm=cosmo.growth_factor(10.))
    ratio = alpha_transfer / alpha
    # The two differ by the matter-domination approximation of D(z_norm = 10) alone: a constant,
    # at the 1% radiation correction, with no scale dependence left over.
    assert np.allclose(ratio, ratio[0], rtol=1e-6), np.ptp(ratio / ratio[0])
    assert abs(ratio[0] - 1.) < 0.01, ratio[0]
    # alpha ~ 1 / (k^2 T(k)) grows towards low k, and is O(0.1) around k = 1e-3 h/Mpc at z = 1.
    assert np.all(np.diff(alpha) < 0.)
    print('test_alpha: prim vs transfer, constant ratio {:.5f}'.format(ratio[0]))


def test_gaussian():
    """fNL = 0 reproduces the Gaussian model bitwise, in every entry point."""
    nonlinear, table, table_now = setup()
    alphak2 = table_alphak2(table)
    assert nonlinear.has_png
    assert np.array_equal(np.asarray(pkell(pars_pk + [0., bphi, 1.], table, table_now)),
                          np.asarray(pkell(pars_pk + [1.], table, table_now)))
    # the bispectrum, with alphak2 given and not
    reference = bkell(pars_bk + [1.])
    for bpars, alpha in [(pars_bk + [0., bphi, 1., 1.], alphak2), (pars_bk + [1.], None)]:
        assert np.array_equal(np.asarray(bkell(bpars, alpha)), np.asarray(reference))
    # and the monomial paths
    assert np.allclose(bkell(pars_bk + [0., bphi, 1., 0.], alphak2, monomials=True),
                       bkell(pars_bk + [0.], monomials=True), rtol=1e-12)
    print('test_gaussian: fNL = 0 identical to the Gaussian model')


def test_tree_level():
    """The scale-dependent bias, against (b1 + b_phi fNL alpha(k) + f(k) mu^2)^2 P_L."""
    nonlinear, table, table_now = setup()

    def tree_only(table, nowiggle=False):
        """Zero every loop column, leaving the tree-level term alone (EFT, so no damping;
        sigma2w = 0 kills GTNS and the NLO counterterm with it).  The trailing entries are
        [alpha k^2, sigma2w, (sigma2_NW, delta_sigma2_NW,) f0]."""
        ntrail = 5 if nowiggle else 3
        table, zero = list(table), np.zeros_like(np.asarray(table[1]))
        for i in range(3, len(table) - ntrail):
            table[i] = zero
        table[1 - ntrail] = 0.
        return tuple(table)

    multipoles = RSDMultipolesPowerSpectrumCalculator(model='EFT')
    fnl, mu = 100., np.linspace(0., 1., 7)[:, None].T
    k = kobs[:, None]
    pars = [2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., fnl, bphi, 0.]
    pkmu = multipoles.get_rsd_pkmu(k, mu, multipoles.set_bias_scheme(pars), tree_only(table),
                                   tree_only(table_now, nowiggle=True), IR_resummation=False,
                                   damping=None)
    # The reference reads the same interpolated columns -- what is under test is the assembly of
    # b1 + b_phi fNL alpha(k), not the interpolation, which differs between the two backends.
    interpolated = multipoles.interp_table(k, table, True)
    pkl, fk_over_f0 = interpolated[0], interpolated[1]
    alpha = table_alphak2(interpolated) / k**2
    assert np.allclose(pkmu, (2. + fnl * bphi * alpha + fk_over_f0 * nonlinear.f0 * mu**2)**2 * pkl,
                       rtol=1e-12)
    print('test_tree_level: scale-dependent bias exact at tree level')


def test_primordial_bispectrum():
    """The primordial term against the local template, and its squeezed limit."""
    nonlinear, table, table_now = setup()
    kpp = k_pkl_pklnw(table, table_now)
    k_, pkl_, alphak2_ = kpp[0], kpp[1], np.asarray(table_alphak2(table))
    bispectrum = BispectrumCalculator(model='FOLPSD')
    fnl = 100.
    # A real-space, unbiased, counterterm-free tracer, so that Z1 = 1 and only F2 and the
    # primordial term are left; no IR resummation, no damping, no AP.
    triangle = dict(f=0., sigma2v=0., Sigma2=0., deltaSigma2=0., qpar=1., qperp=1.,
                    k_pkl_pklnw=kpp, alphak2=alphak2_, damping=None,
                    interpolation_method='cubic')
    gaussian = [1., 0., 0., 0., 0., 0., 0.]

    def bk(k1, k2, x12, fnl):
        return np.asarray(bispectrum.bispectrum(k1, k2, x12, mu1=0., phi=0.,
                                                bpars=gaussian + [fnl, 0., 0., 0.], **triangle))

    # The reference uses the same interpolator as the code -- what is under test is the
    # template, not the interpolation, which differs between the two backends.
    interpolate = lambda values, k: float(np.asarray(bispectrum.interpolation_b(k, k_, values, method='cubic')))
    alpha_of = lambda k: interpolate(alphak2_, k) / k**2
    pk_of = lambda k: interpolate(pkl_, k)

    # 1. the local template, B_prim = M1 M2 M3 x 2 fNL [P_phi(k1) P_phi(k2) + 2 perms], built
    #    from alpha = 1 / M and P_phi = alpha^2 P_L without the ratio rearrangement of the code.
    k1, k2, x12 = 0.1, 0.08, -0.3
    k3 = (k1**2 + k2**2 + 2. * k1 * k2 * x12)**0.5
    alpha, pk = (np.array([f(k) for k in (k1, k2, k3)]) for f in (alpha_of, pk_of))
    pk_phi = alpha**2 * pk
    expected = 2. * fnl * np.prod(1. / alpha) * (pk_phi[0] * pk_phi[1] + pk_phi[1] * pk_phi[2]
                                                 + pk_phi[2] * pk_phi[0])
    assert np.allclose(bk(k1, k2, x12, fnl) - bk(k1, k2, x12, 0.), expected, rtol=1e-10)

    # 2. squeezed limit, B_prim / [P(k1) P(k3)] -> 4 fNL alpha(k3) as k3 -> 0.
    k1 = 0.15
    for k3 in [2e-2, 1e-2, 5e-3]:
        x12 = (k3**2 - 2. * k1**2) / (2. * k1**2)   # k2 = k1, |k1 + k2| = k3
        squeezed = (bk(k1, k1, x12, fnl) - bk(k1, k1, x12, 0.)) / (pk_of(k1) * pk_of(k3))
        ratio = squeezed / (4. * fnl * alpha_of(k3))
        assert abs(ratio - 1.) < 0.05 * (k3 / 2e-2), (k3, ratio)
    print('test_primordial_bispectrum: local template and squeezed limit, ratio {:.4f} at k3 = 5e-3'.format(ratio))


def test_monomials():
    """The monomial decomposition is exact with PNG on, and its basis is optional."""
    nonlinear, table, table_now = setup()
    alphak2 = table_alphak2(table)
    bpars = pars_bk + [100., bphi, 1., 0.]      # X_FoG = 0: the collocation is then exact
    assert np.allclose(bkell(bpars, alphak2, monomials=True), bkell(bpars, alphak2), rtol=1e-10)

    multipoles = RSDMultipolesPowerSpectrumCalculator(model='FOLPSD')
    pars = pars_pk + [100., bphi, 0.]
    tables = multipoles.get_rsd_pkmu_monomials_tables(kobs, 1., 1., table, table_now, nmu=6, ells=(0, 2, 4))
    monomials = np.asarray(multipoles.get_rsd_pkell_from_monomials(tables, multipoles.set_bias_scheme(pars)))
    assert np.allclose(monomials, pkell(pars, table, table_now), rtol=1e-10)

    # fixing fnl collapses the PNG monomials, back to the Gaussian basis.
    gaussian = multipoles.get_rsd_pkmu_monomials_tables(kobs, 1., 1., table, table_now, nmu=6,
                                                        ells=(0, 2, 4), fixed_bias={'fnl': 0.})
    assert len(gaussian['monomials']) < len(tables['monomials'])

    def bk_basis(alphak2):
        bispectrum = BispectrumCalculator(model='FOLPSD')
        return len(bispectrum.Sugiyama_Bell_monomials(
            f=nonlinear.f0, k_pkl_pklnw=k_pkl_pklnw(table, table_now), alphak2=alphak2,
            k1k2pairs=k1k2pairs, qpar=1., qper=1., precision=[4, 6, 6], multipoles=['B000'],
            interpolation_method='linear')['monomials'])

    print('test_monomials: exact; pk basis {:d} -> {:d} monomials with PNG, bk {:d} -> {:d}'.format(
        len(gaussian['monomials']), len(tables['monomials']), bk_basis(None), bk_basis(table_alphak2(table))))


def test_marginalization():
    """The analytic-marginalisation paths carry the scale-dependent bias too."""
    from folps import get_rsd_pkell_marg_const, get_rsd_pkell_marg_derivatives

    nonlinear, table, table_now = setup()
    alphas = [3., -28.9, 0., 0.08, -8.1]   # alpha0, alpha2, alpha4, sn0, sn2
    fnl = 100.
    pars = [2., 0.5, 0.3, 0.1] + alphas[:3] + [0.] + alphas[3:] + [1e4, fnl, bphi, 1.]
    # nmu = 12 here against the default nmu = 6 of get_rsd_pkell: the latter symmetrises its
    # nodes onto [0, 1], so the two quadratures then coincide and the comparison is exact.
    options = dict(kobs=kobs, qpar=1., qper=1., pars=pars, table=table, table_now=table_now,
                   bias_scheme='folps', damping='lor', model='EFT', nmu=12)
    constant = np.asarray(get_rsd_pkell_marg_const(**options))
    derivatives = np.asarray(get_rsd_pkell_marg_derivatives(**options))
    # P_ell = P_ell(alpha_i = 0) + sum_i alpha_i dP_ell/dalpha_i, exactly, PNG included.
    marginalized = constant + np.einsum('i,lik->lk', np.array(alphas), derivatives)
    multipoles = RSDMultipolesPowerSpectrumCalculator(model='EFT')
    direct = np.asarray(multipoles.get_rsd_pkell(kobs, 1., 1., pars, table, table_now, damping='lor'))
    assert np.allclose(marginalized, direct, rtol=1e-10)
    print('test_marginalization: constant + derivatives reproduce the full multipoles')


def test_jax():
    """Under the JAX backend: jit, and a finite gradient with respect to fNL."""
    import jax
    nonlinear, table, table_now = setup()
    alphak2 = table_alphak2(table)

    def pk_of_fnl(fnl):
        return pkell(pars_pk + [fnl, bphi, 1.], table, table_now)[0].sum()

    def bk_of_fnl(fnl):
        return bkell(pars_bk + [fnl, bphi, 1., 1.], alphak2)[0].sum()

    for name, func in [('pk', pk_of_fnl), ('bk', bk_of_fnl)]:
        derivative = jax.jit(jax.grad(func))(0.)
        assert np.isfinite(derivative), name
        finite_difference = (func(1e-2) - func(-1e-2)) / 2e-2
        assert np.allclose(derivative, finite_difference, rtol=1e-4), (name, derivative, finite_difference)
    print('test_jax: gradients with respect to fNL match finite differences')


if __name__ == '__main__':
    tests = [test_alpha, test_gaussian, test_tree_level, test_primordial_bispectrum,
             test_monomials, test_marginalization]
    if backend == 'jax':
        tests.append(test_jax)
    for test in tests:
        test()
    print('All PNG tests passed ({} backend).'.format(backend))
