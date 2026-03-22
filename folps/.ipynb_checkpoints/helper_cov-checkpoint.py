from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from pypower import PowerSpectrumMultipoles, BaseMatrix
from desilike.observables import ObservableCovariance
def construct_bao_cross_power(tracer,kmax,no_syst=False):
#     tracer_to_zrange_map = {
#     'BGS_BRIGHT-21.5': [(0.1, 0.4)],
#     'LRG': [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)],
#     'ELG_LOPnotqso': [(0.8, 1.1), (1.1, 1.6)],
#     'QSO': [(0.8, 2.1)]
# }
    simplified_tracer_map = {
    'BGS': ('BGS_BRIGHT-21.5', (0.1, 0.4)),
    'LRG1': ('LRG', (0.4, 0.6)),
    'LRG2': ('LRG', (0.6, 0.8)),
    'LRG3': ('LRG', (0.8, 1.1)),
    'ELG': ('ELG_LOPnotqso', (1.1, 1.6)),  # Only the second (higher-z) bin as requested
    'QSO': ('QSO', (0.8, 2.1))
    }
    tracer,zrange = simplified_tracer_map[tracer]
    # tracer, zrange = 'ELG_LOPnotqso', (1.1, 1.6)
    base_dir = Path('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/')
    # Statistical covariance matrix
    covstat_fn = base_dir / f'cov_2pt/ezmock/v1/rotated/covariance_rotated_marg-no_power+bao-recon_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_default_FKP_lin_thetacut0.05.npy'
    # Systematic covariance matrices: rotation, HOD, photometric systematics
    covsyst_fns = {syst: base_dir / f'cov_2pt/syst/v1.5/rotated/covariance_syst-{syst}_rotated_power_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_default_FKP_lin_thetacut0.05.npy' for syst in ['rotation', 'hod', 'photo']}
     
    klim = (0.02, kmax, 0.005)
    ells = (0, 2)
    
    covariance = ObservableCovariance.load(covstat_fn)
    # Note: how to get the rotated covariance?
    def get_data_rotation(tracer, zrange):
        import glob
        from desi_y3_files import WindowRotation
        rotation_dir = Path('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/2pt/pk/rotated')
        rotation_fn = glob.glob(str(rotation_dir / 'rotation_wmatrix_smooth_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_default_FKP_lin_nran*_cellsize6_boxsize*_thetacut0.05.npy').format(tracer=tracer, zrange=zrange))
        assert len(rotation_fn) == 1
        return WindowRotation.load(rotation_fn[0])
    
    def rotate_cov(rotation, cov):
        index = cov._index(observables='power', xlim=(0., 0.4))
        mmatrix = np.eye(*cov.shape)
        mmatrix[np.ix_(index, index)] = rotation.mmatrix[0]
        return mmatrix.dot(cov.view()).dot(mmatrix.T)
    
    # Rotate covariance matrix
    covariance_not_rotated = ObservableCovariance.load(base_dir / f'cov_2pt/ezmock/v1/covariance_power+bao-recon_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_default_FKP_lin_thetacut0.05.npy')
    data_rotation = get_data_rotation(tracer, zrange)
    covariance_rotated = covariance_not_rotated.clone(value=rotate_cov(data_rotation, covariance_not_rotated))
    assert np.allclose(covariance_rotated.view(), covariance.view())
    
    # Apply k-cut to 'power' part, select ells=(0, 2)
    covariance = covariance.select(xlim=klim[:-1], projs=list(ells), observables='power', select_projs=True)
    print(covariance)
    # Rescaling factor, see KP3 paper, Table 6
    factor = {('BGS_BRIGHT-21.5', (0.1, 0.4)): 1.39, ('LRG', (0.4, 0.6)): 1.15, ('LRG', (0.6, 0.8)): 1.15, ('LRG', (0.8, 1.1)): 1.22, ('ELG_LOPnotqso', (0.8, 1.1)): 1.25, ('ELG_LOPnotqso', (1.1, 1.6)): 1.29, ('QSO', (0.8, 2.1)): 1.11}[tracer, zrange]
    # Then Hartlap 2007 (https://arxiv.org/abs/astro-ph/0608064) and Percival 2014 (https://arxiv.org/abs/1312.4841) factors, for 7 = (3 bias + 2 counterterms + 2 stochastic terms) fitted parameters
    factor *= covariance.percival2014_factor(7) / covariance.hartlap2007_factor()
    # Rescale covariance
    covariance = covariance.clone(value=factor * covariance.view())
    
    # Then, let's plug in BAO alpha posteriors from KP4
    forfit_dir = Path('/global/cfs/cdirs/desi//survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/forfit_2pt/')
    bao_fn = forfit_dir / f'covariance_stat_bao-recon_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy'
    covariance_bao = ObservableCovariance.load(bao_fn)
    cov_bao = covariance_bao.view(projs=covariance.observables('bao-recon').projs)  # select qpar, qper covariance
    std_bao = np.diag(cov_bao)**0.5
    corr_bao = cov_bao / (std_bao[..., None] * std_bao)
    corr_full = covariance.corrcoef()
    std_full = covariance.std()
    index_bao = covariance._index(observables='bao-recon', concatenate=True)
    assert np.allclose(index_bao, covariance.observables('power').size + np.arange(len(index_bao)))
    # Replace correlation and sigma of the BAO part of the EZmock covariance matrix by the KP4 BAO posterior
    corr_full[np.ix_(index_bao, index_bao)] = corr_bao
    std_full[index_bao] = std_bao
    covariance = covariance_stat = covariance.clone(value=corr_full * std_full[..., None] * std_full)
    
    def add_systematics(covariance_stat, systs=('rotation', 'hod', 'photo')):
        covariance = covariance_stat.deepcopy()
        # Now, let's add systematic covariances: marginalization over mo vectors in the window matrix rotation, HOD systematics (Nathan), photometric systematics (Ruiyang). Here it's a bit more hacky
        # Select indices corresponding to 'power'
        index_power = covariance._index(observables='power', concatenate=True)
        assert np.allclose(index_power, np.arange(covariance.observables('power').size))
        value = covariance._value[np.ix_(index_power, index_power)]
        # Sum over all P(k) systematic covariances
        for s in systs:
            fn = covsyst_fns[s]
            syst = ObservableCovariance.load(fn).select(xlim=klim[:2], projs=list(ells), select_projs=True)
            # value += syst
            if syst.shape == value.shape:
                # print(syst/value)
                value += syst
    
            else:
                # syst must be smaller; check that it's square and even-sized
                syst=syst._value
                m = value.shape[0]      # full dimension (2m × 2m)
                n = syst.shape[0]       # smaller dimension (2n × 2n)
            
                if n % 2 != 0 or m % 2 != 0:
                    raise ValueError("Both covariance matrices must have even dimensions (2n, 2m).")
            
                n2 = n // 2   # number of elements in each block in syst
                m2 = m // 2   # size of each block in value
            
                # Add syst blocks to corresponding blocks in value
                # Top-left block
                # print(syst._value)
                value[:n2, :n2] += syst[:n2, :n2]
            
                # Bottom-right block
                value[m2:m2+n2, m2:m2+n2] += syst[n2:n, n2:n]
        covariance._value[np.ix_(index_power, index_power)] = value
        
        # And BAO systematics
        bao_fn = forfit_dir / f'covariance_syst_bao-recon_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy'
        syst = ObservableCovariance.load(bao_fn).view(projs=covariance.observables('bao-recon').projs)  # select qpar, qper covariance
        index_bao = covariance._index(observables='bao-recon', concatenate=True)
        value_bao=covariance._value[np.ix_(index_bao, index_bao)]
        # print(value_bao.shape,syst.shape)
        value_bao += syst
        return covariance
    
    
    covariance = add_systematics(covariance_stat)
    if no_syst:
        covariance = covariance_stat
    # Now let's cut the data vector
    # P(k) data
    # '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/corrected/pkpoles_corrected_LRG_GCcomb_z0.8-1.1_thetacut0.05.npy'
    power_fn = base_dir / f'baseline_2pt/pk/rotated_corrected/pkpoles_rotated_corrected_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.npy'
    
    power_fn = base_dir / f'baseline_2pt/pk/corrected/pkpoles_corrected_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.npy'
    # P(k) window
    wmatrix_fn = base_dir / f'baseline_2pt/pk/wmatrix_smooth_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.npy'
    # '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/wmatrix_smooth_LRG_GCcomb_z0.6-0.8_thetacut0.05.npy',
    # Power spectrum (pre-recon)
    power = PowerSpectrumMultipoles.load(power_fn)
    
    zeff = power.attrs['zeff']
    k, power = power.select(klim)(ell=ells, return_k=True, complex=False)
    # Let's replace in covariance.observables(), a bit hacky
    observable_power = covariance.observables('power')
    observable_power._value = list(power)
    observable_power.attrs['zeff'] = zeff  # effective redshift
    # BAO alpha
    bao = covariance_bao.observables('bao-recon').view(projs=covariance.observables('bao-recon').projs)  # select qpar, qper
    observable_bao = covariance.observables('bao-recon')
    observable_bao._value = list(bao[:, None])
    observable_bao.attrs['zeff'] = covariance_bao.observables('bao-recon').attrs['zeff']  # effective redshift
    
    # Eventually, let's cut out the P(k) window matrix
    wmatrix = BaseMatrix.load(wmatrix_fn)
    def prepare_wmatrix(wmatrix):
        # For data, zeff in the window matrix is the same as in the P(k) measurement
        assert np.allclose(wmatrix.attrs['zeff'], zeff)
        ellsin = [proj.ell for proj in wmatrix.projsin]
        # print(klim[:2])
        wmatrix = wmatrix.select_x(xinlim=(0.001, 0.4), xoutlim=klim[:2])  # apply k-cut to both input (theory) and output (observed); some margin for input kmax
        wmatrix = wmatrix.select_proj(projsout=[(ell, None) for ell in ells])  # select output ells
        # Let's rebin the theory k (linear interpolation), to avoid a too large window matrix
        kin = np.arange(0.001, 0.35, 0.001)
        from scipy import linalg
        
        def matrix_lininterp(xin, xout):
            # Matrix for linear interpolation
            toret = np.zeros((len(xin), len(xout)), dtype='f8')
            for iout, xout in enumerate(xout):
                iin = np.searchsorted(xin, xout, side='right') - 1
                if 0 <= iin < len(xin) - 1:
                    frac = (xout - xin[iin]) / (xin[iin + 1] - xin[iin])
                    toret[iin, iout] = 1. - frac
                    toret[iin + 1, iout] = frac
                elif np.isclose(xout, xin[-1]):
                    toret[iin, iout] = 1.
            return toret
        
        rebin = linalg.block_diag(*[matrix_lininterp(kin, xin) for xin in wmatrix.xin])
        wmatrix = wmatrix.value.T.dot(rebin.T)  # rebinned window matrix
        return kin, ellsin, wmatrix
    
    kin, ellsin, wmatrix = prepare_wmatrix(wmatrix)
    observable_power.attrs['wmatrix'] = wmatrix
    observable_power.attrs['kin'] = kin
    # Note, all this is done in the forfit* files for P(k) and P(k) + BAO
    forfit_dir = Path('/global/cfs/cdirs/desi//survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/forfit_2pt/')
    ref_fn = forfit_dir / f'forfit_power+bao-recon_syst-rotation-hod-photo_klim_0-0.02-0.20_2-0.02-0.20_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy'
    covariance_ref = ObservableCovariance.load(ref_fn)
    # assert np.allclose(covariance.view(), covariance_ref.view())
    # assert np.allclose(observable_power.view(), covariance_ref.observables('power').view())
    # assert np.allclose(wmatrix, covariance_ref.observables('power').attrs['wmatrix'])
    
    # invcov= np.linalg.inv(covariance._value)
    
    return observable_power, observable_bao, covariance, wmatrix_fn