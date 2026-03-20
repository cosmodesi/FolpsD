
#Script adapted from Marcos's notebook

from pathlib import Path
import numpy as np
import lsstypes as types
from clustering_statistics.tools import get_stats_fn
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# from triumvirate._arrayops import reshape_threept_datatab
from lsstypes import ObservableTree
# from folps.conversion import convert_triumvirate_window3
# from triumvirate.winconv import (
#     BispecWinConv,
#     ThreePointWindow,
#     WinConvFormulae, Multipole,
# )
from pathlib import Path

def percival_factor(Ns, Nd, Np):
    """
    Compute the Percival et al. (2014) correction factor for parameter covariance.

    Parameters
    ----------
    Ns : int
        Number of mocks used to estimate the data covariance.
    Nd : int
        Length of the data vector.
    Np : int
        Number of fitted parameters (including analytically marginalized ones).

    Returns
    -------
    m : float
        Percival correction factor. Parameter covariance should be multiplied by m.
        Parameter errors scale as sqrt(m).
    """
    A = 2.0 / ((Ns - Nd - 1) * (Ns - Nd - 4))
    B = (Ns - Nd - 2) / ((Ns - Nd - 1) * (Ns - Nd - 4))
    m = (1 + B * (Nd - Np)) / (1 + A + B * (Np + 1))
    return m

def build_pk_bk_data_cutsky(
    k_min_p=0.02, k_max_p=0.2,
    k_min_b=0.02, k_max_b0=0.12, k_max_b2=0.08, region='SGC', tracer='LRG2',
    bk_window=False, use_Jaides_window=False,
    use_hartlap=True,binning=1, Nm=None, use_percival=True
):

    print("\n" + "="*70)
    print("P(k) + B(k) Data & Covariance Summary")
    print("="*70)

    stats_dir = Path('/global/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe')
    tracer_dict = {
    'LRG1': ['LRG', (0.4, 0.6)],
    'LRG2': ['LRG', (0.6, 0.8)],
    'LRG3': ['LRG', (0.8, 1.1)],
    'QSO':  ['QSO', (0.8, 2.1)],
    'ELG':  ['ELG_LOP', (1.1, 1.6)],
}


    tracer_type, zrange = tracer_dict[tracer]

    print("\nLoading data spectra (Abacus mocks)")
    print(f"\nTracer: {tracer_type}, z: {zrange}, region")
    print("-" * 70)

    # Number of mocks to average
    n_mocks = 25

    p0_list = []
    p2_list = []
    b000_list = []
    b202_list = []
    pspectrum_list = []
    bspectrum_list = []

    for imock in range(n_mocks):

        fn_pk = get_stats_fn(
            stats_dir=stats_dir,
            kind='mesh2_spectrum',
            version='abacus-2ndgen-complete',
            tracer=tracer_type,
            zrange=zrange,
            region=region,
            weight='default-FKP',
            imock=imock
        )

        fn_bk = get_stats_fn(
            stats_dir=stats_dir,
            kind='mesh3_spectrum',
            version='abacus-2ndgen-complete',
            tracer=tracer_type,
            zrange=zrange,
            region=region,
            weight='default-FKP',
            basis='sugiyama-diagonal',
            imock=imock
        )

        pspectrum = types.read(fn_pk)
        bspectrum = types.read(fn_bk)

        # --- Power spectrum ---
        pspectrum = pspectrum.select(k=slice(0, None, 5))
        if binning ==2:
            pspectrum = pspectrum.select(k=slice(0, None, 2))
        pspectrum = pspectrum.select(k=(k_min_p, k_max_p))
        pspectrum = pspectrum.get(ells=[0, 2])

        p0_list.append(pspectrum.get(ells=0).value())
        p2_list.append(pspectrum.get(ells=2).value())
        pspectrum_list.append(pspectrum)

        # --- Bispectrum ---
        bspectrum = bspectrum.select(k=(k_min_b, k_max_b0))
        if binning ==2:
            bspectrum = bspectrum.select(k=slice(0, None, 2))
        bspectrum = bspectrum.get(ells=[(0,0,0), (2,0,2)])
        bspectrum = bspectrum.at(ells=(2,0,2)).select(k=(k_min_b, k_max_b2))

        b000_list.append(bspectrum.get(ells=(0,0,0)).value())
        b202_list.append(bspectrum.get(ells=(2,0,2)).value())
        bspectrum_list.append(bspectrum)

    # Convert to arrays and take mean
    p0   = np.mean(np.array(p0_list), axis=0)
    p2   = np.mean(np.array(p2_list), axis=0)
    b000 = np.mean(np.array(b000_list), axis=0)
    b202 = np.mean(np.array(b202_list), axis=0)

    # Mean lsstypes objects (used as data in the desilike observables)
    pspectrum = types.mean(pspectrum_list)
    bspectrum = types.mean(bspectrum_list)

    # k grid (same for all mocks)
    k_data = pspectrum.get(ells=0).coords('k')
    kr_b0 = bspectrum.get(ells=(0,0,0)).coords('k')
    kr_b2 = bspectrum.get(ells=(2,0,2)).coords('k')

    print(f"  P0 shape      : {p0.shape}")
    print(f"  P2 shape      : {p2.shape}")
    print(f"  B000 shape    : {b000.shape}")
    print(f"  B202 shape    : {b202.shape}")

    # Total data vector
    total_length = p0.size + p2.size + b000.size + b202.size
    print(f"  Total data vector length : {total_length}")
    print(f"  Total data vector shape  : ({total_length},)")


    print("\nBuilding covariance from holi mocks")
    print("-"*70)

    observables = []
    missing = []
    available = []

    for imock in range(1000):
        if tracer_type == 'ELG_LOP':
            tracer_type_cov = 'ELG_LOPnotqso'
        else:
            tracer_type_cov = tracer_type

        kw = dict(
            stats_dir=stats_dir,
            version='holi-v1-altmtl',
            tracer=tracer_type_cov,
            zrange=zrange,
            region=region,
            weight='default-FKP',
            imock=imock
        )

        fn2 = get_stats_fn(kind='mesh2_spectrum', **kw)
        fn3 = get_stats_fn(kind='mesh3_spectrum', basis='sugiyama-diagonal', **kw)

        if not (fn2.exists() and fn3.exists()):
            missing.append(imock)
            continue

        available.append(imock)

        spectrum2, spectrum3 = types.read(fn2), types.read(fn3)
        tree = ObservableTree(
            [spectrum2, spectrum3],
            observables=['spectrum2', 'spectrum3']
        )
        observables.append(tree)

    print(f"  Missing mocks ({len(missing)})")
    print(f"  Available mocks ({len(available)})")

    covariance = types.cov(observables)

    # ---------------------------------------------------
    # Propagate selections to covariance
    # ---------------------------------------------------

    observable = covariance.observable

    spectrum2 = observable.get(observables='spectrum2')
    spectrum2 = spectrum2.get(ells=[0, 2])
    spectrum2 = spectrum2.select(k=slice(0, None, 5))
    if binning ==2:
        spectrum2 = spectrum2.select(k=slice(0, None, 2))
    spectrum2 = spectrum2.select(k=(k_min_p, k_max_p))

    spectrum3 = observable.get(observables='spectrum3')
    if binning ==2:
        spectrum3 = spectrum3.select(k=slice(0, None, 2))
    spectrum3 = spectrum3.select(k=(k_min_b, k_max_b0))
    spectrum3 = spectrum3.at(ells=(2,0,2)).select(k=(k_min_b, k_max_b2))

    observable = observable.at(observables='spectrum2').match(spectrum2)
    observable = observable.at(observables='spectrum3').match(spectrum3)

    covariance = covariance.at.observable.match(observable)
    # Extract P(k)-only covariance block (P0 and P2 together)
    cov_pk = covariance.at.observable.get(observables='spectrum2').value()

    # ---------------------------------------------------

    cov = covariance.value()
    print(f"  Full Cov shape: {cov.shape}")
    cov_bk = covariance.at.observable.get(observables='spectrum3')
    cov_pk = covariance.at.observable.get(observables='spectrum2')

    # --- Hartlap ---
    # Provide number of simulations used for covariance estimation, if not using all available
    if Nm is None:
        Nm = len(available)
    Nd= total_length
    hartlap = (Nm - Nd - 2)/(Nm - 1)
    

    if use_hartlap:
        cov = cov/hartlap
        print(f"  Hartlap factor: {hartlap:.4f}")

    if use_percival:
        percival= percival_factor(Ns=Nm, Nd=Nd, Np=9) #9params for Pk+Bk
        cov = cov*percival
        print(f"  Percival factor: {percival:.4f}")


    print("\nLoading Pk window matrix")
    print("-"*70)


    # window = types.read(
    #     f'/global/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe/abacus-2ndgen-complete/window_mesh2_spectrum_poles_LRG_z0.6-0.8_{region}_weight-default-FKP_0.h5'
    # )
    window_fn = get_stats_fn(stats_dir=stats_dir, kind='window_mesh2_spectrum', version='abacus-2ndgen-complete', 
                tracer=tracer_type,
                zrange=zrange,
                region=region, weight='default-FKP', imock=0)
    window=types.read(window_fn)

    window = window.at.observable.match(pspectrum)
    window = window.at.theory.select(k=(0, 0.5))
    window_pk = window

    wmatnp = window.value()
    zeff = window.observable.get(ells=0).attrs['zeff']
    k_window = window.theory.get(ells=0).coords('k')

    print(f"  Window matrix shape : {wmatnp.shape}")
    print(f"  Window k shape      : {k_window.shape}")
    print(f"  Effective z         : {zeff}")

    print("\nLoading Bk window matrix")
    print("-"*70)


    if bk_window:
        #Bispectrum Window
        window_fn = get_stats_fn(stats_dir=stats_dir, kind='window_mesh3_spectrum', version='abacus-2ndgen-complete', 
                    tracer=tracer_type,
                    zrange=zrange,
                    region=region, weight='default-FKP', basis='sugiyama-diagonal', imock=0)
        window=types.read(window_fn)

        if use_Jaides_window:
            window=convert_triumvirate_window3(tracer=tracer_type, zrange=zrange, region=region, add_norm=True)

        window = window.at.observable.match(bspectrum)
        # window = window.at.theory.select(k=(0, 0.30))
        # window = window.at.theory.select(k=slice(0, None,1))
        

        w000 = window.at.observable.get(ells=(0,0,0)).value()
        w202 = window.at.observable.get(ells=(2,0,2)).value()
        # wmatnb = window.value()
        wmatnb=window
        zeff = window.observable.get(ells=(0,0,0)).attrs['zeff']
        k_window_b = window.theory.get(ells=(0,0,0)).coords('k')
        k_window_b1d = window.theory.get(ells=(0,0,0)).unravel().coords('k1')
        
        window_bk = window
    
        

        print(f"  Window B0 matrix shape : {w000.shape}")
        print(f"  Window B2 matrix shape : {w202.shape}")
        print(f"  Window Bk theory k shape      : {k_window_b.shape}")
        print(f"  Window Bk theory k range       : {float(k_window_b1d[0]):.3f} - {float(k_window_b1d[-1]):.3f}")
        print(f"  Window Bk theory k binning     : {k_window_b1d[1] - k_window_b1d[0]:.3f}")



    print("\n" + "="*70)
    print("Data vector + covariance + window ready")
    print("="*70 + "\n")


    # Plot the window matrix as a covariance and save it
    plt.figure(figsize=(8, 6))
    plt.imshow(wmatnp, cmap='viridis', aspect='auto')
    plt.colorbar(label='Window Matrix Value')
    plt.title('Window Matrix')
    plt.xlabel('k (h/Mpc)')
    plt.ylabel('k (h/Mpc)')
    plt.tight_layout()
    plt.savefig('window_matrix.png', dpi=300)
    
    return {
        "k_data": k_data,
        "p0": p0,
        "p2": p2,
        "kr_b0":kr_b0,
        "kr_b2":kr_b2,
        "b000": b000,
        "b202": b202,
        "covariance": cov,
        "cov_pk":cov_pk,
        # "window_matrix": wmatnp,
        "window_matrix": window_pk,
        "k_window": k_window,
        "zeff": zeff,
        "Nm": len(available),
        'hartlap_factor': hartlap,
        # "window_matrix_bk": wmatnb if bk_window else None,
        "window_matrix_bk": window_bk if bk_window else None,
        "k_window_b": k_window_b if bk_window else None,
        "bk_lsstypes":bspectrum,
        "pk_lsstypes":pspectrum,
        'cov_bk': cov_bk,
    }

def build_pk_data_cutsky(
    k_min_p=0.02, k_max_p=0.2,
    region='SGC', tracer='LRG2',
    save_fn=None,Nm=None,use_percival=True
):

    print("\n" + "="*70)
    print("P(k) Data & Covariance Summary (No Bispectrum)")
    print("="*70)

    stats_dir = Path('/global/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe')

    tracer_dict = {
        'LRG1': ['LRG', (0.4, 0.6)],
        'LRG2': ['LRG', (0.6, 0.8)],
        'LRG3': ['LRG', (0.8, 1.1)],
        'QSO':  ['QSO', (0.8, 2.1)],
        'ELG':  ['ELG_LOP', (1.1, 1.6)],
    }

    tracer_type, zrange = tracer_dict[tracer]

    print("\nLoading data spectra (Abacus mocks)")
    print(f"\nTracer: {tracer_type}, z: {zrange}, region: {region}")
    print("-" * 70)

    n_mocks = 25

    p0_list = []
    p2_list = []
    pspectrum_list = []
    

    # ---------------------------------------------------
    # Load and average data mocks
    # ---------------------------------------------------

    for imock in range(n_mocks):

        fn_pk = get_stats_fn(
            stats_dir=stats_dir,
            kind='mesh2_spectrum',
            version='abacus-2ndgen-complete',
            tracer=tracer_type,
            zrange=zrange,
            region=region,
            weight='default-FKP',
            imock=imock
        )

        pspectrum = types.read(fn_pk)

        pspectrum = pspectrum.select(k=slice(0, None, 5))
        pspectrum = pspectrum.select(k=(k_min_p, k_max_p))
        pspectrum = pspectrum.get(ells=[0, 2])

        p0_list.append(pspectrum.get(ells=0).value())
        p2_list.append(pspectrum.get(ells=2).value())
        pspectrum_list.append(pspectrum)

    p0 = np.mean(np.array(p0_list), axis=0)
    p2 = np.mean(np.array(p2_list), axis=0)
    pspectrum = types.mean(pspectrum_list)

    k_data = pspectrum.get(ells=0).coords('k')

    print(f"  P0 shape : {p0.shape}")
    print(f"  P2 shape : {p2.shape}")

    total_length = p0.size + p2.size
    print(f"  Total data vector length : {total_length}")

    # ---------------------------------------------------
    # Build covariance from holi mocks
    # ---------------------------------------------------

    print("\nBuilding covariance from holi mocks")
    print("-"*70)

    observables = []

    missing = []
    available = []

    for imock in range(1000):

        if tracer_type == 'ELG_LOP':
            tracer_type_cov = 'ELG_LOPnotqso'
        else:
            tracer_type_cov = tracer_type

        kw = dict(
            stats_dir=stats_dir,
            version='holi-v1-altmtl',
            tracer=tracer_type_cov,
            zrange=zrange,
            region=region,
            weight='default-FKP',
            imock=imock
        )

        fn2 = get_stats_fn(kind='mesh2_spectrum', **kw)

        if not fn2.exists():
            missing.append(imock)
            continue

        available.append(imock)

        spectrum2 = types.read(fn2)

        tree = ObservableTree(
            [spectrum2],
            observables=['spectrum2']
        )

        observables.append(tree)

    # print(f"Available mocks ({len(available)}): {available}")
    # print(f"Missing mocks   ({len(missing)}): {missing}")

    print(f"Available mocks ({len(available)})")
    print(f"Missing mocks   ({len(missing)})")

    if len(observables) == 0:
        raise RuntimeError("No valid mocks found!")
    covariance = types.cov(observables)

    observable = covariance.observable

    spectrum2 = observable.get(observables='spectrum2')
    spectrum2 = spectrum2.get(ells=[0, 2])
    spectrum2 = spectrum2.select(k=slice(0, None, 5))
    spectrum2 = spectrum2.select(k=(k_min_p, k_max_p))

    observable = observable.at(observables='spectrum2').match(spectrum2)
    covariance = covariance.at.observable.match(observable)

    cov = covariance.value()
    

    print(f"  Covariance shape : {cov.shape}")

    # --- Hartlap ---
    # Provide number of simulations used for covariance estimation, if not using all available
    if Nm is None:
        Nm = len(available)
    Nd= total_length
    hartlap = (Nm - Nd - 2)/(Nm - 1)
    print(f"  Hartlap factor: {hartlap:.4f}")

    cov = cov/hartlap

    if use_percival:
        percival= percival_factor(Ns=Nm, Nd=Nd, Np=7) #7params for Pk
        cov = cov*percival
        print(f"  Percival factor: {percival:.4f}")

    # ---------------------------------------------------
    # Window matrix
    # ---------------------------------------------------

    print("\nLoading window matrix")
    print("-"*70)

    window_fn = get_stats_fn(
        stats_dir=stats_dir,
        kind='window_mesh2_spectrum',
        version='abacus-2ndgen-complete',
        tracer=tracer_type,
        zrange=zrange,
        region=region,
        weight='default-FKP',
        imock=0
    )

    window = types.read(window_fn)

    window = window.at.observable.match(pspectrum)
    window = window.at.theory.select(k=(0, 0.5))

    wmatnp = window.value()
    zeff = window.observable.get(ells=0).attrs['zeff']
    k_window = window.theory.get(ells=0).coords('k')

    print(f"  Window matrix shape : {wmatnp.shape}")
    print(f"  Effective z         : {zeff}")

    print("\n" + "="*70)
    print("P(k) data + covariance + window ready")
    print("="*70 + "\n")



    result= {
        "k_data": k_data,
        "p0": p0,
        "p2": p2,
        "covariance": cov,
        "window": window,
        "k_window": k_window,
        "zeff": zeff,
        "Nm": len(available),
        "hartlap_factor": hartlap,
        "pk_lsstypes":pspectrum,
    }

    # --- Optional saving ---
    if save_fn is not None:
        np.savez_compressed(save_fn, **result)

    return result

def build_bk_window_cutsky(
    k_min_p=0.02, k_max_p=0.2,
    k_min_b=0.02, k_max_b0=0.12, k_max_b2=0.08, region='SGC', tracer='LRG2',Nm=None
):

    print("\n" + "="*70)
    print("P(k) + B(k) Data & Covariance Summary")
    print("="*70)

    stats_dir = Path('/global/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe')
    tracer_dict = {
    'LRG1': ['LRG', (0.4, 0.6)],
    'LRG2': ['LRG', (0.6, 0.8)],
    'LRG3': ['LRG', (0.8, 1.1)],
    'QSO':  ['QSO', (0.8, 2.1)],
    'ELG':  ['ELG_LOP', (1.1, 1.6)],
}


    tracer_type, zrange = tracer_dict[tracer]

    print("\nLoading data spectra (Abacus mocks)")
    print(f"\nTracer: {tracer_type}, z: {zrange}, region")
    print("-" * 70)

    # Number of mocks to average
    n_mocks = 25

    p0_list = []
    p2_list = []
    b000_list = []
    b202_list = []

    for imock in range(n_mocks):

        fn_pk = get_stats_fn(
            stats_dir=stats_dir,
            kind='mesh2_spectrum',
            version='abacus-2ndgen-complete',
            tracer=tracer_type,
            zrange=zrange,
            region=region,
            weight='default-FKP',
            imock=imock
        )

        fn_bk = get_stats_fn(
            stats_dir=stats_dir,
            kind='mesh3_spectrum',
            version='abacus-2ndgen-complete',
            tracer=tracer_type,
            zrange=zrange,
            region=region,
            weight='default-FKP',
            basis='sugiyama-diagonal',
            imock=imock
        )

        pspectrum = types.read(fn_pk)
        bspectrum = types.read(fn_bk)

        # --- Power spectrum ---
        pspectrum = pspectrum.select(k=slice(0, None, 5))
        pspectrum = pspectrum.select(k=(k_min_p, k_max_p))
        pspectrum = pspectrum.get(ells=[0, 2])

        p0_list.append(pspectrum.get(ells=0).value())
        p2_list.append(pspectrum.get(ells=2).value())

        # --- Bispectrum ---
        bspectrum = bspectrum.select(k=(k_min_b, k_max_b0))
        bspectrum = bspectrum.get(ells=[(0,0,0), (2,0,2)])
        bspectrum = bspectrum.at(ells=(2,0,2)).select(k=(k_min_b, k_max_b2))

        b000_list.append(bspectrum.get(ells=(0,0,0)).value())
        b202_list.append(bspectrum.get(ells=(2,0,2)).value())

    # Convert to arrays and take mean
    p0   = np.mean(np.array(p0_list), axis=0)
    p2   = np.mean(np.array(p2_list), axis=0)
    b000 = np.mean(np.array(b000_list), axis=0)
    b202 = np.mean(np.array(b202_list), axis=0)

    # k grid (same for all mocks)
    k_data = pspectrum.get(ells=0).coords('k')
    kr_b0 = bspectrum.get(ells=(0,0,0)).coords('k')
    kr_b2 = bspectrum.get(ells=(2,0,2)).coords('k')

    print(f"  P0 shape      : {p0.shape}")
    print(f"  P2 shape      : {p2.shape}")
    print(f"  B000 shape    : {b000.shape}")
    print(f"  B202 shape    : {b202.shape}")

    # Total data vector
    total_length = p0.size + p2.size + b000.size + b202.size
    print(f"  Total data vector length : {total_length}")
    print(f"  Total data vector shape  : ({total_length},)")


    print("\nBuilding covariance from holi mocks")
    print("-"*70)

    observables = []
    missing = []
    available = []


    #Bispectrum Window
    window_fn = get_stats_fn(stats_dir=stats_dir, kind='window_mesh3_spectrum', version='abacus-2ndgen-complete', 
                tracer=tracer_type,
                zrange=zrange,
                region=region, weight='default-FKP', basis='sugiyama-diagonal', imock=0)
    window=types.read(window_fn)

    window = window.at.observable.select(k=slice(0, None,5))    
    window = window.at.observable.select(k=(k_min_b, 0.3))    
    # window = window.at.observable.match(bspectrum)
    window = window.at.theory.select(k=(0, 0.3))
    window = window.at.theory.select(k=slice(0, None,2))
    

    w000 = window.at.observable.get(ells=(0,0,0)).value()
    w202 = window.at.observable.get(ells=(2,0,2)).value()
    wmatnb = window.value()
    zeff = window.observable.get(ells=(0,0,0)).attrs['zeff']
    k_window_b = window.theory.get(ells=(0,0,0)).unravel().coords('k1')
    k_window_obs = window.theory.get(ells=(0,0,0)).unravel().coords('k1')
    print(k_window_obs)
    print(kr_b0)
    

    print(f"  Window B0 matrix shape : {w000.shape}")
    print(f"  Window B2 matrix shape : {w202.shape}")
    print(f"  Window Bk theory k shape       : {k_window_b.shape}")
    print(f"  Window Bk theory k range       : {float(k_window_b[0]):.3f} - {float(k_window_b[-1]):.3f}")
    print(f"  Window Bk theory k binning     : {k_window_b[1] - k_window_b[0]:.3f}")




    
    return {
        "k_window_b": k_window_b,
        "w000": w000,
        "w202": w202,
    }


def build_pk_data_dr1_cutsky(
    k_min_p=0.02, k_max_p=0.2,
    region='SGC', tracer='LRG2',
    save_fn=None,Nm=None,use_percival=True
):

    print("\n" + "="*70)
    print("DR1 CutSky P(k) Data & Covariance Summary")
    print("="*70)

    
    stats_dir = Path('/global/cfs/cdirs/desi/public/dr1/vac/dr1/full-shape-bao-clustering/v1.0')

    tracer_dict = {
        'LRG1': ['LRG', (0.4, 0.6)],
        'LRG2': ['LRG', (0.6, 0.8)],
        'LRG3': ['LRG', (0.8, 1.1)],
        'QSO':  ['QSO', (0.8, 2.1)],
        'ELG':  ['ELG_LOP', (1.1, 1.6)],
    }

    tracer_type, zrange = tracer_dict[tracer]
    zstr = f"{zrange[0]}-{zrange[1]}"
    print(f"\nTracer: {tracer_type}, z: {zrange}, region: {region}")
    print("-"*70)

    # ---------------------------------------------------
    # DATA VECTOR: mean of 25 Abacus cutsky mocks
    # ---------------------------------------------------

    print("\nLoading Abacus cutsky mocks")

    n_mocks = 25

    p0_list = []
    p2_list = []
    pspectrum_list = []

    for imock in range(n_mocks):

        pspectrum =types.read(f'{stats_dir}/AbacusSummit/complete/spectrum/spectrum-poles_{tracer_type}_{region}_z{zstr}_{imock}.h5')
        # pspectrum = types.read(fn_pk)
        pspectrum = pspectrum.select(k=slice(0, None, 5))
        pspectrum = pspectrum.select(k=(k_min_p, k_max_p))
        pspectrum = pspectrum.get(ells=[0, 2])

        p0_list.append(pspectrum.get(ells=0).value())
        p2_list.append(pspectrum.get(ells=2).value())
        pspectrum_list.append(pspectrum)

    p0 = np.mean(np.array(p0_list), axis=0)
    p2 = np.mean(np.array(p2_list), axis=0)
    pspectrum = types.mean(pspectrum_list)

    k_data = pspectrum.get(ells=0).coords('k')

    print(f"P0 shape: {p0.shape}")
    print(f"P2 shape: {p2.shape}")

    total_length = p0.size + p2.size
    print(f"Total data vector length: {total_length}")

    # ---------------------------------------------------
    # COVARIANCE: EZ mocks
    # ---------------------------------------------------

    print("\nBuilding covariance from EZ mocks")

    observables = []
    missing = []
    available = []

    
    ez_dir = f'/global/cfs/cdirs/desi/public/dr1/vac/dr1/full-shape-bao-clustering/v1.0/EZmock/ffa/spectrum/'

    for imock in range(1000):

        try:
            fn = f'{ez_dir}/spectrum-poles_{tracer_type}_{region}_z{zstr}_{imock}.h5'
            spectrum2 = types.read(fn)

            tree = ObservableTree(
                [spectrum2],
                observables=['spectrum2']
            )

            observables.append(tree)
            available.append(imock)

        except Exception:

            missing.append(imock)

    print(f"Available mocks ({len(available)})")
    print(f"Missing mocks   ({len(missing)})")

    if len(observables) == 0:
        raise RuntimeError("No valid EZ mocks found!")

    covariance = types.cov(observables)

    observable = covariance.observable

    spectrum2 = observable.get(observables='spectrum2')
    spectrum2 = spectrum2.get(ells=[0, 2])
    spectrum2 = spectrum2.select(k=slice(0, None, 5))
    spectrum2 = spectrum2.select(k=(k_min_p, k_max_p))

    observable = observable.at(observables='spectrum2').match(spectrum2)
    covariance = covariance.at.observable.match(observable)

    cov = covariance.value()
    print(f"Covariance shape : {cov.shape}")

    # ---------------------------------------------------
    # Hartlap correction
    # ---------------------------------------------------

   
    # Provide number of simulations used for covariance estimation, if not using all available
    if Nm is None:
        Nm = len(available)
    Nd = total_length

    hartlap = (Nm - Nd - 2) / (Nm - 1)

    print(f"Hartlap factor: {hartlap:.4f}")

    cov = cov / hartlap

    if use_percival:
        percival= percival_factor(Ns=Nm, Nd=Nd, Np=7) #7params for Pk
        cov = cov*percival
        print(f"  Percival factor: {percival:.4f}")

    # ---------------------------------------------------
    # Window matrix
    # ---------------------------------------------------

    print("\nLoading window matrix")

    
    window_fn = fn = f'{ez_dir}/window_spectrum-poles_{tracer_type}_{region}_z{zstr}.h5'
    window = types.read(window_fn)

    window = window.at.observable.match(pspectrum)
    window = window.at.theory.select(k=(0, 0.5))

    wmatnp = window.value()

    # zeff = window.observable.get(ells=0).attrs['zeff']
    zeff = pspectrum.attrs['zeff']
    k_window = window.theory.get(ells=0).coords('k')

    print(f"Window matrix shape : {wmatnp.shape}")
    print(f"Effective z         : {zeff}")

    print("\n" + "="*70)
    print("DR1 P(k) data + covariance + window ready")
    print("="*70 + "\n")

    result = {
        "k_data": k_data,
        "p0": p0,
        "p2": p2,
        "covariance": cov,
        "covariance_lsstypes": covariance,
        "window": window,
        "k_window": k_window,
        "zeff": zeff,
        "Nm": Nm,
        "hartlap_factor": hartlap,
        "pk_lsstypes": pspectrum,
    }

    if save_fn is not None:
        np.savez_compressed(save_fn, **result)

    return result