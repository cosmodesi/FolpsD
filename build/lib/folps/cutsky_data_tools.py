
#Script adapted from Marcos's notebook

from pathlib import Path
import numpy as np
import lsstypes as types
from clustering_statistics.tools import get_stats_fn
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from triumvirate._arrayops import reshape_threept_datatab
from lsstypes import ObservableTree
from triumvirate.winconv import (
    BispecWinConv,
    ThreePointWindow,
    WinConvFormulae, Multipole,
)
from pathlib import Path

def build_pk_bk_data_cutsky(
    k_min_p=0.02, k_max_p=0.2,
    k_min_b=0.02, k_max_b0=0.12, k_max_b2=0.08, region='SGC', tracer='LRG2',
    bk_window=False
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

    for imock in range(1000):
        kw = dict(
            stats_dir=stats_dir,
            version='holi-v1-altmtl',
            tracer=tracer_type,
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
    spectrum2 = spectrum2.select(k=(k_min_p, k_max_p))

    spectrum3 = observable.get(observables='spectrum3')
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


    # --- Hartlap ---
    Nm = len(available)
    Nd= total_length
    hartlap = (Nm - Nd - 2)/(Nm - 1)
    print(f"  Hartlap factor: {hartlap:.4f}")

    cov = cov/hartlap

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

        

        window = window.at.observable.match(bspectrum)
        window = window.at.theory.select(k=(0, 0.3))
        window = window.at.theory.select(k=slice(0, None,2))
        

        w000 = window.at.observable.get(ells=(0,0,0)).value()
        w202 = window.at.observable.get(ells=(2,0,2)).value()
        wmatnb = window.value()
        zeff = window.observable.get(ells=(0,0,0)).attrs['zeff']
        k_window_b = window.theory.get(ells=(0,0,0)).unravel().coords('k1')
        print(k_window_b)
        

        print(f"  Window B0 matrix shape : {w000.shape}")
        print(f"  Window B2 matrix shape : {w202.shape}")
        print(f"  Window Bk theory k shape      : {k_window_b.shape}")
        print(f"  Window Bk theory k range       : {float(k_window_b[0]):.3f} - {float(k_window_b[-1]):.3f}")
        print(f"  Window Bk theory k binning     : {k_window_b[1] - k_window_b[0]:.3f}")



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
        "window_matrix": wmatnp,
        "k_window": k_window,
        "zeff": zeff,
        "Nm": len(available),
        'hartlap_factor': hartlap,
    }

def build_pk_data_cutsky(
    k_min_p=0.02, k_max_p=0.2,
    region='SGC', tracer='LRG2',
    save_fn=None
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

    p0 = np.mean(np.array(p0_list), axis=0)
    p2 = np.mean(np.array(p2_list), axis=0)

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
    cov_pk = cov  # identical in this case

    print(f"  Covariance shape : {cov.shape}")

    # --- Hartlap ---
    Nm = len(available)
    Nd= total_length
    hartlap = (Nm - Nd - 2)/(Nm - 1)
    print(f"  Hartlap factor: {hartlap:.4f}")

    cov = cov/hartlap

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
        "cov_pk": cov_pk,
        "window_matrix": wmatnp,
        "k_window": k_window,
        "zeff": zeff,
        "Nm": len(available),
        "hartlap_factor": hartlap,
    }

    # --- Optional saving ---
    if save_fn is not None:
        np.savez_compressed(save_fn, **result)

    return result

def build_bk_window_cutsky(
    k_min_p=0.02, k_max_p=0.2,
    k_min_b=0.02, k_max_b0=0.12, k_max_b2=0.08, region='SGC', tracer='LRG2',
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