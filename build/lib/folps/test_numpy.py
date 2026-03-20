import time
import numpy as np
import os

# Select NumPy backend before importing folps.py
os.environ["FOLPS_BACKEND"] = "numpy"  # <- NumPy this time
from folps import *
from cosmo_class import *

use_class = True
try:
    from cosmo_class import run_class
    from classy import Class as _Class  # noqa: F401
except Exception:
    use_class = False

if use_class:
    classy = run_class(h=0.6711, ombh2=0.022, omch2=0.122, omnuh2=0.0006442,
                       As=2e-9, ns=0.965, z=0.3, z_scale=[0.97],
                       N_ur=2.0328,
                       khmin=0.0001, khmax=2.0, nbk=1000, spectra='cb')
else:
    data_path = 'inputpkT.txt'
    k_arr, pk_arr = np.loadtxt(data_path, unpack=True)
    classy = {'k': k_arr, 'pk': pk_arr}

kwargs = {
    'z': 0.3,
    'h': 0.6711,
    'Omega_m': 0.3211636237981114,
    'f0': np.float64(0.6880638641959066),
    'fnu': 0.004453689063655854
}

# Precompute mmatrices once
matrix = MatrixCalculator(A_full=True)
mmatrices = matrix.get_mmatrices()

# -------------------------------
# NumPy baseline timing
# -------------------------------
start = time.time()
nonlinear = NonLinearPowerSpectrumCalculator(mmatrices=mmatrices,
                                             kernels='fk',
                                             **kwargs)
table, table_now = nonlinear.calculate_loop_table(
    k=classy['k'],
    pklin=classy['pk'],
    cosmo=None,
    **kwargs
)
end = time.time()
print(f"NumPy backend (loop table) took {end - start:.3f} s")

# Bias parameters
b1 = 1.645
b2 = -0.46
bs2 = -4./7*(b1 - 1)
b3nl = 32./315*(b1 - 1)
alpha0 = 3
alpha2 = -28.9
alpha4 = 0.0
ctilde = 0.0
PshotP = 1. / 0.0002118763
alphashot0 = 0.08
alphashot2 = -8.1
X_Fog_pk = 1
pars = np.asarray([b1, b2, bs2, b3nl, alpha0, alpha2, alpha4,
                   ctilde, alphashot0, alphashot2, PshotP, X_Fog_pk])

qpar, qper = 1., 1.

k = np.logspace(np.log10(0.01), np.log10(0.3), num=100)

# -------------------------------
# NumPy version of compute_pkells
# -------------------------------
def compute_pkells_numpy(k, pklin, qpar, qper, pars, kwargs):
    nonlinear = NonLinearPowerSpectrumCalculator(
        mmatrices=mmatrices,
        kernels='fk',
        **kwargs
    )
    table, table_now = nonlinear.calculate_loop_table(
        k=k,
        pklin=pklin,
        cosmo=None,
        **kwargs
    )
    multipoles = RSDMultipolesPowerSpectrumCalculator(model='FOLPSD')
    P0, P2, P4 = multipoles.get_rsd_pkell(
        kobs=k, qpar=qpar, qper=qper, pars=pars,
        table=table, table_now=table_now,
        bias_scheme='folps', damping='lor'
    )
    return P0, P2, P4

# Run NumPy version
start = time.time()
P0, P2, P4 = compute_pkells_numpy(np.asarray(classy['k']),
                                  np.asarray(classy['pk']),
                                  qpar, qper, pars, kwargs)
end = time.time()
print(f"NumPy backend (compute_pkells) took {end - start:.3f} s")
