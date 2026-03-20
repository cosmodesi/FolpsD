import time
import numpy as np
import jax
import jax.numpy as jnp

# Make sure JAX uses 64-bit floats
jax.config.update("jax_enable_x64", True)
import os

# Select the backend before importing folps.py
os.environ["FOLPS_BACKEND"] = "jax"  #'numpy' or 'jax'
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

# Precompute mmatrices once (don’t wrap this in jit)
matrix = MatrixCalculator(A_full=True)
mmatrices = matrix.get_mmatrices()

# NumPy baseline timing
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
print(f"NumPy backend took {end - start:.3f} s")

# Bias parameters
b1 = 1.645
b2 = -0.46
bs2 = -4./7*(b1 - 1)
b3nl = 32./315*(b1 - 1)
# EFT parameters
alpha0 = 3                 #units: [Mpc/h]^2
alpha2 = -28.9             #units: [Mpc/h]^2
alpha4 = 0.0               #units: [Mpc/h]^2
ctilde = 0.0               #units: [Mpc/h]^4
# Stochatic parameters
PshotP = 1. / 0.0002118763
alphashot0 = 0.08
alphashot2 = -8.1          #units: [Mpc/h]^2
X_Fog_pk = 1
pars = jnp.asarray([b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, PshotP, X_Fog_pk])

qpar, qper = 1., 1.


k = np.logspace(np.log10(0.01), np.log10(0.3), num=100) # array of  output k in [h/Mpc]


# -------------------------------
# JAX JIT version
# -------------------------------

@jax.jit
def compute_pkells(k, pklin, qpar, qper, pars, kwargs):
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

# Convert inputs to JAX arrays
k_jax = jnp.asarray(classy['k'])
pk_jax = jnp.asarray(classy['pk'])
qpar, qper = 1.0, 1.0   # example values
# pars = {}               # fill in with your actual pars dict

# First run (includes compilation)
start = time.time()
P0, P2, P4 = compute_pkells(k_jax, pk_jax, qpar, qper, pars, kwargs)
P0.block_until_ready()
end = time.time()
print(f"JAX JIT first run (compile+exec) took {end - start:.3f} s")

# Second run (cached, fast)
start = time.time()
P0, P2, P4 = compute_pkells(k_jax, pk_jax, qpar, qper, pars, kwargs)
P0.block_until_ready()
end = time.time()
print(f"JAX JIT cached run took {end - start:.3f} s")
