#!/usr/bin/env python3
"""
Simple end-to-end example for FOLPSpipe using the NumPy backend and the
bundled fallback linear power spectrum (`folps/inputpkT.txt`). This script
computes the nonlinear loop table and the RSD multipoles and saves the
results to `examples/run_simple_output.npz`.

Adapted from `run_folps_numpy.ipynb`.
"""
import os
import numpy as np

# Select backend before importing folps
os.environ['FOLPS_BACKEND'] = os.environ.get('FOLPS_BACKEND', 'numpy')  # 'numpy' or 'jax'

from folps import MatrixCalculator, NonLinearPowerSpectrumCalculator, RSDMultipolesPowerSpectrumCalculator, qpar_qperp


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_path = os.path.join(repo_root, 'folps', 'inputpkT.txt')

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Fallback linear power spectrum not found at {input_path}")

    k_arr, pk_arr = np.loadtxt(input_path, unpack=True)

    # 1) Precompute or load matrices (backend-independent)
    matrix = MatrixCalculator()
    mmatrices = matrix.get_mmatrices()

    # 2) Build nonlinear loop table
    nonlinear = NonLinearPowerSpectrumCalculator(mmatrices=mmatrices, kernels='fk', z=0.3)
    table, table_now = nonlinear.calculate_loop_table(k=k_arr, pklin=pk_arr, cosmo=None, z=0.3)

    # 3) Nuisance parameters (example values)
    b1 = 1.645
    b2 = -0.46
    bs2 = -4.0/7.0*(b1 - 1)
    b3nl = 32.0/315.0*(b1 - 1)
    alpha0 = 3.0
    alpha2 = -28.9
    alpha4 = 0.0
    ctilde = 0.0
    PshotP = 1.0 / 0.0002118763
    alphashot0 = 0.08
    alphashot2 = -8.1
    X_Fog_pk = 1.0
    pars = [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, PshotP, X_Fog_pk]

    qpar, qper = qpar_qperp(Omega_fid=0.31, Omega_m=0.3211636237981114, z_pk=0.3)

    # 4) Compute multipoles
    kout = np.logspace(np.log10(0.01), np.log10(0.3), num=100)
    multipoles = RSDMultipolesPowerSpectrumCalculator(model='FOLPSD')
    P0, P2, P4 = multipoles.get_rsd_pkell(kobs=kout, qpar=qpar, qper=qper, pars=pars, table=table, table_now=table_now, bias_scheme='folps', damping='lor')

    out_path = os.path.join(repo_root, 'examples', 'run_simple_output.npz')
    np.savez(out_path, k=kout, P0=P0, P2=P2, P4=P4)
    print(f"Saved results to {out_path}. Keys: k, P0, P2, P4")


if __name__ == '__main__':
    main()
