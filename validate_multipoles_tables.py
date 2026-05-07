"""
Validation: compute_redshift_space_power_multipoles_tables

Checks that the bias-decomposed multipole tables reconstruct the same P_ell(k)
as a direct call to get_rsd_pkell, for A_full=False, ctilde=0, model="EFT".
"""
import os
import numpy as np
from pathlib import Path

os.environ["FOLPS_BACKEND"] = "numpy"

from folps.folps import (
    MatrixCalculator,
    NonLinearPowerSpectrumCalculator,
    RSDMultipolesPowerSpectrumCalculator,
)

# ── 1. Linear power spectrum ──────────────────────────────────────────────────
data_path = Path(__file__).resolve().parent / "folps" / "inputpkT.txt"
k_arr, pk_arr = np.loadtxt(data_path, unpack=True)

kwargs = {
    "z": 0.3,
    "h": 0.6711,
    "Omega_m": 0.3211636237981114,
    "f0": np.float64(0.6880638641959066),
    "fnu": 0.004453689063655854,
}

# ── 2. Loop tables (A_full=False) ─────────────────────────────────────────────
print("Building M-matrices (A_full=False)...")
matrix = MatrixCalculator(A_full=False, save_dir=None)
mmatrices = matrix.get_mmatrices()

nonlinear = NonLinearPowerSpectrumCalculator(mmatrices=mmatrices, kernels="fk", **kwargs)
print("Computing loop tables...")
table, table_nw = nonlinear.calculate_loop_table(k=k_arr, pklin=pk_arr, **kwargs)
kobs = table[0]

# ── 3. Bias parameters (ctilde=0) ─────────────────────────────────────────────
b1, b2 = 1.645, -0.46
bs2    = -4.0 / 7.0 * (b1 - 1.0)
b3nl   =  32.0 / 315.0 * (b1 - 1.0)
alpha0, alpha2, alpha4 = 3.0, -28.9, 0.0
ctilde = 0.0
pshot_pk = 1.0 / 0.0002118763
alphashot0, alphashot2 = 0.08, -8.1
x_fog_pk = 0.0   # irrelevant for EFT (W=1)

pars = [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4,
        ctilde, alphashot0, alphashot2, pshot_pk, x_fog_pk]

# ── 4. Direct multipoles via get_rsd_pkell ────────────────────────────────────
qpar, qper = 1.0, 1.0
nmu = 4

calc = RSDMultipolesPowerSpectrumCalculator(model="EFT")

print("Direct get_rsd_pkell...")
pkell_direct = calc.get_rsd_pkell(
    kobs=kobs, qpar=qpar, qper=qper,
    pars=pars, table=table, table_now=table_nw,
    bias_scheme="folps", damping=None, nmu=nmu,
)
p0_direct, p2_direct, p4_direct = pkell_direct

# ── 5. Table-based multipoles ─────────────────────────────────────────────────
print("compute_redshift_space_power_multipoles_tables...")
_, p0tab, p2tab, p4tab = calc.compute_redshift_space_power_multipoles_tables(
    kobs=kobs, qpar=qpar, qper=qper,
    table=table, table_nw=table_nw,
    nmu=nmu,
)

# Coefficient vector: matches column order of the tables
# [1, b1, b1^2, b2, b1*b2, b2^2, bs2, b1*bs2, b2*bs2, bs2^2,
#  b3nl, b1*b3nl, alpha0, alpha2, alpha4,
#  PshotP*alphashot0, PshotP*alphashot2]
coeffs = np.array([
    1,
    b1, b1**2,
    b2, b1*b2, b2**2,
    bs2, b1*bs2, b2*bs2, bs2**2,
    b3nl, b1*b3nl,
    alpha0, alpha2, alpha4,
    pshot_pk * alphashot0,
    pshot_pk * alphashot2,
])

p0_table = p0tab @ coeffs
p2_table = p2tab @ coeffs
p4_table = p4tab @ coeffs

# ── 6. Compare ────────────────────────────────────────────────────────────────
def max_frac_diff(a, b, name):
    denom = np.abs(a)
    mask  = denom > 1e-10 * np.max(denom)   # ignore near-zero points
    frac  = np.max(np.abs(a[mask] - b[mask]) / denom[mask])
    status = "PASS" if frac < 1e-5 else "FAIL"
    print(f"  [{status}]  {name:5s}  max fractional diff = {frac:.2e}")
    return frac

print("\nValidation results:")
d0 = max_frac_diff(p0_direct, p0_table, "P0")
d2 = max_frac_diff(p2_direct, p2_table, "P2")
d4 = max_frac_diff(p4_direct, p4_table, "P4")

if max(d0, d2, d4) < 1e-5:
    print("\nAll multipoles agree to better than 1e-5. Validation PASSED.")
else:
    print("\nValidation FAILED — check bias decomposition.")

# ── 7. FoG damping validation (Lorentzian, X_FoG_p = 1.0) ────────────────────
x_fog_val  = 1.0
damp_val   = 'lor'

# 7a. Sanity: X_FoG_p=0 with lor gives W=1 everywhere → must match W=1 tables
print("\n── FoG sanity check: X_FoG_p=0 (lor) should equal W=1 tables ──")
_, p0tab_fog0, p2tab_fog0, p4tab_fog0 = calc.compute_redshift_space_power_multipoles_tables(
    kobs=kobs, qpar=qpar, qper=qper,
    table=table, table_nw=table_nw,
    nmu=nmu, X_FoG_p=0.0, damping=damp_val,
)
max_frac_diff(p0tab.ravel(), p0tab_fog0.ravel(), "P0")
max_frac_diff(p2tab.ravel(), p2tab_fog0.ravel(), "P2")
max_frac_diff(p4tab.ravel(), p4tab_fog0.ravel(), "P4")

# 7b. Compute FoG tables with X_FoG_p = 1.0
print(f"\n── FoG cross-validation: X_FoG_p={x_fog_val}, damping='{damp_val}' ──")
_, p0tab_fog, p2tab_fog, p4tab_fog = calc.compute_redshift_space_power_multipoles_tables(
    kobs=kobs, qpar=qpar, qper=qper,
    table=table, table_nw=table_nw,
    nmu=nmu, X_FoG_p=x_fog_val, damping=damp_val,
)
p0_fog = p0tab_fog @ coeffs
p2_fog = p2tab_fog @ coeffs
p4_fog = p4tab_fog @ coeffs

# 7c. Manual reference: call _get_pkmu_bias_table_at_mu (no FoG) at each GL node,
#     compute W_lor explicitly from sigma2w (column 27 of interp_table), apply to
#     cols 0-14, then accumulate with GL weights — independent of the FoG code path.
interped    = calc.interp_table(kobs, table, False)
sigma2w_ref = np.asarray(interped[27])   # shape (nk,)
f0_ref      = np.asarray(interped[-1])   # shape (nk,) but constant in k

nus_gl, ws_gl = np.polynomial.legendre.leggauss(2 * nmu)
L0_gl = np.polynomial.legendre.Legendre((1,))(nus_gl)
L2_gl = np.polynomial.legendre.Legendre((0, 0, 1))(nus_gl)
L4_gl = np.polynomial.legendre.Legendre((0, 0, 0, 0, 1))(nus_gl)

p0_ref_fog = np.zeros_like(kobs, dtype=float)
p2_ref_fog = np.zeros_like(kobs, dtype=float)
p4_ref_fog = np.zeros_like(kobs, dtype=float)

for ii, (nu, w) in enumerate(zip(nus_gl, ws_gl)):
    T_nofog = calc._get_pkmu_bias_table_at_mu(
        kobs, nu, table, table_nw, qpar, qper, IR_resummation=True
    )
    # W_lor(k, nu): with qpar=qper=1, kap=kobs
    W_lor = 1.0 / (1.0 + x_fog_val**2 * (f0_ref * kobs * nu)**2 * sigma2w_ref)
    T_fog_manual = T_nofog.copy()
    T_fog_manual[:, :15] *= W_lor[:, None]
    pk_nu = T_fog_manual @ coeffs
    p0_ref_fog += 0.5 * w * L0_gl[ii] * pk_nu
    p2_ref_fog += 2.5 * w * L2_gl[ii] * pk_nu
    p4_ref_fog += 4.5 * w * L4_gl[ii] * pk_nu

d0f = max_frac_diff(p0_ref_fog, p0_fog, "P0")
d2f = max_frac_diff(p2_ref_fog, p2_fog, "P2")
d4f = max_frac_diff(p4_ref_fog, p4_fog, "P4")

if max(d0f, d2f, d4f) < 1e-5:
    print(f"\nFoG validation PASSED (damping='{damp_val}', X_FoG_p={x_fog_val}).")
else:
    print(f"\nFoG validation FAILED — check W_fog application in bias table.")

# 7d. Qualitative: FoG should suppress power at high k
ratio = p0_fog[-1] / p0_direct[-1]
direction = "< 1 (suppressed)" if ratio < 1.0 else ">= 1 (unexpected)"
print(f"\nFoG suppression at k_max: P0_fog/P0_nofog = {ratio:.4f}  {direction}")

# ── 8. Direct get_rsd_pkell vs table reconstruction (FoG, lor) ───────────────
print(f"\n── Direct get_rsd_pkell vs table (X_FoG_p={x_fog_val}, damping='{damp_val}') ──")

pars_fog = [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4,
            ctilde, alphashot0, alphashot2, pshot_pk, x_fog_val]

calc_fog = RSDMultipolesPowerSpectrumCalculator(model="folpsD")
pkell_fog_direct = calc_fog.get_rsd_pkell(
    kobs=kobs, qpar=qpar, qper=qper,
    pars=pars_fog, table=table, table_now=table_nw,
    bias_scheme="folps", damping=damp_val, nmu=nmu,
)
p0_direct_fog, p2_direct_fog, p4_direct_fog = pkell_fog_direct

d0fd = max_frac_diff(p0_direct_fog, p0_fog, "P0")
d2fd = max_frac_diff(p2_direct_fog, p2_fog, "P2")
d4fd = max_frac_diff(p4_direct_fog, p4_fog, "P4")

if max(d0fd, d2fd, d4fd) < 1e-5:
    print(f"\nDirect vs table FoG validation PASSED (damping='{damp_val}', X_FoG_p={x_fog_val}).")
else:
    print(f"\nDirect vs table FoG validation FAILED.")
