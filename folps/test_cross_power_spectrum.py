import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as host_np


AUTO_RTOL = 1.0e-10
AUTO_ATOL = 1.0e-6
SYMM_RTOL = 1.0e-11
SYMM_ATOL = 1.0e-7
BACKEND_RTOL = 5.0e-3
BACKEND_ATOL = 5.0e-2


def _assert_allclose(name, actual, expected, rtol, atol):
    actual = host_np.asarray(actual)
    expected = host_np.asarray(expected)
    diff = host_np.abs(actual - expected)
    rel = diff / host_np.maximum(host_np.abs(expected), 1.0e-300)
    max_abs = host_np.nanmax(diff)
    max_rel = host_np.nanmax(rel)
    if not host_np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(
            f"{name} failed: max_abs={max_abs:.6e}, "
            f"max_rel={max_rel:.6e}, rtol={rtol:.1e}, atol={atol:.1e}"
        )
    print(f"[check] PASS {name}: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}, rtol={rtol:.1e}, atol={atol:.1e}")


def _assert_not_allclose(name, actual, expected, rtol, atol):
    actual = host_np.asarray(actual)
    expected = host_np.asarray(expected)
    if host_np.allclose(actual, expected, rtol=rtol, atol=atol):
        diff = host_np.abs(actual - expected)
        rel = diff / host_np.maximum(host_np.abs(expected), 1.0e-300)
        raise AssertionError(
            f"{name} failed: arrays were unexpectedly close, "
            f"max_abs={host_np.nanmax(diff):.6e}, max_rel={host_np.nanmax(rel):.6e}"
        )
    diff = host_np.abs(actual - expected)
    rel = diff / host_np.maximum(host_np.abs(expected), 1.0e-300)
    print(f"[check] PASS {name}: changed, max_abs={host_np.nanmax(diff):.6e}, max_rel={host_np.nanmax(rel):.6e}")


def _assert_raises_value_error(name, func, message_parts):
    try:
        func()
    except ValueError as exc:
        message = str(exc)
        missing = [part for part in message_parts if part not in message]
        if missing:
            raise AssertionError(f"{name} raised ValueError with incomplete message: {message!r}")
        print(f"[check] PASS {name}: ValueError={message!r}")
    else:
        raise AssertionError(f"{name} failed: ValueError was not raised")


def _load_linear_pk(root):
    k, pk = host_np.loadtxt(root / "inputpkT.txt", unpack=True)
    return k, pk


def _prepare_pknow_on_target_k(k_target, pknow_result):
    if isinstance(pknow_result, tuple):
        k_pknow = host_np.asarray(pknow_result[0], dtype=host_np.float64)
        pknow_arr = host_np.asarray(pknow_result[1], dtype=host_np.float64)
    else:
        k_pknow = host_np.asarray(k_target, dtype=host_np.float64)
        pknow_arr = host_np.asarray(pknow_result, dtype=host_np.float64)

    if pknow_arr.shape[0] != k_target.shape[0] or k_pknow.shape[0] != k_target.shape[0]:
        pknow_arr = host_np.interp(k_target, k_pknow, pknow_arr)
    return pknow_arr


def _row_like(xp, k, value):
    return xp.asarray(value) + xp.zeros_like(k)


def _synthetic_interp_table(
        xp, k_eval, A_full, row_values=None, pkl=0.0, Fkoverf0=0.0,
        sigma2w=0.0, f0=0.6880638641959066):
    row_values = {} if row_values is None else dict(row_values)
    nrows = 37 if A_full else 31
    table = [_row_like(xp, k_eval, 0.0) for _ in range(nrows)]
    table[0] = _row_like(xp, k_eval, pkl)
    table[1] = _row_like(xp, k_eval, Fkoverf0)
    for idx, value in row_values.items():
        table[idx] = _row_like(xp, k_eval, value)
    table[33 if A_full else 27] = _row_like(xp, k_eval, sigma2w)
    table[-3] = _row_like(xp, k_eval, 0.0)
    table[-2] = _row_like(xp, k_eval, 0.0)
    table[-1] = xp.asarray(f0)
    return tuple(table)


def _synthetic_raw_table(
        xp, k_grid, A_full, row_values=None, pkl=0.0, Fkoverf0=0.0,
        sigma2w=0.0, sigma2=0.0, delta_sigma2=0.0, f0=0.6880638641959066):
    row_values = {} if row_values is None else dict(row_values)
    nrows = 38 if A_full else 32
    table = [_row_like(xp, k_grid, 0.0) for _ in range(nrows)]
    table[0] = xp.asarray(k_grid)
    table[1] = _row_like(xp, k_grid, pkl)
    table[2] = _row_like(xp, k_grid, Fkoverf0)
    for idx, value in row_values.items():
        table[idx + 1] = _row_like(xp, k_grid, value)
    table[(33 if A_full else 27) + 1] = xp.asarray(sigma2w)
    table[-3] = xp.asarray(sigma2)
    table[-2] = xp.asarray(delta_sigma2)
    table[-1] = xp.asarray(f0)
    return tuple(table)


def _weights_leggauss(nmu, ells):
    mu, wmu = host_np.polynomial.legendre.leggauss(2 * nmu)
    mu = mu[nmu:]
    wmu = (wmu[nmu:] + wmu[nmu - 1::-1]) / 2.0
    weighted = []
    for ell in ells:
        coeffs = [0.0] * ell + [1.0]
        weighted.append(wmu * (2 * ell + 1) * host_np.polynomial.legendre.legval(mu, coeffs))
    return mu, host_np.asarray(weighted)


def _manual_pkell(multipoles, xp, kobs, qpar, qper, pars_a, pars_b, cross_nuisance,
                  table, table_now, nmu=8, ells=(0, 2, 4), IR_resummation=True):
    muobs_host, wmu_host = _weights_leggauss(nmu, ells)
    muobs = xp.asarray(muobs_host)
    wmu = xp.asarray(wmu_host)
    jac = (qpar * qper**2)**(-1)
    kap = multipoles.k_ap(kobs[:, None], muobs, qpar, qper)
    muap = multipoles.mu_ap(muobs, qpar, qper)[None, :]
    pkmu = jac * multipoles.get_rsd_pkmu(
        kap, muap, pars_a, table, table_now,
        IR_resummation=IR_resummation, damping=None,
        pars_b=pars_b, cross_nuisance=cross_nuisance,
    )
    return xp.sum(pkmu * wmu[:, None, :], axis=-1)


def _make_params(xp, b1, b2, bs2, b3nl, nuisance=None):
    if nuisance is None:
        nuisance = (0.7, -1.3, 0.2, 0.0, 0.015, -0.45, 4800.0, 0.0)
    return xp.asarray([b1, b2, bs2, b3nl, *nuisance])


def _with_x_fog(xp, pars, x_fog):
    values = host_np.array(host_np.asarray(pars), dtype=host_np.float64)
    values[-1] = x_fog
    return xp.asarray(values)


def _build_tables(root, backend, xp, kwargs, A_full):
    from folps import MatrixCalculator, NonLinearPowerSpectrumCalculator

    k_lin, pk_lin = _load_linear_pk(root)
    pknow = None
    if backend == "jax":
        import jax.numpy as jnp
        from folps import extrapolate_pklin, get_pknow_jax

        k_extrap, pk_extrap = extrapolate_pklin(k=k_lin, pk=pk_lin)
        pknow_result = get_pknow_jax(k=jnp.asarray(k_extrap), pk=jnp.asarray(pk_extrap), h=kwargs["h"])
        pknow = jnp.asarray(_prepare_pknow_on_target_k(k_lin, pknow_result))

    matrix = MatrixCalculator(A_full=A_full, use_TNS_model=False, save_dir="output_matrices" if A_full else None)
    nonlinear = NonLinearPowerSpectrumCalculator(mmatrices=matrix.get_mmatrices(), kernels="fk", **kwargs)
    table, table_now = nonlinear.calculate_loop_table(
        k=xp.asarray(k_lin),
        pklin=xp.asarray(pk_lin),
        pknow=pknow,
        cosmo=None,
        **kwargs,
    )
    return table, table_now


def _check_linear_limit(multipoles, xp, table, table_now, pars_a, pars_b, cross_nuisance, A_full):
    del table_now
    f0 = table[-1]
    k_grid_host = host_np.array([0.01, 0.05, 0.10, 0.20])
    k_grid = xp.asarray(k_grid_host)
    pkl_grid = xp.asarray(95.0 + 11.0 * k_grid_host)
    Fkoverf0_grid = xp.asarray(0.71 + 0.23 * k_grid_host)
    linear_table = _synthetic_raw_table(
        xp, k_grid, A_full, pkl=pkl_grid, Fkoverf0=Fkoverf0_grid, f0=f0,
    )
    linear_now = _synthetic_raw_table(
        xp, k_grid, A_full, pkl=pkl_grid, Fkoverf0=Fkoverf0_grid, f0=f0,
    )
    k_eval = xp.asarray(host_np.array([0.019, 0.061, 0.123, 0.181]))[:, None]
    mu_eval = xp.asarray(host_np.array([0.0, 0.35, 0.78]))[None, :]
    got = multipoles.get_rsd_pkmu(
        k_eval, mu_eval, pars_a, linear_table, linear_now,
        IR_resummation=False, damping=None,
        pars_b=pars_b, cross_nuisance=cross_nuisance,
    )
    pkl = xp.asarray(95.0 + 11.0 * host_np.asarray(k_eval))
    fk = xp.asarray((0.71 + 0.23 * host_np.asarray(k_eval)) * host_np.asarray(f0))
    expected = (pars_a[0] + fk * mu_eval**2) * (pars_b[0] + fk * mu_eval**2) * pkl
    _assert_allclose("synthetic linear Kaiser pkmu", got, expected, rtol=1.0e-12, atol=1.0e-10)


def _check_cross_parameter_validation(multipoles, xp, table, pars_a, pars_b, A_full):
    f0 = table[-1]
    k_grid = xp.asarray(host_np.array([0.01, 0.05, 0.10, 0.20]))
    raw_table = _synthetic_raw_table(xp, k_grid, A_full, f0=f0)
    raw_now = _synthetic_raw_table(xp, k_grid, A_full, f0=f0)
    k_eval = xp.asarray(host_np.array([0.05]))[:, None]
    mu_eval = xp.asarray(host_np.array([0.40]))[None, :]

    _assert_raises_value_error(
        "missing explicit cross nuisance",
        lambda: multipoles.get_rsd_pkmu(
            k_eval, mu_eval, pars_a, raw_table, raw_now,
            IR_resummation=False, damping=None, pars_b=pars_b,
        ),
        ["cross_nuisance", "pair-level EFT and stochastic parameters", "cross-spectrum"],
    )

    interp_table = _synthetic_interp_table(xp, k_eval, A_full, f0=f0, pkl=1.7)
    _assert_raises_value_error(
        "invalid cross_nuisance length",
        lambda: multipoles.get_eft_pkmu(
            k_eval, mu_eval, pars_a, interp_table, damping=None,
            pars_b=pars_b, cross_nuisance=xp.asarray([0.0] * 9),
        ),
        ["cross_nuisance", "8", "12"],
    )
    _assert_raises_value_error(
        "invalid tracer-A parameter length",
        lambda: multipoles.get_eft_pkmu(
            k_eval, mu_eval, pars_a[:-1], interp_table, damping=None,
            pars_b=pars_b, cross_nuisance=pars_a[4:],
        ),
        ["exactly 12 values"],
    )
    _assert_raises_value_error(
        "invalid tracer-B parameter length",
        lambda: multipoles.get_eft_pkmu(
            k_eval, mu_eval, pars_a, interp_table, damping=None,
            pars_b=pars_b[:-1], cross_nuisance=pars_a[4:],
        ),
        ["exactly 12 values"],
    )

    got_8 = multipoles.get_eft_pkmu(
        k_eval, mu_eval, pars_a, interp_table, damping=None,
        pars_b=pars_b, cross_nuisance=pars_a[4:],
    )
    got_12 = multipoles.get_eft_pkmu(
        k_eval, mu_eval, pars_a, interp_table, damping=None,
        pars_b=pars_b, cross_nuisance=pars_a,
    )
    _assert_allclose("cross_nuisance 12-value extraction", got_12, got_8, rtol=AUTO_RTOL, atol=AUTO_ATOL)


def _check_synthetic_pair_contractions(multipoles, xp, table, A_full):
    k_eval = xp.asarray(host_np.array([0.025, 0.085, 0.165]))[:, None]
    mu_eval = xp.asarray(host_np.array([0.25, 0.72]))[None, :]
    f0 = table[-1]
    zero_nuisance = xp.asarray([0.0] * 8)
    pars_a = _make_params(xp, 1.7, 0.31, -0.22, 0.13, nuisance=[0.0] * 8)
    pars_b = _make_params(xp, 1.2, -0.27, 0.19, -0.08, nuisance=[0.0] * 8)
    b1_a, b2_a, bs2_a, b3nl_a = pars_a[:4]
    b1_b, b2_b, bs2_b, b3nl_b = pars_b[:4]

    def row_value(seed):
        return seed + 0.37 * k_eval

    def evaluate(row_values=None, pkl=0.0, Fkoverf0=0.0, sigma2w=0.0):
        synthetic = _synthetic_interp_table(
            xp, k_eval, A_full, row_values=row_values, pkl=pkl,
            Fkoverf0=Fkoverf0, sigma2w=sigma2w, f0=f0,
        )
        return multipoles.get_eft_pkmu(
            k_eval, mu_eval, pars_a, synthetic, damping=None,
            pars_b=pars_b, cross_nuisance=zero_nuisance,
        )

    def check_row(name, row_idx, coefficient):
        row = row_value(0.21 + 0.13 * row_idx)
        got = evaluate({row_idx: row})
        expected = coefficient(row)
        _assert_allclose(name, got, expected, rtol=AUTO_RTOL, atol=AUTO_ATOL)

    density_terms = [
        ("density Ploop_dd", 2, lambda row: b1_a * b1_b * row),
        ("density Pb1b2", 5, lambda row: (b1_a * b2_b + b2_a * b1_b) * row),
        ("density Pb1bs2", 6, lambda row: (b1_a * bs2_b + bs2_a * b1_b) * row),
        ("density Pb22", 7, lambda row: b2_a * b2_b * row),
        ("density Pb2bs2", 8, lambda row: (b2_a * bs2_b + bs2_a * b2_b) * row),
        ("density Pb2s2", 9, lambda row: bs2_a * bs2_b * row),
        ("density sigma23pkl", 10, lambda row: (b1_a * b3nl_b + b3nl_a * b1_b) * row),
    ]
    for name, row_idx, coefficient in density_terms:
        check_row(name, row_idx, coefficient)

    velocity_terms = [
        ("density-velocity Ploop_dt", 3, lambda row: f0 * mu_eval**2 * (b1_a + b1_b) * row),
        ("density-velocity Pb2t", 11, lambda row: f0 * mu_eval**2 * (b2_a + b2_b) * row),
        ("density-velocity Pbs2t", 12, lambda row: f0 * mu_eval**2 * (bs2_a + bs2_b) * row),
        ("velocity-velocity Ploop_tt", 4, lambda row: f0**2 * mu_eval**4 * row),
    ]
    for name, row_idx, coefficient in velocity_terms:
        check_row(name, row_idx, coefficient)

    a_terms = [
        ("A I1udd_1", 13, lambda row: b1_a * b1_b * f0 * mu_eval**2 * row),
        ("A I2uud_1", 14, lambda row: 0.5 * (b1_a + b1_b) * f0**2 * mu_eval**2 * row),
        ("A I2uud_2", 15, lambda row: 0.5 * (b1_a + b1_b) * f0**2 * mu_eval**4 * row),
        ("A I3uuu_2", 16, lambda row: f0**3 * mu_eval**4 * row),
        ("A I3uuu_3", 17, lambda row: f0**3 * mu_eval**6 * row),
    ]
    for name, row_idx, coefficient in a_terms:
        check_row(name, row_idx, coefficient)

    d_terms = [
        ("D2 I2uudd_1D", 18, lambda row: b1_a * b1_b * f0**2 * mu_eval**2 * row),
        ("D2 I2uudd_2D", 19, lambda row: b1_a * b1_b * f0**2 * mu_eval**4 * row),
        ("D3 I3uuud_2D", 20, lambda row: 0.5 * (b1_a + b1_b) * f0**3 * mu_eval**4 * row),
        ("D3 I3uuud_3D", 21, lambda row: 0.5 * (b1_a + b1_b) * f0**3 * mu_eval**6 * row),
        ("D4 I4uuuu_2D", 22, lambda row: f0**4 * mu_eval**4 * row),
        ("D4 I4uuuu_3D", 23, lambda row: f0**4 * mu_eval**6 * row),
        ("D4 I4uuuu_4D", 24, lambda row: f0**4 * mu_eval**8 * row),
        ("D3 I3uuud_1B", 25, lambda row: 0.5 * (b1_a + b1_b) * f0**3 * mu_eval**2 * row),
        ("D4 I4uuuu_1B", 26, lambda row: f0**4 * mu_eval**2 * row),
    ]
    for name, row_idx, coefficient in d_terms:
        check_row(name, row_idx, coefficient)

    pkl = 2.4 + 0.31 * k_eval
    Fkoverf0 = 0.62 + 0.17 * k_eval
    sigma2w = 0.41 + 0.03 * k_eval
    got_g = evaluate(pkl=pkl, Fkoverf0=Fkoverf0, sigma2w=sigma2w)
    Pdt_L = pkl * Fkoverf0
    Ptt_L = pkl * Fkoverf0**2
    expected_g = -(
        (k_eval * mu_eval * f0)**2
        * sigma2w
        * (
            b1_a * b1_b * pkl
            + (b1_a + b1_b) * f0 * mu_eval**2 * Pdt_L
            + f0**2 * mu_eval**4 * Ptt_L
        )
    )
    _assert_allclose("G cross-Kaiser polynomial", got_g, expected_g, rtol=AUTO_RTOL, atol=AUTO_ATOL)

    if not A_full:
        return

    full_a_terms = [
        ("A_full I1udd_1_b2", 27, lambda row: 0.25 * (b2_a * b1_b + b1_a * b2_b) * f0 * mu_eval**2 * row),
        ("A_full I2uud_1_b2", 28, lambda row: 0.25 * (b2_a + b2_b) * f0**2 * mu_eval**2 * row),
        ("A_full I2uud_2_b2", 29, lambda row: 0.25 * (b2_a + b2_b) * f0**2 * mu_eval**4 * row),
        ("A_full I1udd_1_bs2", 30, lambda row: 0.25 * (bs2_a * b1_b + b1_a * bs2_b) * f0 * mu_eval**2 * row),
        ("A_full I2uud_1_bs2", 31, lambda row: 0.25 * (bs2_a + bs2_b) * f0**2 * mu_eval**2 * row),
        ("A_full I2uud_2_bs2", 32, lambda row: 0.25 * (bs2_a + bs2_b) * f0**2 * mu_eval**4 * row),
    ]
    for name, row_idx, coefficient in full_a_terms:
        check_row(name, row_idx, coefficient)


def _check_cross_damping_modes(xp, A_full):
    from folps import RSDMultipolesPowerSpectrumCalculator, fog_damping

    multipoles = RSDMultipolesPowerSpectrumCalculator(model="FOLPSD")
    multipoles_eft = RSDMultipolesPowerSpectrumCalculator(model="EFT")
    k_eval = xp.asarray(host_np.array([0.055, 0.135, 0.215]))[:, None]
    mu_eval = xp.asarray(host_np.array([0.30, 0.70, 0.95]))[None, :]
    f0 = 0.6880638641959066
    sigma2w = _row_like(xp, k_eval, 8.0)
    loop_row = 9.0 + 1.7 * k_eval
    table = _synthetic_interp_table(
        xp,
        k_eval,
        A_full,
        row_values={2: loop_row},
        pkl=0.0,
        Fkoverf0=0.0,
        sigma2w=sigma2w,
        f0=f0,
    )
    pars_a = _make_params(xp, 1.70, 0.0, 0.0, 0.0, nuisance=[0.0] * 7 + [0.65])
    pars_b = _make_params(xp, 1.25, 0.0, 0.0, 0.0, nuisance=[0.0] * 7 + [2.10])
    cross_nuisance = xp.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.35])
    base_loop = pars_a[0] * pars_b[0] * loop_row

    _assert_raises_value_error(
        "invalid cross_damping_mode",
        lambda: multipoles.get_eft_pkmu(
            k_eval,
            mu_eval,
            pars_a,
            table,
            damping="lor",
            pars_b=pars_b,
            cross_nuisance=cross_nuisance,
            cross_damping_mode="invalid",
        ),
        ["cross_damping_mode", "single", "geometric"],
    )
    _assert_raises_value_error(
        "invalid FOLPSD cross damping name",
        lambda: multipoles.get_eft_pkmu(
            k_eval,
            mu_eval,
            pars_a,
            table,
            damping="gaussian",
            pars_b=pars_b,
            cross_nuisance=cross_nuisance,
            cross_damping_mode="single",
        ),
        ["cross-spectrum damping", "exp", "lor", "vdg"],
    )

    # use multipoles_eft here because multipoles with model='folpsD' always damp even when damping=None
    no_damping_single = multipoles_eft.get_eft_pkmu(
        k_eval,
        mu_eval,
        pars_a,
        table,
        damping=None,
        pars_b=pars_b,
        cross_nuisance=cross_nuisance,
        cross_damping_mode="single",
    )
    no_damping_geometric = multipoles_eft.get_eft_pkmu(
        k_eval,
        mu_eval,
        pars_a,
        table,
        damping=None,
        pars_b=pars_b,
        cross_nuisance=cross_nuisance,
        cross_damping_mode="geometric",
    )
    _assert_allclose(
        "FOLPSEFT cross damping=None mode independence",
        no_damping_geometric,
        no_damping_single,
        rtol=1.0e-13,
        atol=1.0e-10,
    )

    for damping in ("exp", "lor", "vdg"):
        kmu = k_eval * mu_eval
        W_single = fog_damping(
            (kmu, cross_nuisance[-1]), (kmu, cross_nuisance[-1]),
            f=f0, sigma2v=sigma2w, damping=damping
        )
        single = multipoles.get_eft_pkmu(
            k_eval,
            mu_eval,
            pars_a,
            table,
            damping=damping,
            pars_b=pars_b,
            cross_nuisance=cross_nuisance,
            cross_damping_mode="single",
        )
        _assert_allclose(
            f"{damping} single cross damping uses X_FoG_ab",
            single,
            W_single * base_loop,
            rtol=1.0e-12,
            atol=1.0e-10,
        )

        default_single = multipoles.get_eft_pkmu(
            k_eval,
            mu_eval,
            pars_a,
            table,
            damping=damping,
            pars_b=pars_b,
            cross_nuisance=cross_nuisance,
        )
        _assert_allclose(
            f"{damping} default cross_damping_mode is single",
            default_single,
            single,
            rtol=1.0e-13,
            atol=1.0e-10,
        )

        pars_a_alt = _with_x_fog(xp, pars_a, 4.30)
        pars_b_alt = _with_x_fog(xp, pars_b, 0.20)
        single_changed_tracer_x = multipoles.get_eft_pkmu(
            k_eval,
            mu_eval,
            pars_a_alt,
            table,
            damping=damping,
            pars_b=pars_b_alt,
            cross_nuisance=cross_nuisance,
            cross_damping_mode="single",
        )
        _assert_allclose(
            f"{damping} single ignores tracer X_FoG values",
            single_changed_tracer_x,
            single,
            rtol=1.0e-13,
            atol=1.0e-10,
        )

        cross_nuisance_alt = xp.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.60])
        single_changed_pair_x = multipoles.get_eft_pkmu(
            k_eval,
            mu_eval,
            pars_a,
            table,
            damping=damping,
            pars_b=pars_b,
            cross_nuisance=cross_nuisance_alt,
            cross_damping_mode="single",
        )
        _assert_not_allclose(
            f"{damping} single responds to X_FoG_ab",
            single_changed_pair_x,
            single,
            rtol=1.0e-6,
            atol=1.0e-8,
        )
        W_a = fog_damping(
            (kmu, pars_a[-1]), (kmu, pars_a[-1]),
            f=f0, sigma2v=sigma2w, damping=damping,
        )
        W_b = fog_damping(
            (kmu, pars_b[-1]), (kmu, pars_b[-1]),
            f=f0, sigma2v=sigma2w, damping=damping,
        )
        geometric = multipoles.get_eft_pkmu(
            k_eval,
            mu_eval,
            pars_a,
            table,
            damping=damping,
            pars_b=pars_b,
            cross_nuisance=cross_nuisance,
            cross_damping_mode="geometric",
        )
        _assert_allclose(
            f"{damping} geometric cross damping uses sqrt(W_A W_B)",
            geometric,
            xp.sqrt(W_a * W_b) * base_loop,
            rtol=1.0e-12,
            atol=1.0e-10,
        )

        geometric_changed_pair_x = multipoles.get_eft_pkmu(
            k_eval,
            mu_eval,
            pars_a,
            table,
            damping=damping,
            pars_b=pars_b,
            cross_nuisance=cross_nuisance_alt,
            cross_damping_mode="geometric",
        )
        _assert_allclose(
            f"{damping} geometric ignores X_FoG_ab",
            geometric_changed_pair_x,
            geometric,
            rtol=1.0e-13,
            atol=1.0e-10,
        )

        geometric_changed_tracer_x = multipoles.get_eft_pkmu(
            k_eval,
            mu_eval,
            pars_a_alt,
            table,
            damping=damping,
            pars_b=pars_b,
            cross_nuisance=cross_nuisance,
            cross_damping_mode="geometric",
        )
        _assert_not_allclose(
            f"{damping} geometric responds to tracer X_FoG",
            geometric_changed_tracer_x,
            geometric,
            rtol=1.0e-6,
            atol=1.0e-8,
        )

        pars_equal_a = _with_x_fog(xp, pars_a, 1.80)
        pars_equal_b = _with_x_fog(xp, pars_b, 1.80)
        W_equal = fog_damping(
            (kmu, 1.80), (kmu, 1.80),
            f=f0, sigma2v=sigma2w, damping=damping,
        )
        geometric_equal = multipoles.get_eft_pkmu(
            k_eval,
            mu_eval,
            pars_equal_a,
            table,
            damping=damping,
            pars_b=pars_equal_b,
            cross_nuisance=cross_nuisance,
            cross_damping_mode="geometric",
        )
        _assert_allclose(
            f"{damping} geometric equal-X limit",
            geometric_equal,
            W_equal * base_loop,
            rtol=1.0e-12,
            atol=1.0e-10,
        )


def _check_folpsd_cross_damping_multipole_identities(xp, table, table_now):
    from folps import RSDMultipolesPowerSpectrumCalculator

    multipoles = RSDMultipolesPowerSpectrumCalculator(model="FOLPSD")
    b1 = 1.645
    pars_a = _make_params(
        xp,
        b1,
        -0.46,
        -4.0 / 7.0 * (b1 - 1.0),
        32.0 / 315.0 * (b1 - 1.0),
        nuisance=[0.7, -1.3, 0.2, 0.0, 0.015, -0.45, 4800.0, 1.25],
    )
    b1_b = 1.10
    pars_b = _make_params(
        xp,
        b1_b,
        0.23,
        -4.0 / 7.0 * (b1_b - 1.0),
        32.0 / 315.0 * (b1_b - 1.0),
        nuisance=[0.2, -0.4, 0.1, 0.0, 0.0, 0.0, 3600.0, 2.80],
    )
    cross_nuisance = xp.asarray([0.4, -0.9, 0.15, 0.0, 0.01, -0.2, 3600.0, 1.90])
    kobs = xp.asarray(host_np.array([0.025, 0.075, 0.14, 0.19]))

    for damping in ("exp", "lor", "vdg"):
        auto = multipoles.get_rsd_pkell(
            kobs=kobs,
            qpar=1.0,
            qper=1.0,
            pars=pars_a,
            table=table,
            table_now=table_now,
            damping=damping,
            nmu=8,
            IR_resummation=True,
        )
        for mode in ("single", "geometric"):
            cross_auto = multipoles.get_rsd_pkell(
                kobs=kobs,
                qpar=1.0,
                qper=1.0,
                pars=pars_a,
                table=table,
                table_now=table_now,
                damping=damping,
                nmu=8,
                IR_resummation=True,
                pars_b=pars_a,
                cross_nuisance=pars_a[4:],
                cross_damping_mode=mode,
            )
            _assert_allclose(
                f"{damping} {mode} FOLPSD multipole auto limit",
                cross_auto,
                auto,
                rtol=AUTO_RTOL,
                atol=AUTO_ATOL,
            )

            pells_ab = multipoles.get_rsd_pkell(
                kobs=kobs,
                qpar=1.02,
                qper=0.98,
                pars=pars_a,
                table=table,
                table_now=table_now,
                damping=damping,
                nmu=8,
                IR_resummation=True,
                pars_b=pars_b,
                cross_nuisance=cross_nuisance,
                cross_damping_mode=mode,
            )
            pells_ba = multipoles.get_rsd_pkell(
                kobs=kobs,
                qpar=1.02,
                qper=0.98,
                pars=pars_b,
                table=table,
                table_now=table_now,
                damping=damping,
                nmu=8,
                IR_resummation=True,
                pars_b=pars_a,
                cross_nuisance=cross_nuisance,
                cross_damping_mode=mode,
            )
            _assert_allclose(
                f"{damping} {mode} FOLPSD multipole exchange symmetry",
                pells_ab,
                pells_ba,
                rtol=SYMM_RTOL,
                atol=SYMM_ATOL,
            )


def _check_current_auto_regression(root, backend, xp, table, table_now):
    from folps import RSDMultipolesPowerSpectrumCalculator

    b1 = 1.645
    b2 = -0.46
    bs2 = -4.0 / 7.0 * (b1 - 1.0)
    b3nl = 32.0 / 315.0 * (b1 - 1.0)
    pars = xp.asarray([
        b1, b2, bs2, b3nl,
        3.0, -28.9, 0.0, 0.0,
        0.08, -8.1, 1.0 / 0.0002118763, 1.0,
    ])
    ref_path = root / f"test_outputs_{backend}" / f"results_{backend}.npz"
    ref = host_np.load(ref_path)
    kobs = xp.asarray(ref["k"])

    multipoles = RSDMultipolesPowerSpectrumCalculator(model="FOLPSD")
    got = multipoles.get_rsd_pkell(
        kobs=kobs,
        qpar=1.0,
        qper=1.0,
        pars=pars,
        table=table,
        table_now=table_now,
        bias_scheme="folps",
        damping="lor",
        damping_method="loop",
        use_GTNS=True,
    )

    expected = host_np.asarray([ref["p0"], ref["p2"], ref["p4"]])
    _assert_allclose(f"{backend} current auto regression", got, expected, rtol=AUTO_RTOL, atol=AUTO_ATOL)


def _run_backend_case(backend, outpath):
    if backend == "jax":
        import jax

        jax.config.update("jax_enable_x64", True)
    os.environ["FOLPS_BACKEND"] = backend

    import folps as folps_mod
    from folps import RSDMultipolesPowerSpectrumCalculator

    xp = folps_mod.np
    root = Path(__file__).resolve().parent
    os.chdir(root)

    kwargs = {
        "z": 0.3,
        "h": 0.6711,
        "Omega_m": 0.3211636237981114,
        "f0": host_np.float64(0.6880638641959066),
        "fnu": 0.004453689063655854,
    }

    lrg = _make_params(xp, 1.645, -0.46, -4.0 / 7.0 * (1.645 - 1.0), 32.0 / 315.0 * (1.645 - 1.0))
    elg = _make_params(xp, 1.10, 0.23, -4.0 / 7.0 * (1.10 - 1.0), 32.0 / 315.0 * (1.10 - 1.0))
    cross_nuisance = xp.asarray([0.4, -0.9, 0.15, 0.0, 0.01, -0.2, 3600.0, 0.0])

    saved = {}
    for A_full in (False, True):
        table, table_now = _build_tables(root, backend, xp, kwargs, A_full=A_full)
        multipoles = RSDMultipolesPowerSpectrumCalculator(model="EFT")
        k_eval = xp.asarray(host_np.array([0.018, 0.054, 0.117, 0.186]))[:, None]
        mu_eval = xp.asarray(host_np.array([0.0, 0.31, 0.74]))[None, :]

        for ir_flag in (False, True):
            auto = multipoles.get_rsd_pkmu(
                k_eval, mu_eval, lrg, table, table_now,
                IR_resummation=ir_flag, damping=None,
            )
            cross_auto = multipoles.get_rsd_pkmu(
                k_eval, mu_eval, lrg, table, table_now,
                IR_resummation=ir_flag, damping=None,
                pars_b=lrg, cross_nuisance=lrg[4:],
            )
            _assert_allclose(f"A_full={A_full} pkmu auto limit IR={ir_flag}", cross_auto, auto, AUTO_RTOL, AUTO_ATOL)

            pells_auto = multipoles.get_rsd_pkell(
                kobs=xp.asarray(host_np.array([0.02, 0.07, 0.13, 0.19])),
                qpar=1.0,
                qper=1.0,
                pars=lrg,
                table=table,
                table_now=table_now,
                damping=None,
                nmu=8,
                IR_resummation=ir_flag,
            )
            pells_cross_auto = multipoles.get_rsd_pkell(
                kobs=xp.asarray(host_np.array([0.02, 0.07, 0.13, 0.19])),
                qpar=1.0,
                qper=1.0,
                pars=lrg,
                table=table,
                table_now=table_now,
                damping=None,
                nmu=8,
                IR_resummation=ir_flag,
                pars_b=lrg,
                cross_nuisance=lrg[4:],
            )
            _assert_allclose(f"A_full={A_full} multipole auto limit IR={ir_flag}",
                             pells_cross_auto, pells_auto, AUTO_RTOL, AUTO_ATOL)

        pk_ab = multipoles.get_rsd_pkmu(
            k_eval, mu_eval, lrg, table, table_now,
            IR_resummation=True, damping=None,
            pars_b=elg, cross_nuisance=cross_nuisance,
        )
        pk_ba = multipoles.get_rsd_pkmu(
            k_eval, mu_eval, elg, table, table_now,
            IR_resummation=True, damping=None,
            pars_b=lrg, cross_nuisance=cross_nuisance,
        )
        _assert_allclose(f"A_full={A_full} pkmu exchange symmetry", pk_ab, pk_ba, SYMM_RTOL, SYMM_ATOL)

        kobs = xp.asarray(host_np.array([0.025, 0.075, 0.14, 0.19]))
        pells_ab = multipoles.get_rsd_pkell(
            kobs=kobs,
            qpar=1.02,
            qper=0.98,
            pars=lrg,
            table=table,
            table_now=table_now,
            damping=None,
            nmu=8,
            IR_resummation=True,
            pars_b=elg,
            cross_nuisance=cross_nuisance,
        )
        pells_ba = multipoles.get_rsd_pkell(
            kobs=kobs,
            qpar=1.02,
            qper=0.98,
            pars=elg,
            table=table,
            table_now=table_now,
            damping=None,
            nmu=8,
            IR_resummation=True,
            pars_b=lrg,
            cross_nuisance=cross_nuisance,
        )
        _assert_allclose(f"A_full={A_full} AP multipole exchange symmetry", pells_ab, pells_ba, SYMM_RTOL, SYMM_ATOL)

        manual = _manual_pkell(
            multipoles, xp, kobs, 1.02, 0.98, lrg, elg, cross_nuisance,
            table, table_now, nmu=8, IR_resummation=True,
        )
        _assert_allclose(f"A_full={A_full} AP manual multipole integration", pells_ab, manual, AUTO_RTOL, AUTO_ATOL)

        if not A_full:
            _check_cross_parameter_validation(multipoles, xp, table, lrg, elg, A_full)
        _check_linear_limit(multipoles, xp, table, table_now, lrg, elg, xp.asarray([0.0] * 8), A_full)
        _check_synthetic_pair_contractions(multipoles, xp, table, A_full)
        _check_cross_damping_modes(xp, A_full)
        _check_folpsd_cross_damping_multipole_identities(xp, table, table_now)

        if A_full:
            _check_current_auto_regression(root, backend, xp, table, table_now)
            saved["pells_cross"] = host_np.asarray(pells_ab)
            saved["pkmu_cross"] = host_np.asarray(pk_ab)

    host_np.savez(outpath, **saved)
    print(f"[test_cross_power_spectrum:{backend}] PASS")


def _run_parent():
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="folps_cross_power_") as tmpdir:
        tmp = Path(tmpdir)
        outputs = {}
        for backend in ("numpy", "jax"):
            outpath = tmp / f"cross_{backend}.npz"
            cmd = [sys.executable, str(root / "test_cross_power_spectrum.py"),
                   "--backend-case", backend, "--out", str(outpath)]
            env = os.environ.copy()
            env["FOLPS_BACKEND"] = backend
            subprocess.run(cmd, cwd=root, env=env, check=True)
            outputs[backend] = host_np.load(outpath)

        _assert_allclose(
            "NumPy/JAX cross multipoles",
            outputs["jax"]["pells_cross"],
            outputs["numpy"]["pells_cross"],
            rtol=BACKEND_RTOL,
            atol=BACKEND_ATOL,
        )
        _assert_allclose(
            "NumPy/JAX cross pkmu",
            outputs["jax"]["pkmu_cross"],
            outputs["numpy"]["pkmu_cross"],
            rtol=BACKEND_RTOL,
            atol=BACKEND_ATOL,
        )

    print("[test_cross_power_spectrum] Overall status: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-case", choices=("numpy", "jax"))
    parser.add_argument("--out")
    args = parser.parse_args()

    if args.backend_case:
        if not args.out:
            raise ValueError("--out is required with --backend-case")
        _run_backend_case(args.backend_case, args.out)
    else:
        _run_parent()


if __name__ == "__main__":
    main()
