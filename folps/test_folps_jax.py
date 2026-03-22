import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Make JAX use double precision to match the NumPy pipeline behavior.
jax.config.update("jax_enable_x64", True)

# Force a non-interactive backend so the script works in headless environments.
matplotlib.use("Agg")

# Select JAX backend before importing folps internals.
os.environ["FOLPS_BACKEND"] = "jax"

from cosmo_class import run_class
from folps import (
    BispectrumCalculator,
    MatrixCalculator,
    NonLinearPowerSpectrumCalculator,
    RSDMultipolesPowerSpectrumCalculator,
    extrapolate_pklin,
    get_pknow_jax,
    get_rsd_pkell_marg_const,
    get_rsd_pkell_marg_derivatives,
)


def _load_linear_pk() -> dict:
    """Load linear P(k) either from CLASS or from the local fallback file."""
    try:
        from classy import Class as _Class  # noqa: F401

        return run_class(
            h=0.6711,
            ombh2=0.022,
            omch2=0.122,
            omnuh2=0.0006442,
            As=2e-9,
            ns=0.965,
            z=0.3,
            z_scale=[0.97],
            N_ur=2.0328,
            khmin=0.0001,
            khmax=2.0,
            nbk=1000,
            spectra="cb",
        )
    except Exception:
        data_path = Path(__file__).resolve().parent / "inputpkT.txt"
        k_arr, pk_arr = np.loadtxt(data_path, unpack=True)
        return {"k": k_arr, "pk": pk_arr}


def _assert_finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")


def _extract_pknow_array(pknow_result):
    """Accept either get_pknow_jax outputs: (k, pnow) or pnow-only."""
    if isinstance(pknow_result, tuple):
        if len(pknow_result) != 2:
            raise ValueError("Unexpected tuple format for pknow result.")
        return pknow_result[1]
    return pknow_result


def _prepare_pknow_on_target_k(k_target: np.ndarray, pknow_result) -> jnp.ndarray:
    """Return pknow sampled on k_target to avoid shape mismatches in FOLPS internals."""
    if isinstance(pknow_result, tuple):
        if len(pknow_result) != 2:
            raise ValueError("Unexpected tuple format for pknow result.")
        k_pknow = np.asarray(pknow_result[0], dtype=np.float64)
        pknow_arr = np.asarray(pknow_result[1], dtype=np.float64)
    else:
        k_pknow = np.asarray(k_target, dtype=np.float64)
        pknow_arr = np.asarray(pknow_result, dtype=np.float64)

    k_target = np.asarray(k_target, dtype=np.float64)
    if pknow_arr.shape[0] != k_target.shape[0] or k_pknow.shape[0] != k_target.shape[0]:
        pknow_arr = np.interp(k_target, k_pknow, pknow_arr)

    return jnp.asarray(pknow_arr)


def _print_timing_table(rows: list[tuple[str, str]]) -> None:
    title = "[test_folps_jax] JAX JIT timing summary"
    metric_width = max(len("Metric"), max(len(metric) for metric, _ in rows))
    value_width = max(len("Value"), max(len(value) for _, value in rows))

    top = f"+-{'-' * metric_width}-+-{'-' * value_width}-+"
    header = f"| {'Metric'.ljust(metric_width)} | {'Value'.rjust(value_width)} |"

    print(title)
    print(top)
    print(header)
    print(top)
    for metric, value in rows:
        print(f"| {metric.ljust(metric_width)} | {value.rjust(value_width)} |")
    print(top)


def _jit_speedup(first_seconds: float, cached_seconds: float) -> tuple[float, float]:
    if cached_seconds <= 0:
        return np.inf, 100.0
    speedup = first_seconds / cached_seconds
    gain = (1.0 - cached_seconds / first_seconds) * 100.0 if first_seconds > 0 else 0.0
    return speedup, gain


def _plot_power_spectrum(
    k: np.ndarray,
    p0: np.ndarray,
    p2: np.ndarray,
    p4: np.ndarray,
    p0_marg: np.ndarray,
    p2_marg: np.ndarray,
    p4_marg: np.ndarray,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlabel(r"$k\, [h\, \mathrm{Mpc}^{-1}]$", fontsize=14)
    ax.set_ylabel(r"$k\, P_{\ell}(k)\, [h^{-1}\, \mathrm{Mpc}]^2$", fontsize=14)

    ax.plot(k, k * p0, color="navy", ls="-", label=r"$\ell = 0$")
    ax.plot(k, k * p2, color="maroon", ls="-", label=r"$\ell = 2$")
    ax.plot(k, k * p4, color="darkgreen", ls="-", label=r"$\ell = 4$")

    ax.plot(k, k * p0_marg, color="navy", ls=":", lw=3, label=r"$\ell = 0$ (marginalized)")
    ax.plot(k, k * p2_marg, color="maroon", ls=":", lw=3)
    ax.plot(k, k * p4_marg, color="darkgreen", ls=":", lw=3)

    ax.set_xlim([k[0], 0.2])
    ax.set_ylim([-250, 1900])
    leg = ax.legend(loc="best")
    leg.get_frame().set_linewidth(0.0)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def _plot_bispectrum(
    k_ev: np.ndarray,
    b000: np.ndarray,
    b110: np.ndarray,
    b220: np.ndarray,
    b202: np.ndarray,
    b022: np.ndarray,
    b112: np.ndarray,
    outpath: Path,
) -> None:
    xmax = 0.2
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].set_ylabel(r"$k^2\, B(k,k)\, [h^{-1}\, \mathrm{Mpc}]^4$", fontsize=14)
    axs[0].plot(k_ev, k_ev**2 * b000, label="B000", ls="-", color="red")
    axs[0].plot(k_ev, k_ev**2 * b110, label="B110", ls="-", color="blue")
    axs[0].plot(k_ev, k_ev**2 * b220, label="B220", ls="-", color="green")
    axs[0].set_xlim([0, xmax])
    axs[0].legend(fontsize=12, loc="best")
    axs[0].set_title("B_l1l2L * H_l1l2L", fontsize=16, pad=15)

    axs[1].set_xlabel(r"$k\, [h\, \mathrm{Mpc}^{-1}]$", fontsize=14)
    axs[1].set_ylabel(r"$k^2\, B(k,k)\, [h^{-1}\, \mathrm{Mpc}]^4$", fontsize=14)
    axs[1].plot(k_ev, k_ev**2 * b202, label="B202", ls="-", color="red")
    axs[1].plot(k_ev, k_ev**2 * b022, label="B022", ls="--", color="green")
    axs[1].plot(k_ev, k_ev**2 * b112, label="B112", ls="-", color="blue")
    axs[1].set_xlim([0, xmax])
    axs[1].legend(fontsize=12, loc="best")
    axs[1].set_title("B_l1l2L * H_l1l2L", fontsize=16, pad=15)

    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def run_test_folps_jax() -> None:
    root = Path(__file__).resolve().parent
    output_dir = root / "test_outputs_jax"
    output_dir.mkdir(parents=True, exist_ok=True)

    classy = _load_linear_pk()

    kwargs = {
        "z": 0.3,
        "h": 0.6711,
        "Omega_m": 0.3211636237981114,
        "f0": np.float64(0.6880638641959066),
        "fnu": 0.004453689063655854,
    }

    b1 = 1.645
    b2 = -0.46
    bs2 = -4.0 / 7.0 * (b1 - 1.0)
    b3nl = 32.0 / 315.0 * (b1 - 1.0)
    alpha0, alpha2, alpha4 = 3.0, -28.9, 0.0
    ctilde = 0.0
    pshot_pk = 1.0 / 0.0002118763
    alphashot0, alphashot2 = 0.08, -8.1
    x_fog_pk = 1.0

    pars_pk = jnp.asarray(
        [
            b1,
            b2,
            bs2,
            b3nl,
            alpha0,
            alpha2,
            alpha4,
            ctilde,
            alphashot0,
            alphashot2,
            pshot_pk,
            x_fog_pk,
        ]
    )

    k_jax = jnp.asarray(classy["k"])
    pk_jax = jnp.asarray(classy["pk"])
    k_np = np.asarray(classy["k"], dtype=np.float64)

    t0 = time.perf_counter()
    k_extrap, pk_extrap = extrapolate_pklin(k=np.asarray(classy["k"]), pk=np.asarray(classy["pk"]))
    pknow_result = get_pknow_jax(k=jnp.asarray(k_extrap), pk=jnp.asarray(pk_extrap), h=kwargs["h"])
    pknow_jax = _prepare_pknow_on_target_k(k_np, pknow_result)
    t_pknow = time.perf_counter() - t0

    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = time.perf_counter()
        matrix = MatrixCalculator(A_full=True, save_dir='output_matrices')
        mmatrices = matrix.get_mmatrices()
        t_matrix = time.perf_counter() - t0

        nonlinear = NonLinearPowerSpectrumCalculator(mmatrices=mmatrices, kernels="fk", **kwargs)

        # Setup run used for marginalization terms and bispectrum inputs.
        t0 = time.perf_counter()
        table, table_now = nonlinear.calculate_loop_table(
            k=k_jax,
            pklin=pk_jax,
            pknow=pknow_jax,
            cosmo=None,
            **kwargs,
        )
        t_loop_setup = time.perf_counter() - t0

        p0_c, p2_c, p4_c = get_rsd_pkell_marg_const(
            kobs=k_jax,
            qpar=1.0,
            qper=1.0,
            pars=pars_pk,
            table=table,
            table_now=table_now,
            bias_scheme="folps",
            damping="lor",
            model="FOLPSD",
        )
        p0_i, p2_i, p4_i = get_rsd_pkell_marg_derivatives(
            kobs=k_jax,
            qpar=1.0,
            qper=1.0,
            pars=pars_pk,
            table=table,
            table_now=table_now,
            bias_scheme="folps",
            damping="lor",
            model="FOLPSD",
        )

    p0_marg = p0_c + (alpha0 * p0_i[0] + alpha2 * p0_i[1] + alpha4 * p0_i[2] + alphashot0 * p0_i[3] + alphashot2 * p0_i[4])
    p2_marg = p2_c + (alpha0 * p2_i[0] + alpha2 * p2_i[1] + alpha4 * p2_i[2] + alphashot0 * p2_i[3] + alphashot2 * p2_i[4])
    p4_marg = p4_c + (alpha0 * p4_i[0] + alpha2 * p4_i[1] + alpha4 * p4_i[2] + alphashot0 * p4_i[3] + alphashot2 * p4_i[4])

    @jax.jit
    def compute_pkells_jit(k, pklin, pknow, qpar, qper, pars, kw):
        nonlinear_local = NonLinearPowerSpectrumCalculator(
            mmatrices=mmatrices,
            kernels="fk",
            **kw,
        )
        table_local, table_now_local = nonlinear_local.calculate_loop_table(
            k=k,
            pklin=pklin,
            pknow=pknow,
            cosmo=None,
            **kw,
        )
        multipoles = RSDMultipolesPowerSpectrumCalculator(model="FOLPSD")
        p0_local, p2_local, p4_local = multipoles.get_rsd_pkell(
            kobs=k,
            qpar=qpar,
            qper=qper,
            pars=pars,
            table=table_local,
            table_now=table_now_local,
            bias_scheme="folps",
            damping="lor",
        )
        return p0_local, p2_local, p4_local

    t0 = time.perf_counter()
    p0_jit_1, p2_jit_1, p4_jit_1 = compute_pkells_jit(k_jax, pk_jax, pknow_jax, 1.0, 1.0, pars_pk, kwargs)
    p0_jit_1 = jax.block_until_ready(p0_jit_1)
    p2_jit_1 = jax.block_until_ready(p2_jit_1)
    p4_jit_1 = jax.block_until_ready(p4_jit_1)
    t_pk_jit_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    p0_jit_2, p2_jit_2, p4_jit_2 = compute_pkells_jit(k_jax, pk_jax, pknow_jax, 1.0, 1.0, pars_pk, kwargs)
    p0_jit_2 = jax.block_until_ready(p0_jit_2)
    p2_jit_2 = jax.block_until_ready(p2_jit_2)
    p4_jit_2 = jax.block_until_ready(p4_jit_2)
    t_pk_jit_cached = time.perf_counter() - t0

    pk_speedup, pk_gain = _jit_speedup(t_pk_jit_first, t_pk_jit_cached)

    k_ev = np.linspace(0.01, 0.2, num=40)
    k1k2_pairs = np.vstack([k_ev, k_ev]).T
    k1k2_pairs_jax = jnp.asarray(k1k2_pairs)

    pshot_bk = 0.0
    bshot = 0.0
    c1, c2 = 0.0, 0.0
    x_fog_bk = 1.0
    f0 = nonlinear.f0
    pars_bk = jnp.asarray([b1, b2, bs2, c1, c2, bshot, pshot_bk, x_fog_bk])

    k_pkl_pklnw = jnp.asarray(np.array([np.asarray(table[0]), np.asarray(table[1]), np.asarray(table_now[1])]))

    @jax.jit
    def compute_bispectrum_jit(bpars, f, qpar, qper):
        bispectrum = BispectrumCalculator(model="FOLPSD")
        return bispectrum.Sugiyama_Bell(
            f=f,
            bpars=bpars,
            k_pkl_pklnw=k_pkl_pklnw,
            k1k2pairs=k1k2_pairs_jax,
            qpar=qpar,
            qper=qper,
            precision=[10, 10, 10],
            damping="lor",
            multipoles=["B000", "B110", "B220", "B202", "B022", "B112"],
            renormalize=True,
            interpolation_method="linear",
            bias_scheme="folps",
        )

    t0 = time.perf_counter()
    b000_1, b110_1, b220_1, b202_1, b022_1, b112_1 = compute_bispectrum_jit(pars_bk, f0, 1.0, 1.0)
    b000_1 = jax.block_until_ready(b000_1)
    b110_1 = jax.block_until_ready(b110_1)
    b220_1 = jax.block_until_ready(b220_1)
    b202_1 = jax.block_until_ready(b202_1)
    b022_1 = jax.block_until_ready(b022_1)
    b112_1 = jax.block_until_ready(b112_1)
    t_bk_jit_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    b000_2, b110_2, b220_2, b202_2, b022_2, b112_2 = compute_bispectrum_jit(pars_bk, f0, 1.0, 1.0)
    b000_2 = jax.block_until_ready(b000_2)
    b110_2 = jax.block_until_ready(b110_2)
    b220_2 = jax.block_until_ready(b220_2)
    b202_2 = jax.block_until_ready(b202_2)
    b022_2 = jax.block_until_ready(b022_2)
    b112_2 = jax.block_until_ready(b112_2)
    t_bk_jit_cached = time.perf_counter() - t0

    bk_speedup, bk_gain = _jit_speedup(t_bk_jit_first, t_bk_jit_cached)

    p0_plot = np.asarray(p0_jit_2)
    p2_plot = np.asarray(p2_jit_2)
    p4_plot = np.asarray(p4_jit_2)

    b000_plot = np.asarray(b000_2)
    b110_plot = np.asarray(b110_2)
    b220_plot = np.asarray(b220_2)
    b202_plot = np.asarray(b202_2)
    b022_plot = np.asarray(b022_2)
    b112_plot = np.asarray(b112_2)

    _assert_finite("P0 (JIT)", p0_plot)
    _assert_finite("P2 (JIT)", p2_plot)
    _assert_finite("P4 (JIT)", p4_plot)
    _assert_finite("B000 (JIT)", b000_plot)
    _assert_finite("B202 (JIT)", b202_plot)

    power_fig = output_dir / "power_spectrum_jax.png"
    bispec_fig = output_dir / "bispectrum_jax.png"
    results_npz = output_dir / "results_jax.npz"
    _plot_power_spectrum(
        np.asarray(k_jax),
        p0_plot,
        p2_plot,
        p4_plot,
        np.asarray(p0_marg),
        np.asarray(p2_marg),
        np.asarray(p4_marg),
        power_fig,
    )
    _plot_bispectrum(
        k_ev,
        b000_plot,
        b110_plot,
        b220_plot,
        b202_plot,
        b022_plot,
        b112_plot,
        bispec_fig,
    )

    np.savez(
        results_npz,
        k=np.asarray(k_jax),
        p0=p0_plot,
        p2=p2_plot,
        p4=p4_plot,
        p0_marg=np.asarray(p0_marg),
        p2_marg=np.asarray(p2_marg),
        p4_marg=np.asarray(p4_marg),
        k_bis=np.asarray(k_ev),
        b000=b000_plot,
        b110=b110_plot,
        b220=b220_plot,
        b202=b202_plot,
        b022=b022_plot,
        b112=b112_plot,
    )

    _print_timing_table(
        [
            ("Pnow precompute", f"{t_pknow:.3f} s"),
            ("Matrix build", f"{t_matrix:.3f} s"),
            ("Loop table setup", f"{t_loop_setup:.3f} s"),
            ("PK JIT first run (compile+exec)", f"{t_pk_jit_first:.3f} s"),
            ("PK JIT cached run", f"{t_pk_jit_cached:.3f} s"),
            ("PK speedup", f"{pk_speedup:.2f}x"),
            ("PK improvement", f"{pk_gain:.1f}%"),
            ("BK JIT first run (compile+exec)", f"{t_bk_jit_first:.3f} s"),
            ("BK JIT cached run", f"{t_bk_jit_cached:.3f} s"),
            ("BK speedup", f"{bk_speedup:.2f}x"),
            ("BK improvement", f"{bk_gain:.1f}%"),
        ]
    )
    print(f"  Numeric results:      {results_npz}")
    print(f"  Power spectrum figure: {power_fig}")
    print(f"  Bispectrum figure:     {bispec_fig}")


if __name__ == "__main__":
    run_test_folps_jax()