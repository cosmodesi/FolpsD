import os
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Force a non-interactive backend so the script works in headless environments.
matplotlib.use("Agg")

# Select NumPy backend before importing folps internals.
os.environ["FOLPS_BACKEND"] = "numpy"

from cosmo_class import run_class
from folps import (
    BispectrumCalculator,
    MatrixCalculator,
    NonLinearPowerSpectrumCalculator,
    RSDMultipolesPowerSpectrumCalculator,
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


def _print_timing_table(timing_rows: list[tuple[str, float]]) -> None:
    title = "[test_folps_numpy] Timing summary"
    step_width = max(len("Step"), max(len(step) for step, _ in timing_rows))
    time_width = len("Time [s]")

    top = f"+-{'-' * step_width}-+-{'-' * time_width}-+"
    header = f"| {'Step'.ljust(step_width)} | {'Time [s]'.rjust(time_width)} |"

    print(title)
    print(top)
    print(header)
    print(top)
    for step, seconds in timing_rows:
        print(f"| {step.ljust(step_width)} | {seconds:>{time_width}.3f} |")
    print(top)


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


def run_test_folps_numpy() -> None:
    root = Path(__file__).resolve().parent
    output_dir = root / "test_outputs_numpy"
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

    pars_pk = np.asarray(
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

    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = time.perf_counter()
        matrix = MatrixCalculator(A_full=True, save_dir='output_matrices')
        mmatrices = matrix.get_mmatrices()
        t_matrix = time.perf_counter() - t0

        nonlinear = NonLinearPowerSpectrumCalculator(mmatrices=mmatrices, kernels="fk", **kwargs)

        t0 = time.perf_counter()
        table, table_now = nonlinear.calculate_loop_table(
            k=np.asarray(classy["k"]),
            pklin=np.asarray(classy["pk"]),
            cosmo=None,
            **kwargs,
        )
        t_loop = time.perf_counter() - t0

        multipoles = RSDMultipolesPowerSpectrumCalculator(model="FOLPSD")
        t0 = time.perf_counter()
        p0, p2, p4 = multipoles.get_rsd_pkell(
            kobs=table[0],
            qpar=1.0,
            qper=1.0,
            pars=pars_pk,
            table=table,
            table_now=table_now,
            bias_scheme="folps",
            damping="lor",
        )
        t_pk = time.perf_counter() - t0

        p0_c, p2_c, p4_c = get_rsd_pkell_marg_const(
            kobs=table[0],
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
            kobs=table[0],
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

    k_ev = np.linspace(0.01, 0.2, num=40)
    k1k2_pairs = np.vstack([k_ev, k_ev]).T

    pshot_bk = 0.0
    bshot = 0.0
    c1, c2 = 0.0, 0.0
    x_fog_bk = 1.0
    f0 = nonlinear.f0
    pars_bk = [b1, b2, bs2, c1, c2, bshot, pshot_bk, x_fog_bk]

    k_pkl_pklnw = np.array([table[0], table[1], table_now[1]])
    bispectrum = BispectrumCalculator(model="FOLPSD")

    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = time.perf_counter()
        b000, b110, b220, b202, b022, b112 = bispectrum.Sugiyama_Bell(
            f=f0,
            bpars=pars_bk,
            k_pkl_pklnw=k_pkl_pklnw,
            k1k2pairs=k1k2_pairs,
            qpar=1.0,
            qper=1.0,
            precision=[10, 10, 10],
            damping="lor",
            multipoles=["B000", "B110", "B220", "B202", "B022", "B112"],
            renormalize=True,
            interpolation_method="linear",
            bias_scheme="folps",
        )
        t_bk = time.perf_counter() - t0

    _assert_finite("P0", np.asarray(p0))
    _assert_finite("P2", np.asarray(p2))
    _assert_finite("P4", np.asarray(p4))
    _assert_finite("B000", np.asarray(b000))
    _assert_finite("B202", np.asarray(b202))

    power_fig = output_dir / "power_spectrum_numpy.png"
    bispec_fig = output_dir / "bispectrum_numpy.png"
    results_npz = output_dir / "results_numpy.npz"
    _plot_power_spectrum(np.asarray(table[0]), np.asarray(p0), np.asarray(p2), np.asarray(p4), np.asarray(p0_marg), np.asarray(p2_marg), np.asarray(p4_marg), power_fig)
    _plot_bispectrum(k_ev, np.asarray(b000), np.asarray(b110), np.asarray(b220), np.asarray(b202), np.asarray(b022), np.asarray(b112), bispec_fig)

    np.savez(
        results_npz,
        k=np.asarray(table[0]),
        p0=np.asarray(p0),
        p2=np.asarray(p2),
        p4=np.asarray(p4),
        p0_marg=np.asarray(p0_marg),
        p2_marg=np.asarray(p2_marg),
        p4_marg=np.asarray(p4_marg),
        k_bis=np.asarray(k_ev),
        b000=np.asarray(b000),
        b110=np.asarray(b110),
        b220=np.asarray(b220),
        b202=np.asarray(b202),
        b022=np.asarray(b022),
        b112=np.asarray(b112),
    )

    _print_timing_table(
        [
            ("Matrix build", t_matrix),
            ("Loop table", t_loop),
            ("Power spectrum multipoles", t_pk),
            ("Bispectrum multipoles", t_bk),
        ]
    )
    print(f"  Numeric results:      {results_npz}")
    print(f"  Power spectrum figure: {power_fig}")
    print(f"  Bispectrum figure:     {bispec_fig}")


if __name__ == "__main__":
    run_test_folps_numpy()