#!/usr/bin/env python3
"""Generate reproducible figures for the FOLPS two-tracer cross-power note."""

from __future__ import annotations

import csv
import hashlib
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ["FOLPS_BACKEND"] = "numpy"
sys.dont_write_bytecode = True

_MPL_CONFIG_DIR = tempfile.TemporaryDirectory(prefix="folps_cross_power_mpl_")
os.environ["MPLCONFIGDIR"] = _MPL_CONFIG_DIR.name

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from folps import (
    MatrixCalculator,
    NonLinearPowerSpectrumCalculator,
    RSDMultipolesPowerSpectrumCalculator,
)


ELLS = (0, 2, 4)
NMU = 16
K_GRID = np.linspace(0.01, 0.30, 145)
K_DISPLAY_MAX = 0.20
MU_SLICES = np.asarray([0.0, 0.5, 1.0])
GEOM_REL_FLOOR = 1.0e-12
AUTO_RTOL = 1.0e-10
AUTO_ATOL = 1.0e-6
SYMM_RTOL = 1.0e-11
SYMM_ATOL = 1.0e-7
CANONICAL_RTOL = 1.0e-12
CANONICAL_ATOL = 1.0e-8


plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.7,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
    }
)


@dataclass
class RunContext:
    repo_root: Path
    docs_dir: Path
    figure_dir: Path
    table_dir: Path
    matrix_path: Path
    matrix_hash_before: str
    table: tuple
    table_now: tuple
    multipoles: RSDMultipolesPowerSpectrumCalculator
    pars_a: list[float]
    pars_b: list[float]
    cross_nuisance_ab: list[float]
    cosmo: dict[str, float]
    figure_paths: list[Path] = field(default_factory=list)
    csv_paths: list[Path] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def find_repo_root() -> Path:
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "folps" / "__init__.py").exists() and (
            candidate / "folps" / "inputpkT.txt"
        ).exists():
            return candidate
    raise FileNotFoundError("Run this script from within the FolpsD repository.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value: object) -> object:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if np.isnan(number):
        return "nan"
    if np.isposinf(number):
        return "inf"
    if np.isneginf(number):
        return "-inf"
    return f"{number:.17g}"


def write_rows_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: fmt(row.get(column, "")) for column in columns})


def assert_finite(name: str, array: object) -> None:
    values = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or Inf.")


def max_abs_rel(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    diff = np.abs(np.asarray(actual) - np.asarray(expected))
    rel = diff / np.maximum(np.abs(expected), 1.0e-300)
    return float(np.nanmax(diff)), float(np.nanmax(rel))


def coevolution_bk2(b1: float) -> float:
    return -2.0 / 7.0 * (b1 - 1.0)


def coevolution_btd(b1: float) -> float:
    return 23.0 / 42.0 * (b1 - 1.0)


def make_priordoc_params(
    b1: float,
    b2: float,
    bK2: float,
    btd: float,
    nuisance: list[float],
) -> list[float]:
    if len(nuisance) != 8:
        raise ValueError("nuisance must contain exactly eight values.")
    return [b1, b2, bK2, btd, *nuisance]


def baseline_parameters() -> tuple[list[float], list[float], list[float]]:
    # Illustrative tutorial values only; these are not DESI best-fit parameters.
    nuisance_a = [0.7, -1.3, 0.2, 0.0, 0.015, -0.45, 4800.0, 0.0]
    nuisance_b = [0.2, -0.4, 0.1, 0.0, 0.0, 0.0, 3600.0, 0.0]

    b1_a = 1.645
    b1_b = 1.10
    pars_a = make_priordoc_params(
        b1_a,
        0.20,
        coevolution_bk2(b1_a),
        coevolution_btd(b1_a),
        nuisance_a,
    )
    pars_b = make_priordoc_params(
        b1_b,
        -0.23,
        coevolution_bk2(b1_b),
        coevolution_btd(b1_b),
        nuisance_b,
    )

    # Pair-level cross nuisance values are chosen directly, not averaged from
    # the two auto-spectrum nuisance parameter sets.
    cross_nuisance_ab = [0.4, -0.9, 0.15, 0.0, 0.01, -0.2, 3600.0, 0.0]
    return pars_a, pars_b, cross_nuisance_ab


def build_context() -> RunContext:
    repo_root = find_repo_root()
    docs_dir = repo_root / "docs" / "cross_power"
    figure_dir = docs_dir / "figures"
    table_dir = docs_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    input_power_path = repo_root / "folps" / "inputpkT.txt"
    matrix_path = (
        repo_root
        / "folps"
        / "output_matrices"
        / "matrices_nfftlog128_Afull-True_use_TNS-False.npy"
    )
    if not matrix_path.exists():
        raise FileNotFoundError(
            "Required public matrix cache is missing: "
            f"{matrix_path.relative_to(repo_root)}"
        )
    matrix_hash_before = sha256_file(matrix_path)

    k_input, p_input = np.loadtxt(input_power_path, unpack=True)
    cosmo = {
        "z": 0.3,
        "h": 0.6711,
        "Omega_m": 0.3211636237981114,
        "f0": np.float64(0.6880638641959066),
        "fnu": 0.004453689063655854,
    }

    # Instantiating sets the public FOLPS global table schema flags. The actual
    # matrices are loaded from the existing repository cache below.
    MatrixCalculator(A_full=True, use_TNS_model=False, save_dir=None)
    mmatrices = np.load(matrix_path, allow_pickle=True).item()
    nonlinear = NonLinearPowerSpectrumCalculator(
        mmatrices=mmatrices,
        kernels="fk",
        **cosmo,
    )
    table, table_now = nonlinear.calculate_loop_table(
        k=k_input,
        pklin=p_input,
        cosmo=None,
        **cosmo,
    )

    pars_a, pars_b, cross_nuisance_ab = baseline_parameters()
    return RunContext(
        repo_root=repo_root,
        docs_dir=docs_dir,
        figure_dir=figure_dir,
        table_dir=table_dir,
        matrix_path=matrix_path,
        matrix_hash_before=matrix_hash_before,
        table=table,
        table_now=table_now,
        multipoles=RSDMultipolesPowerSpectrumCalculator(model="EFT"),
        pars_a=pars_a,
        pars_b=pars_b,
        cross_nuisance_ab=cross_nuisance_ab,
        cosmo=cosmo,
    )


def save_figure(ctx: RunContext, fig: plt.Figure, basename: str) -> None:
    for suffix in ("pdf", "png"):
        path = ctx.figure_dir / f"{basename}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        ctx.figure_paths.append(path)
    plt.close(fig)


def pkell(
    ctx: RunContext,
    pars: list[float],
    *,
    pars_b: list[float] | None = None,
    cross_nuisance: list[float] | None = None,
    kobs: np.ndarray = K_GRID,
    qpar: float = 1.0,
    qper: float = 1.0,
    ir: bool = True,
    damping: str | None = None,
    cross_damping_mode: str = "single",
    multipoles: RSDMultipolesPowerSpectrumCalculator | None = None,
) -> np.ndarray:
    calc = ctx.multipoles if multipoles is None else multipoles
    return np.asarray(
        calc.get_rsd_pkell(
            kobs=kobs,
            qpar=qpar,
            qper=qper,
            pars=pars,
            table=ctx.table,
            table_now=ctx.table_now,
            bias_scheme="priordoc",
            damping=damping,
            nmu=NMU,
            ells=ELLS,
            IR_resummation=ir,
            pars_b=pars_b,
            cross_nuisance=cross_nuisance,
            bias_scheme_b="priordoc" if pars_b is not None else None,
            cross_damping_mode=cross_damping_mode,
        )
    )


def canonical_params(ctx: RunContext, pars: list[float]) -> list[float]:
    return list(ctx.multipoles.set_bias_scheme(pars, bias_scheme="priordoc"))


def pkmu_triplet(
    ctx: RunContext,
    pars_a: list[float],
    pars_b: list[float],
    cross_nuisance: list[float],
    *,
    k_values: np.ndarray = K_GRID,
    mu_values: np.ndarray = MU_SLICES,
    table: tuple | None = None,
    table_now: tuple | None = None,
    ir: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = ctx.table if table is None else table
    table_now = ctx.table_now if table_now is None else table_now
    k2d = np.asarray(k_values)[:, None]
    mu2d = np.asarray(mu_values)[None, :]
    pars_a_canonical = canonical_params(ctx, pars_a)
    pars_b_canonical = canonical_params(ctx, pars_b)
    pk_aa = np.asarray(
        ctx.multipoles.get_rsd_pkmu(
            k2d,
            mu2d,
            pars_a_canonical,
            table,
            table_now,
            IR_resummation=ir,
            damping=None,
        )
    )
    pk_ab = np.asarray(
        ctx.multipoles.get_rsd_pkmu(
            k2d,
            mu2d,
            pars_a_canonical,
            table,
            table_now,
            IR_resummation=ir,
            damping=None,
            pars_b=pars_b_canonical,
            cross_nuisance=cross_nuisance,
        )
    )
    pk_bb = np.asarray(
        ctx.multipoles.get_rsd_pkmu(
            k2d,
            mu2d,
            pars_b_canonical,
            table,
            table_now,
            IR_resummation=ir,
            damping=None,
        )
    )
    return pk_aa, pk_ab, pk_bb


def geometric_ratio(
    pk_aa: np.ndarray,
    pk_ab: np.ndarray,
    pk_bb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    product = np.asarray(pk_aa) * np.asarray(pk_bb)
    scale = max(float(np.nanmax(np.abs(product))), 1.0)
    valid = np.isfinite(product) & np.isfinite(pk_ab) & (product > GEOM_REL_FLOOR * scale)
    geom = np.full_like(pk_ab, np.nan, dtype=float)
    r_minus_one = np.full_like(pk_ab, np.nan, dtype=float)
    geom[valid] = np.sqrt(product[valid])
    r_minus_one[valid] = pk_ab[valid] / geom[valid] - 1.0
    return geom, r_minus_one, valid


def style_axes(axes: np.ndarray | list[plt.Axes], *, xlim_display: bool = True) -> None:
    for ax in np.ravel(axes):
        ax.axvspan(K_DISPLAY_MAX, K_GRID.max(), color="0.95", zorder=-10)
        if xlim_display:
            ax.set_xlim(K_GRID.min(), K_DISPLAY_MAX)
        ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")


def make_cross_power_multipoles(ctx: RunContext) -> None:
    pell_aa = pkell(ctx, ctx.pars_a)
    pell_ab = pkell(
        ctx,
        ctx.pars_a,
        pars_b=ctx.pars_b,
        cross_nuisance=ctx.cross_nuisance_ab,
    )
    pell_bb = pkell(ctx, ctx.pars_b)
    for name, values in (
        ("Pell_AA", pell_aa),
        ("Pell_AB", pell_ab),
        ("Pell_BB", pell_bb),
    ):
        assert_finite(name, values)

    rows: list[dict[str, object]] = []
    for i, kval in enumerate(K_GRID):
        row: dict[str, object] = {"k": kval}
        for ell_index, ell in enumerate(ELLS):
            row[f"P{ell}_AA"] = pell_aa[ell_index, i]
            row[f"P{ell}_AB"] = pell_ab[ell_index, i]
            row[f"P{ell}_BB"] = pell_bb[ell_index, i]
        rows.append(row)
    columns = [
        "k",
        "P0_AA",
        "P0_AB",
        "P0_BB",
        "P2_AA",
        "P2_AB",
        "P2_BB",
        "P4_AA",
        "P4_AB",
        "P4_BB",
    ]
    path = ctx.table_dir / "cross_power_multipoles.csv"
    write_rows_csv(path, columns, rows)
    ctx.csv_paths.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.7), sharex=True)
    colors = {"AA": "#356AA0", "AB": "#208A63", "BB": "#C46D2D"}
    for ell_index, (ell, ax) in enumerate(zip(ELLS, axes)):
        ax.plot(K_GRID, K_GRID * pell_aa[ell_index], color=colors["AA"], label=r"$AA$")
        ax.plot(K_GRID, K_GRID * pell_ab[ell_index], color=colors["AB"], label=r"$AB$")
        ax.plot(K_GRID, K_GRID * pell_bb[ell_index], color=colors["BB"], label=r"$BB$")
        ax.axhline(0.0, color="0.25", lw=0.8)
        ax.set_title(rf"$\ell={ell}$")
        if ell_index == 0:
            ax.set_ylabel(r"$kP_\ell(k)\ [(h^{-1}\,\mathrm{Mpc})^2]$")
            ax.legend(frameon=False, ncols=3, loc="best")
    style_axes(axes)
    fig.suptitle("Illustrative auto- and cross-power multipoles", y=1.02)
    save_figure(ctx, fig, "cross_power_multipoles")


def make_cross_vs_geometric_mean_pkmu(ctx: RunContext) -> None:
    pk_aa, pk_ab, pk_bb = pkmu_triplet(ctx, ctx.pars_a, ctx.pars_b, ctx.cross_nuisance_ab)
    geom, r_minus_one, valid = geometric_ratio(pk_aa, pk_ab, pk_bb)
    assert_finite("pkmu_AA", pk_aa)
    assert_finite("pkmu_AB", pk_ab)
    assert_finite("pkmu_BB", pk_bb)
    finite_r = r_minus_one[np.isfinite(r_minus_one)]
    if finite_r.size == 0:
        raise ValueError("No valid geometric-mean samples were found.")
    ctx.metrics["baseline_max_abs_r_minus_one"] = float(np.nanmax(np.abs(finite_r)))
    ctx.metrics["baseline_r_min"] = float(np.nanmin(finite_r + 1.0))
    ctx.metrics["baseline_r_max"] = float(np.nanmax(finite_r + 1.0))

    rows: list[dict[str, object]] = []
    for i, kval in enumerate(K_GRID):
        for j, mu_value in enumerate(MU_SLICES):
            rows.append(
                {
                    "k": kval,
                    "mu": mu_value,
                    "P_AA": pk_aa[i, j],
                    "P_AB": pk_ab[i, j],
                    "P_BB": pk_bb[i, j],
                    "geometric_mean": geom[i, j],
                    "r_minus_one": r_minus_one[i, j],
                    "valid_geometric_mean": valid[i, j],
                }
            )
    path = ctx.table_dir / "cross_vs_geometric_mean_pkmu.csv"
    write_rows_csv(
        path,
        [
            "k",
            "mu",
            "P_AA",
            "P_AB",
            "P_BB",
            "geometric_mean",
            "r_minus_one",
            "valid_geometric_mean",
        ],
        rows,
    )
    ctx.csv_paths.append(path)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(6.9, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.08},
    )
    colors = ["#2A6FBB", "#24856B", "#B65D34"]
    for j, (mu_value, color) in enumerate(zip(MU_SLICES, colors)):
        ax_top.plot(
            K_GRID,
            K_GRID * pk_ab[:, j],
            color=color,
            label=rf"$P^{{AB}},\ \mu={mu_value:.1f}$",
        )
        ax_top.plot(
            K_GRID,
            K_GRID * geom[:, j],
            color=color,
            ls="--",
            label=rf"$\sqrt{{P^{{AA}}P^{{BB}}}},\ \mu={mu_value:.1f}$",
        )
        ax_bottom.plot(K_GRID, r_minus_one[:, j], color=color, label=rf"$\mu={mu_value:.1f}$")
    ax_top.set_ylabel(r"$kP(k,\mu)\ [(h^{-1}\,\mathrm{Mpc})^2]$")
    ax_top.legend(frameon=False, ncols=2)
    ax_bottom.axhline(0.0, color="0.25", lw=0.8)
    ax_bottom.set_ylabel(r"$r(k,\mu)-1$")
    ax_bottom.legend(frameon=False, ncols=3)
    style_axes([ax_top, ax_bottom])
    ax_top.set_xlabel("")
    ax_top.tick_params(labelbottom=False)
    save_figure(ctx, fig, "cross_vs_geometric_mean_pkmu")


def linear_only_tables(ctx: RunContext) -> tuple[tuple, tuple]:
    table = list(ctx.table)
    table_now = list(ctx.table_now)

    linear_table: list[object] = []
    for index, row in enumerate(table):
        if index in (0, 1, 2, len(table) - 1):
            linear_table.append(np.array(row, copy=True) if hasattr(row, "shape") else row)
        else:
            linear_table.append(np.zeros_like(row, dtype=float) if hasattr(row, "shape") else 0.0)

    linear_now: list[object] = []
    for index, row in enumerate(table_now):
        if index == 0:
            linear_now.append(np.array(row, copy=True))
        elif index == 1:
            linear_now.append(np.array(table[1], copy=True))
        elif index == 2:
            linear_now.append(np.array(table[2], copy=True))
        elif index == len(table_now) - 1:
            linear_now.append(row)
        else:
            linear_now.append(np.zeros_like(row, dtype=float) if hasattr(row, "shape") else 0.0)
    return tuple(linear_table), tuple(linear_now)


def zero_nuisance() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def params_with_biases(
    b1_a: float,
    b2_a: float,
    bK2_a: float,
    btd_a: float,
    b1_b: float,
    b2_b: float,
    bK2_b: float,
    btd_b: float,
    nuisance_a: list[float],
    nuisance_b: list[float],
) -> tuple[list[float], list[float]]:
    return (
        make_priordoc_params(b1_a, b2_a, bK2_a, btd_a, nuisance_a),
        make_priordoc_params(b1_b, b2_b, bK2_b, btd_b, nuisance_b),
    )


def make_cross_geometric_mean_bias_dependence(ctx: RunContext) -> None:
    b1_a, b2_a, bK2_a, btd_a = ctx.pars_a[:4]
    b1_b, b2_b, bK2_b, btd_b = ctx.pars_b[:4]
    zero = zero_nuisance()
    linear_table, linear_now = linear_only_tables(ctx)
    mu_values = np.asarray([0.5])

    cases: list[dict[str, object]] = []
    pars_linear = params_with_biases(
        b1_a, 0.0, 0.0, 0.0, b1_b, 0.0, 0.0, 0.0, zero, zero
    )
    cases.append(
        {
            "case": "linear_deterministic",
            "label": "Linear deterministic",
            "pars": pars_linear,
            "cross_nuisance": zero,
            "table": linear_table,
            "table_now": linear_now,
            "ir": False,
            "color": "#333333",
        }
    )
    cases.append(
        {
            "case": "matter_loops_linear_bias",
            "label": "Matter/RSD loops, linear bias",
            "pars": pars_linear,
            "cross_nuisance": zero,
            "table": ctx.table,
            "table_now": ctx.table_now,
            "ir": False,
            "color": "#2A6FBB",
        }
    )
    cases.append(
        {
            "case": "add_b2",
            "label": r"Add $b_2$",
            "pars": params_with_biases(
                b1_a, b2_a, 0.0, 0.0, b1_b, b2_b, 0.0, 0.0, zero, zero
            ),
            "cross_nuisance": zero,
            "table": ctx.table,
            "table_now": ctx.table_now,
            "ir": False,
            "color": "#208A63",
        }
    )
    cases.append(
        {
            "case": "add_bK2_btd",
            "label": r"Add $b_{K^2},b_{\rm td}$",
            "pars": params_with_biases(
                b1_a,
                b2_a,
                bK2_a,
                btd_a,
                b1_b,
                b2_b,
                bK2_b,
                btd_b,
                zero,
                zero,
            ),
            "cross_nuisance": zero,
            "table": ctx.table,
            "table_now": ctx.table_now,
            "ir": False,
            "color": "#C46D2D",
        }
    )
    cases.append(
        {
            "case": "add_cross_eft_stochastic",
            "label": "Add cross EFT/stochastic",
            "pars": params_with_biases(
                b1_a,
                b2_a,
                bK2_a,
                btd_a,
                b1_b,
                b2_b,
                bK2_b,
                btd_b,
                zero,
                zero,
            ),
            "cross_nuisance": ctx.cross_nuisance_ab,
            "table": ctx.table,
            "table_now": ctx.table_now,
            "ir": False,
            "color": "#8A4E9E",
        }
    )

    rows: list[dict[str, object]] = []
    fig, ax = plt.subplots(figsize=(6.9, 4.3))
    linear_residual = np.nan
    for case in cases:
        pars_a, pars_b = case["pars"]  # type: ignore[misc]
        pk_aa, pk_ab, pk_bb = pkmu_triplet(
            ctx,
            pars_a,
            pars_b,
            case["cross_nuisance"],  # type: ignore[arg-type]
            mu_values=mu_values,
            table=case["table"],  # type: ignore[arg-type]
            table_now=case["table_now"],  # type: ignore[arg-type]
            ir=bool(case["ir"]),
        )
        geom, r_minus_one, valid = geometric_ratio(pk_aa, pk_ab, pk_bb)
        finite_r = r_minus_one[np.isfinite(r_minus_one)]
        if finite_r.size == 0:
            raise ValueError(f"No valid geometric samples for case {case['case']}.")
        if case["case"] == "linear_deterministic":
            linear_residual = float(np.nanmax(np.abs(finite_r)))
        ax.plot(
            K_GRID,
            r_minus_one[:, 0],
            color=str(case["color"]),
            label=str(case["label"]),
        )
        for i, kval in enumerate(K_GRID):
            rows.append(
                {
                    "k": kval,
                    "mu": mu_values[0],
                    "case": str(case["case"]),
                    "P_AA": pk_aa[i, 0],
                    "P_AB": pk_ab[i, 0],
                    "P_BB": pk_bb[i, 0],
                    "geometric_mean": geom[i, 0],
                    "r_minus_one": r_minus_one[i, 0],
                    "valid_geometric_mean": valid[i, 0],
                    "construction": str(case["label"]),
                }
            )

    if not np.isfinite(linear_residual) or linear_residual > 1.0e-11:
        raise AssertionError(f"Linear deterministic r=1 residual is too large: {linear_residual:.6e}")
    ctx.metrics["linear_deterministic_max_abs_r_minus_one"] = linear_residual

    path = ctx.table_dir / "cross_geometric_mean_bias_dependence.csv"
    write_rows_csv(
        path,
        [
            "k",
            "mu",
            "case",
            "P_AA",
            "P_AB",
            "P_BB",
            "geometric_mean",
            "r_minus_one",
            "valid_geometric_mean",
            "construction",
        ],
        rows,
    )
    ctx.csv_paths.append(path)

    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.set_ylabel(r"$r(k,\mu=0.5)-1$")
    ax.legend(frameon=False, loc="best")
    style_axes([ax])
    save_figure(ctx, fig, "cross_geometric_mean_bias_dependence")


def vary_param(pars: list[float], index: int, delta: float) -> list[float]:
    varied = list(pars)
    varied[index] += delta
    return varied


def make_cross_nonlinear_bias_response(ctx: RunContext) -> None:
    baseline = pkell(
        ctx,
        ctx.pars_a,
        pars_b=ctx.pars_b,
        cross_nuisance=ctx.cross_nuisance_ab,
    )[0]
    assert_finite("baseline P0_AB", baseline)
    floor = max(float(np.nanmax(np.abs(baseline))) * 1.0e-10, 1.0e-300)
    if np.any(np.abs(baseline) <= floor):
        raise ValueError("Baseline P0_AB is too small for a stable fractional response.")

    variations = [
        ("b2_A_plus_0p40", r"$b_2^A+0.40$", "A", 1, 0.40, "#2A6FBB"),
        ("b2_B_plus_0p40", r"$b_2^B+0.40$", "B", 1, 0.40, "#7AA6D6"),
        ("bK2_A_plus_0p12", r"$b_{K^2}^A+0.12$", "A", 2, 0.12, "#208A63"),
        ("bK2_B_plus_0p12", r"$b_{K^2}^B+0.12$", "B", 2, 0.12, "#83BC9A"),
        ("btd_A_plus_0p20", r"$b_{\rm td}^A+0.20$", "A", 3, 0.20, "#C46D2D"),
        ("btd_B_plus_0p20", r"$b_{\rm td}^B+0.20$", "B", 3, 0.20, "#E1A06B"),
    ]

    rows: list[dict[str, object]] = []
    fig, ax = plt.subplots(figsize=(6.9, 4.2))
    for name, label, tracer, index, delta, color in variations:
        pars_a = vary_param(ctx.pars_a, index, delta) if tracer == "A" else ctx.pars_a
        pars_b = vary_param(ctx.pars_b, index, delta) if tracer == "B" else ctx.pars_b
        varied = pkell(
            ctx,
            pars_a,
            pars_b=pars_b,
            cross_nuisance=ctx.cross_nuisance_ab,
        )[0]
        response = (varied - baseline) / baseline
        assert_finite(f"nonlinear response {name}", response)
        ax.plot(K_GRID, response, color=color, label=label)
        baseline_value = ctx.pars_a[index] if tracer == "A" else ctx.pars_b[index]
        for i, kval in enumerate(K_GRID):
            rows.append(
                {
                    "k": kval,
                    "variation": name,
                    "tracer": tracer,
                    "parameter_index": index,
                    "baseline_value": baseline_value,
                    "delta": delta,
                    "P0_baseline_AB": baseline[i],
                    "P0_varied_AB": varied[i],
                    "fractional_response": response[i],
                }
            )

    path = ctx.table_dir / "cross_nonlinear_bias_response.csv"
    write_rows_csv(
        path,
        [
            "k",
            "variation",
            "tracer",
            "parameter_index",
            "baseline_value",
            "delta",
            "P0_baseline_AB",
            "P0_varied_AB",
            "fractional_response",
        ],
        rows,
    )
    ctx.csv_paths.append(path)

    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.set_ylabel(r"$(P_{0,\rm varied}^{AB}-P_{0,\rm base}^{AB})/P_{0,\rm base}^{AB}$")
    ax.legend(frameon=False, ncols=2)
    style_axes([ax])
    save_figure(ctx, fig, "cross_nonlinear_bias_response")


def make_cross_nuisance_response(ctx: RunContext) -> None:
    baseline = pkell(
        ctx,
        ctx.pars_a,
        pars_b=ctx.pars_b,
        cross_nuisance=ctx.cross_nuisance_ab,
    )
    variations = [
        ("eft", "EFT counterterms", "alpha0_ab", r"$\alpha_0^{AB}+1.0$", 0, 1.0, "#2A6FBB"),
        ("eft", "EFT counterterms", "alpha2_ab", r"$\alpha_2^{AB}+1.0$", 1, 1.0, "#208A63"),
        ("eft", "EFT counterterms", "alpha4_ab", r"$\alpha_4^{AB}+1.0$", 2, 1.0, "#C46D2D"),
        ("nlo", "NLO counterterm", "ctilde_ab", r"$\widetilde c^{AB}+2.0$", 3, 2.0, "#8A4E9E"),
        (
            "stochastic",
            "Stochastic terms",
            "alphashot0_ab",
            r"$\alpha_{\rm shot,0}^{AB}+0.05$",
            4,
            0.05,
            "#A33F57",
        ),
        (
            "stochastic",
            "Stochastic terms",
            "alphashot2_ab",
            r"$\alpha_{\rm shot,2}^{AB}+1.0$",
            5,
            1.0,
            "#6F6A2A",
        ),
    ]
    group_rows = {
        "eft": 0,
        "nlo": 1,
        "stochastic": 2,
    }

    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(12.8, 8.4),
        sharex=True,
        gridspec_kw={"hspace": 0.18, "wspace": 0.18},
    )
    for group, group_label, name, label, index, delta, color in variations:
        nuisance = list(ctx.cross_nuisance_ab)
        nuisance[index] += delta
        varied = pkell(ctx, ctx.pars_a, pars_b=ctx.pars_b, cross_nuisance=nuisance)
        delta_p = varied - baseline
        assert_finite(f"nuisance response {name}", delta_p)
        row_index = group_rows[group]
        for ell_index, ell in enumerate(ELLS):
            ax = axes[row_index, ell_index]
            ax.plot(K_GRID, K_GRID * delta_p[ell_index], color=color, label=label)
            for i, kval in enumerate(K_GRID):
                rows.append(
                    {
                        "k": kval,
                        "ell": ell,
                        "group": group,
                        "variation": name,
                        "parameter_index": index,
                        "baseline_value": ctx.cross_nuisance_ab[index],
                        "delta": delta,
                        "Pell_baseline_AB": baseline[ell_index, i],
                        "Pell_varied_AB": varied[ell_index, i],
                        "delta_Pell_AB": delta_p[ell_index, i],
                        "k_delta_Pell_AB": K_GRID[i] * delta_p[ell_index, i],
                    }
                )
    path = ctx.table_dir / "cross_nuisance_response.csv"
    write_rows_csv(
        path,
        [
            "k",
            "ell",
            "group",
            "variation",
            "parameter_index",
            "baseline_value",
            "delta",
            "Pell_baseline_AB",
            "Pell_varied_AB",
            "delta_Pell_AB",
            "k_delta_Pell_AB",
        ],
        rows,
    )
    ctx.csv_paths.append(path)

    row_titles = {
        "eft": "EFT counterterms",
        "nlo": r"NLO $\widetilde c^{AB}$ counterterm",
        "stochastic": "Stochastic terms",
    }
    for row_key, row_index in group_rows.items():
        for ell_index, ell in enumerate(ELLS):
            ax = axes[row_index, ell_index]
            ax.axhline(0.0, color="0.25", lw=0.8)
            if row_index == 0:
                ax.set_title(rf"$\ell={ell}$")
            if ell_index == 0:
                ax.set_ylabel(row_titles[row_key] + "\n" + r"$k\,\Delta P_\ell^{AB}$")
            ax.legend(frameon=False, fontsize=7, loc="best")
    style_axes(axes)
    for ax in axes[:-1, :].ravel():
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    save_figure(ctx, fig, "cross_nuisance_response")


def stable_fractional_difference(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    scale = max(float(np.nanmax(np.abs(denominator))), 1.0)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1.0e-10 * scale)
    frac = np.full_like(numerator, np.nan, dtype=float)
    frac[valid] = numerator[valid] / denominator[valid]
    return frac


def make_cross_damping_modes(ctx: RunContext) -> None:
    damping_calc = RSDMultipolesPowerSpectrumCalculator(model="FOLPSD")
    pars_a = list(ctx.pars_a)
    pars_b = list(ctx.pars_b)
    cross_nuisance = list(ctx.cross_nuisance_ab)
    pars_a[-1] = 0.30
    pars_b[-1] = 6.00
    cross_nuisance[-1] = 1.20

    pell_single = pkell(
        ctx,
        pars_a,
        pars_b=pars_b,
        cross_nuisance=cross_nuisance,
        damping="vdg",
        cross_damping_mode="single",
        multipoles=damping_calc,
    )
    pell_geometric = pkell(
        ctx,
        pars_a,
        pars_b=pars_b,
        cross_nuisance=cross_nuisance,
        damping="vdg",
        cross_damping_mode="geometric",
        multipoles=damping_calc,
    )
    assert_finite("FOLPSD cross damping single", pell_single)
    assert_finite("FOLPSD cross damping geometric", pell_geometric)
    frac = stable_fractional_difference(pell_geometric - pell_single, pell_single)

    rows: list[dict[str, object]] = []
    for i, kval in enumerate(K_GRID):
        row: dict[str, object] = {"k": kval}
        for ell_index, ell in enumerate(ELLS):
            row[f"P{ell}_single"] = pell_single[ell_index, i]
            row[f"P{ell}_geometric"] = pell_geometric[ell_index, i]
            row[f"fractional_difference_P{ell}"] = frac[ell_index, i]
        rows.append(row)
    path = ctx.table_dir / "cross_damping_modes.csv"
    write_rows_csv(
        path,
        [
            "k",
            "P0_single",
            "P0_geometric",
            "P2_single",
            "P2_geometric",
            "P4_single",
            "P4_geometric",
            "fractional_difference_P0",
            "fractional_difference_P2",
            "fractional_difference_P4",
        ],
        rows,
    )
    ctx.csv_paths.append(path)

    finite_frac = frac[np.isfinite(frac)]
    if finite_frac.size == 0:
        raise ValueError("No stable fractional differences for cross damping modes.")
    ctx.metrics["cross_damping_modes_max_abs_fractional_difference"] = float(np.nanmax(np.abs(finite_frac)))

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.8, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.12},
    )
    for ell_index, ell in enumerate(ELLS):
        axes[0, ell_index].plot(
            K_GRID,
            K_GRID * pell_single[ell_index],
            color="#2A6FBB",
            label=r"single",
        )
        axes[0, ell_index].plot(
            K_GRID,
            K_GRID * pell_geometric[ell_index],
            color="#C46D2D",
            ls="--",
            label=r"geometric",
        )
        axes[1, ell_index].plot(K_GRID, frac[ell_index], color="#333333")
        axes[1, ell_index].axhline(0.0, color="0.25", lw=0.8)
        axes[0, ell_index].set_title(rf"$\ell={ell}$")
        if ell_index == 0:
            axes[0, ell_index].set_ylabel(r"$kP_\ell^{AB}$")
            axes[1, ell_index].set_ylabel(r"$(P_{\ell,\rm geo}^{AB}-P_{\ell,\rm single}^{AB})/P_{\ell,\rm single}^{AB}$")
            axes[0, ell_index].legend(frameon=False)
    style_axes(axes)
    for ax in axes[0, :]:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    fig.suptitle(r"Illustrative FolpsD cross damping modes, VDG", y=1.01)
    save_figure(ctx, fig, "cross_damping_modes")


def make_cross_ir_resummation(ctx: RunContext) -> None:
    pell_ir = pkell(
        ctx,
        ctx.pars_a,
        pars_b=ctx.pars_b,
        cross_nuisance=ctx.cross_nuisance_ab,
        ir=True,
    )
    pell_no_ir = pkell(
        ctx,
        ctx.pars_a,
        pars_b=ctx.pars_b,
        cross_nuisance=ctx.cross_nuisance_ab,
        ir=False,
    )
    delta = pell_ir - pell_no_ir
    assert_finite("IR residuals", delta)

    rows: list[dict[str, object]] = []
    for i, kval in enumerate(K_GRID):
        row: dict[str, object] = {"k": kval}
        for ell_index, ell in enumerate(ELLS):
            row[f"P{ell}_IR_on_AB"] = pell_ir[ell_index, i]
            row[f"P{ell}_IR_off_AB"] = pell_no_ir[ell_index, i]
            row[f"delta_P{ell}_AB"] = delta[ell_index, i]
        rows.append(row)
    path = ctx.table_dir / "cross_ir_resummation.csv"
    write_rows_csv(
        path,
        [
            "k",
            "P0_IR_on_AB",
            "P0_IR_off_AB",
            "delta_P0_AB",
            "P2_IR_on_AB",
            "P2_IR_off_AB",
            "delta_P2_AB",
            "P4_IR_on_AB",
            "P4_IR_off_AB",
            "delta_P4_AB",
        ],
        rows,
    )
    ctx.csv_paths.append(path)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.8, 6.1),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.12},
    )
    for ell_index, ell in enumerate(ELLS):
        axes[0, ell_index].plot(K_GRID, K_GRID * pell_ir[ell_index], color="#2A6FBB", label="IR on")
        axes[0, ell_index].plot(
            K_GRID,
            K_GRID * pell_no_ir[ell_index],
            color="#C46D2D",
            ls="--",
            label="IR off",
        )
        axes[0, ell_index].set_title(rf"$\ell={ell}$")
        axes[1, ell_index].plot(K_GRID, K_GRID * delta[ell_index], color="#333333")
        axes[1, ell_index].axhline(0.0, color="0.25", lw=0.8)
        if ell_index == 0:
            axes[0, ell_index].set_ylabel(r"$kP_\ell^{AB}$")
            axes[1, ell_index].set_ylabel(r"$k\Delta P_\ell^{AB}$")
            axes[0, ell_index].legend(frameon=False)
    style_axes(axes)
    for ax in axes[0, :]:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    save_figure(ctx, fig, "cross_ir_resummation")


def make_cross_ap_remapping(ctx: RunContext) -> None:
    pell_fid = pkell(
        ctx,
        ctx.pars_a,
        pars_b=ctx.pars_b,
        cross_nuisance=ctx.cross_nuisance_ab,
        qpar=1.0,
        qper=1.0,
    )
    pell_ap = pkell(
        ctx,
        ctx.pars_a,
        pars_b=ctx.pars_b,
        cross_nuisance=ctx.cross_nuisance_ab,
        qpar=1.02,
        qper=0.98,
    )
    delta = pell_ap - pell_fid
    assert_finite("AP residuals", delta)

    rows: list[dict[str, object]] = []
    for i, kval in enumerate(K_GRID):
        row: dict[str, object] = {"k": kval, "qpar_fid": 1.0, "qper_fid": 1.0, "qpar_ap": 1.02, "qper_ap": 0.98}
        for ell_index, ell in enumerate(ELLS):
            row[f"P{ell}_fid_AB"] = pell_fid[ell_index, i]
            row[f"P{ell}_AP_AB"] = pell_ap[ell_index, i]
            row[f"delta_P{ell}_AP_minus_fid_AB"] = delta[ell_index, i]
        rows.append(row)
    path = ctx.table_dir / "cross_ap_remapping.csv"
    write_rows_csv(
        path,
        [
            "k",
            "qpar_fid",
            "qper_fid",
            "qpar_ap",
            "qper_ap",
            "P0_fid_AB",
            "P0_AP_AB",
            "delta_P0_AP_minus_fid_AB",
            "P2_fid_AB",
            "P2_AP_AB",
            "delta_P2_AP_minus_fid_AB",
            "P4_fid_AB",
            "P4_AP_AB",
            "delta_P4_AP_minus_fid_AB",
        ],
        rows,
    )
    ctx.csv_paths.append(path)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.8, 6.1),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.12},
    )
    for ell_index, ell in enumerate(ELLS):
        axes[0, ell_index].plot(K_GRID, K_GRID * pell_fid[ell_index], color="#2A6FBB", label="qpar=qper=1")
        axes[0, ell_index].plot(
            K_GRID,
            K_GRID * pell_ap[ell_index],
            color="#C46D2D",
            ls="--",
            label="qpar=1.02, qper=0.98",
        )
        axes[0, ell_index].set_title(rf"$\ell={ell}$")
        axes[1, ell_index].plot(K_GRID, K_GRID * delta[ell_index], color="#333333")
        axes[1, ell_index].axhline(0.0, color="0.25", lw=0.8)
        if ell_index == 0:
            axes[0, ell_index].set_ylabel(r"$kP_\ell^{AB}$")
            axes[1, ell_index].set_ylabel(r"$k\Delta P_\ell^{AB}$")
            axes[0, ell_index].legend(frameon=False, fontsize=7)
    style_axes(axes)
    for ax in axes[0, :]:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    save_figure(ctx, fig, "cross_ap_remapping")


def make_cross_implementation_residuals(ctx: RunContext) -> None:
    pell_ab = pkell(ctx, ctx.pars_a, pars_b=ctx.pars_b, cross_nuisance=ctx.cross_nuisance_ab)
    pell_ba = pkell(ctx, ctx.pars_b, pars_b=ctx.pars_a, cross_nuisance=ctx.cross_nuisance_ab)
    exchange = pell_ab - pell_ba
    exchange_max_abs, exchange_max_rel = max_abs_rel(pell_ab, pell_ba)

    pell_aa_auto = pkell(ctx, ctx.pars_a)
    pell_aa_pair = pkell(ctx, ctx.pars_a, pars_b=ctx.pars_a, cross_nuisance=ctx.pars_a[4:])
    auto_limit = pell_aa_pair - pell_aa_auto
    auto_max_abs, auto_max_rel = max_abs_rel(pell_aa_pair, pell_aa_auto)

    pars_a_canonical = canonical_params(ctx, ctx.pars_a)
    pell_aa_canonical = np.asarray(
        ctx.multipoles.get_rsd_pkell(
            kobs=K_GRID,
            qpar=1.0,
            qper=1.0,
            pars=pars_a_canonical,
            table=ctx.table,
            table_now=ctx.table_now,
            bias_scheme="folps",
            damping=None,
            nmu=NMU,
            ells=ELLS,
            IR_resummation=True,
        )
    )
    canonical_residual = pell_aa_auto - pell_aa_canonical
    canonical_max_abs, canonical_max_rel = max_abs_rel(pell_aa_auto, pell_aa_canonical)

    if not np.allclose(pell_ab, pell_ba, rtol=SYMM_RTOL, atol=SYMM_ATOL):
        raise AssertionError("Exchange symmetry check failed.")
    if not np.allclose(pell_aa_pair, pell_aa_auto, rtol=AUTO_RTOL, atol=AUTO_ATOL):
        raise AssertionError("Auto-limit recovery check failed.")
    if not np.allclose(pell_aa_auto, pell_aa_canonical, rtol=CANONICAL_RTOL, atol=CANONICAL_ATOL):
        raise AssertionError("Prior-document/canonical conversion check failed.")

    ctx.metrics["exchange_max_abs"] = exchange_max_abs
    ctx.metrics["exchange_max_rel"] = exchange_max_rel
    ctx.metrics["auto_limit_max_abs"] = auto_max_abs
    ctx.metrics["auto_limit_max_rel"] = auto_max_rel
    ctx.metrics["priordoc_canonical_max_abs"] = canonical_max_abs
    ctx.metrics["priordoc_canonical_max_rel"] = canonical_max_rel

    rows: list[dict[str, object]] = []
    for i, kval in enumerate(K_GRID):
        row: dict[str, object] = {"k": kval}
        for ell_index, ell in enumerate(ELLS):
            row[f"exchange_residual_P{ell}_AB_minus_BA"] = exchange[ell_index, i]
            row[f"auto_limit_residual_P{ell}_pair_minus_auto"] = auto_limit[ell_index, i]
            row[f"priordoc_canonical_residual_P{ell}"] = canonical_residual[ell_index, i]
        rows.append(row)
    path = ctx.table_dir / "cross_implementation_residuals.csv"
    write_rows_csv(
        path,
        [
            "k",
            "exchange_residual_P0_AB_minus_BA",
            "auto_limit_residual_P0_pair_minus_auto",
            "priordoc_canonical_residual_P0",
            "exchange_residual_P2_AB_minus_BA",
            "auto_limit_residual_P2_pair_minus_auto",
            "priordoc_canonical_residual_P2",
            "exchange_residual_P4_AB_minus_BA",
            "auto_limit_residual_P4_pair_minus_auto",
            "priordoc_canonical_residual_P4",
        ],
        rows,
    )
    ctx.csv_paths.append(path)

    residual_sets = [
        ("Exchange symmetry", exchange, "#2A6FBB"),
        ("Auto limit", auto_limit, "#208A63"),
        ("Priordoc/canonical", canonical_residual, "#C46D2D"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), sharex=True)
    floor = 1.0e-16
    for ax, (title, residual, color) in zip(axes, residual_sets):
        for ell_index, ell in enumerate(ELLS):
            ax.semilogy(
                K_GRID,
                np.maximum(np.abs(residual[ell_index]), floor),
                color=color,
                alpha=0.95 - 0.2 * ell_index,
                ls=["-", "--", ":"][ell_index],
                label=rf"$\ell={ell}$",
            )
        ax.set_title(title)
        ax.set_ylabel(r"$|\Delta P_\ell|$")
        ax.legend(frameon=False)
    style_axes(axes)
    save_figure(ctx, fig, "cross_implementation_residuals")


def write_summary(ctx: RunContext, runtime_seconds: float) -> None:
    ctx.metrics["script_runtime_seconds"] = runtime_seconds
    ctx.metrics["k_min_h_per_Mpc"] = float(K_GRID.min())
    ctx.metrics["k_max_h_per_Mpc"] = float(K_GRID.max())
    ctx.metrics["number_of_k_points"] = float(K_GRID.size)
    ctx.metrics["nmu"] = float(NMU)

    rows = [
        {
            "quantity": "maximum_exchange_symmetry_residual_abs",
            "value": ctx.metrics["exchange_max_abs"],
            "description": "max |P_ell^AB - P_ell^BA| over ell=(0,2,4) and the script k grid",
        },
        {
            "quantity": "maximum_exchange_symmetry_residual_rel",
            "value": ctx.metrics["exchange_max_rel"],
            "description": "max relative exchange-symmetry residual",
        },
        {
            "quantity": "maximum_auto_limit_residual_abs",
            "value": ctx.metrics["auto_limit_max_abs"],
            "description": "max |P_ell^AA,pair - P_ell^AA,auto|",
        },
        {
            "quantity": "maximum_auto_limit_residual_rel",
            "value": ctx.metrics["auto_limit_max_rel"],
            "description": "max relative auto-limit residual",
        },
        {
            "quantity": "maximum_priordoc_canonical_residual_abs",
            "value": ctx.metrics["priordoc_canonical_max_abs"],
            "description": "max |P_ell(priordoc input) - P_ell(equivalent canonical folps input)|",
        },
        {
            "quantity": "maximum_priordoc_canonical_residual_rel",
            "value": ctx.metrics["priordoc_canonical_max_rel"],
            "description": "max relative prior-document/canonical residual",
        },
        {
            "quantity": "maximum_baseline_abs_r_minus_one",
            "value": ctx.metrics["baseline_max_abs_r_minus_one"],
            "description": "max |P_AB/sqrt(P_AA P_BB)-1| over valid baseline pkmu samples",
        },
        {
            "quantity": "cross_damping_modes_max_abs_fractional_difference",
            "value": ctx.metrics["cross_damping_modes_max_abs_fractional_difference"],
            "description": "max |P_ell,geometric/P_ell,single - 1| over stable damping-mode samples",
        },
        {
            "quantity": "linear_deterministic_max_abs_r_minus_one",
            "value": ctx.metrics["linear_deterministic_max_abs_r_minus_one"],
            "description": "diagnostic synthetic-table linear limit check for r=1",
        },
        {
            "quantity": "script_runtime_seconds",
            "value": runtime_seconds,
            "description": "wall-clock runtime for docs/cross_power/source/make_cross_power_figures.py",
        },
        {
            "quantity": "k_min_h_per_Mpc",
            "value": K_GRID.min(),
            "description": "minimum plotted and tabulated k",
        },
        {
            "quantity": "k_max_h_per_Mpc",
            "value": K_GRID.max(),
            "description": "maximum plotted and tabulated k",
        },
        {
            "quantity": "number_of_k_points",
            "value": K_GRID.size,
            "description": "number of points in the common output k grid",
        },
        {
            "quantity": "nmu",
            "value": NMU,
            "description": "positive-mu Gauss-Legendre quadrature order used for multipoles",
        },
    ]
    path = ctx.table_dir / "cross_power_numerical_summary.csv"
    write_rows_csv(path, ["quantity", "value", "description"], rows)
    ctx.csv_paths.append(path)


def validate_csv_files(ctx: RunContext) -> None:
    for path in ctx.csv_paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for line_number, row in enumerate(reader, start=2):
                valid_geometric = row.get("valid_geometric_mean", "1") == "1"
                for key, value in row.items():
                    if value in ("", None):
                        continue
                    try:
                        number = float(value)
                    except ValueError:
                        continue
                    if np.isfinite(number):
                        continue
                    geometric_masked = (
                        path.name == "cross_vs_geometric_mean_pkmu.csv"
                        and key in {"geometric_mean", "r_minus_one"}
                        and not valid_geometric
                    )
                    damping_masked = (
                        path.name == "cross_damping_modes.csv"
                        and key.startswith("fractional_difference_P")
                    )
                    if not geometric_masked and not damping_masked:
                        raise ValueError(
                            f"Unexpected non-finite value in {path.name}:{line_number} column {key}"
                        )


def validate_outputs(ctx: RunContext) -> None:
    expected_figures = [
        "cross_power_multipoles",
        "cross_vs_geometric_mean_pkmu",
        "cross_geometric_mean_bias_dependence",
        "cross_nonlinear_bias_response",
        "cross_nuisance_response",
        "cross_damping_modes",
        "cross_ir_resummation",
        "cross_ap_remapping",
        "cross_implementation_residuals",
    ]
    expected_csvs = [
        "cross_power_multipoles.csv",
        "cross_vs_geometric_mean_pkmu.csv",
        "cross_geometric_mean_bias_dependence.csv",
        "cross_nonlinear_bias_response.csv",
        "cross_nuisance_response.csv",
        "cross_damping_modes.csv",
        "cross_ir_resummation.csv",
        "cross_ap_remapping.csv",
        "cross_implementation_residuals.csv",
        "cross_power_numerical_summary.csv",
    ]
    missing: list[str] = []
    for stem in expected_figures:
        for suffix in ("pdf", "png"):
            path = ctx.figure_dir / f"{stem}.{suffix}"
            if not path.exists() or path.stat().st_size == 0:
                missing.append(str(path.relative_to(ctx.repo_root)))
    for name in expected_csvs:
        path = ctx.table_dir / name
        if not path.exists() or path.stat().st_size == 0:
            missing.append(str(path.relative_to(ctx.repo_root)))
    if missing:
        raise FileNotFoundError("Missing generated outputs: " + ", ".join(missing))

    validate_csv_files(ctx)

    matrix_hash_after = sha256_file(ctx.matrix_path)
    if matrix_hash_after != ctx.matrix_hash_before:
        raise AssertionError("Production matrix file changed during figure generation.")
    if (ctx.repo_root / "folpsX.py").exists():
        raise AssertionError("Unexpected folpsX.py was created.")


def print_residual_report(ctx: RunContext) -> None:
    print("[cross-power figures] validation residuals")
    print(f"  exchange symmetry max abs: {ctx.metrics['exchange_max_abs']:.6e}")
    print(f"  exchange symmetry max rel: {ctx.metrics['exchange_max_rel']:.6e}")
    print(f"  auto limit max abs: {ctx.metrics['auto_limit_max_abs']:.6e}")
    print(f"  auto limit max rel: {ctx.metrics['auto_limit_max_rel']:.6e}")
    print(f"  priordoc/canonical max abs: {ctx.metrics['priordoc_canonical_max_abs']:.6e}")
    print(f"  priordoc/canonical max rel: {ctx.metrics['priordoc_canonical_max_rel']:.6e}")
    print(f"  baseline max |r-1|: {ctx.metrics['baseline_max_abs_r_minus_one']:.6e}")
    print(
        "  cross damping modes max |fractional difference|: "
        f"{ctx.metrics['cross_damping_modes_max_abs_fractional_difference']:.6e}"
    )
    print(
        "  linear deterministic max |r-1|: "
        f"{ctx.metrics['linear_deterministic_max_abs_r_minus_one']:.6e}"
    )
    print(f"  baseline r range: {ctx.metrics['baseline_r_min']:.6e} to {ctx.metrics['baseline_r_max']:.6e}")
    print("[cross-power figures] outputs")
    print(f"  figures: {len(ctx.figure_paths)} files")
    print(f"  csv tables: {len(ctx.csv_paths)} files")
    print(
        "  optional A_full comparison omitted: only the A_full=True, "
        "use_TNS_model=False production matrix cache is present."
    )


def main() -> None:
    start = time.perf_counter()
    ctx = build_context()

    make_cross_power_multipoles(ctx)
    make_cross_vs_geometric_mean_pkmu(ctx)
    make_cross_geometric_mean_bias_dependence(ctx)
    make_cross_nonlinear_bias_response(ctx)
    make_cross_nuisance_response(ctx)
    make_cross_damping_modes(ctx)
    make_cross_ir_resummation(ctx)
    make_cross_ap_remapping(ctx)
    make_cross_implementation_residuals(ctx)

    runtime_seconds = time.perf_counter() - start
    write_summary(ctx, runtime_seconds)
    validate_outputs(ctx)
    print_residual_report(ctx)
    print(f"[cross-power figures] runtime seconds: {runtime_seconds:.3f}")
    print("[cross-power figures] PASS")


if __name__ == "__main__":
    main()
