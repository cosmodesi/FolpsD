import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def _format_bool(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _print_table(rows):
    headers = ("Quantity", "Max abs", "Max rel", "Allclose")
    widths = [
        max(len(headers[0]), max(len(r[0]) for r in rows)),
        max(len(headers[1]), max(len(r[1]) for r in rows)),
        max(len(headers[2]), max(len(r[2]) for r in rows)),
        max(len(headers[3]), max(len(r[3]) for r in rows)),
    ]

    sep = f"+-{'-' * widths[0]}-+-{'-' * widths[1]}-+-{'-' * widths[2]}-+-{'-' * widths[3]}-+"
    hdr = f"| {headers[0].ljust(widths[0])} | {headers[1].rjust(widths[1])} | {headers[2].rjust(widths[2])} | {headers[3].rjust(widths[3])} |"

    print(sep)
    print(hdr)
    print(sep)
    for q, a, r, ok in rows:
        print(f"| {q.ljust(widths[0])} | {a.rjust(widths[1])} | {r.rjust(widths[2])} | {ok.rjust(widths[3])} |")
    print(sep)


def _max_rel_diff(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    den = np.maximum(np.abs(a), eps)
    return float(np.max(np.abs(a - b) / den))


def _align_to_reference(k_ref: np.ndarray, k_other: np.ndarray, y_other: np.ndarray) -> np.ndarray:
    if k_ref.shape == k_other.shape and np.allclose(k_ref, k_other, rtol=0.0, atol=0.0):
        return y_other
    return np.interp(k_ref, k_other, y_other)


def _compare_array(name: str, ref: np.ndarray, other: np.ndarray, rtol: float, atol: float):
    max_abs = float(np.max(np.abs(ref - other)))
    max_rel = _max_rel_diff(ref, other)
    ok = bool(np.allclose(ref, other, rtol=rtol, atol=atol))
    return (name, f"{max_abs:.3e}", f"{max_rel:.3e}", _format_bool(ok), ok)


def _run_script(script_path: Path):
    cmd = [sys.executable, str(script_path)]
    subprocess.run(cmd, cwd=str(script_path.parent), check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compare numeric outputs from test_folps_numpy.py and test_folps_jax.py"
    )
    parser.add_argument("--rtol", type=float, default=5e-3, help="Relative tolerance for equivalence")
    parser.add_argument("--atol", type=float, default=5e-2, help="Absolute tolerance for equivalence")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Do not re-run backend tests; only compare existing NPZ files",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    numpy_script = root / "test_folps_numpy.py"
    jax_script = root / "test_folps_jax.py"

    numpy_npz = root / "test_outputs_numpy" / "results_numpy.npz"
    jax_npz = root / "test_outputs_jax" / "results_jax.npz"

    if not args.skip_run:
        print("Running NumPy backend test...")
        _run_script(numpy_script)
        print("Running JAX backend test...")
        _run_script(jax_script)

    if not numpy_npz.exists() or not jax_npz.exists():
        raise FileNotFoundError(
            "Missing NPZ results. Expected files:\n"
            f"  - {numpy_npz}\n"
            f"  - {jax_npz}\n"
            "Run without --skip-run or execute both backend tests first."
        )

    np_res = np.load(numpy_npz)
    jx_res = np.load(jax_npz)

    k_np = np.asarray(np_res["k"])
    k_jx = np.asarray(jx_res["k"])
    k_bis_np = np.asarray(np_res["k_bis"])
    k_bis_jx = np.asarray(jx_res["k_bis"])

    rows = []
    pass_flags = []

    for name in ("p0", "p2", "p4", "p0_marg", "p2_marg", "p4_marg"):
        ref = np.asarray(np_res[name])
        other = _align_to_reference(k_np, k_jx, np.asarray(jx_res[name]))
        q, a, r, ok_text, ok = _compare_array(name, ref, other, rtol=args.rtol, atol=args.atol)
        rows.append((q, a, r, ok_text))
        pass_flags.append(ok)

    for name in ("b000", "b110", "b220", "b202", "b022", "b112"):
        ref = np.asarray(np_res[name])
        other = _align_to_reference(k_bis_np, k_bis_jx, np.asarray(jx_res[name]))
        q, a, r, ok_text, ok = _compare_array(name, ref, other, rtol=args.rtol, atol=args.atol)
        rows.append((name, a, r, ok_text))
        pass_flags.append(ok)

    print("\n[compare_folps_numpy_vs_jax] Equivalence report")
    print(f"Tolerances: atol={args.atol:.2e}, rtol={args.rtol:.2e}")
    _print_table(rows)

    passed = all(pass_flags)
    print(f"Overall status: {_format_bool(passed)}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
