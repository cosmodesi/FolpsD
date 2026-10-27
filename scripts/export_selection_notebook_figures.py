#!/usr/bin/env python3
"""Export the existing embedded PNG outputs of the executed selection tutorial.

Run from the repository root: python scripts/export_selection_notebook_figures.py
Does not execute or change the notebook. Requires no dependencies outside stdlib.
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "selection_1loop_tutorial.ipynb"
DEST = ROOT / "docs" / "selection_1loop_theory" / "figures"
NAMES = [
    "01_linear_spectrum",
    "02_baseline_multipoles",
    "03_linear_selection",
    "04_quadratic_kernel_check",
    "05_selected_multipoles",
    "06_power_spectrum_components",
    "07_matched_tree_vs_one_loop",
    "08_selection_responses",
    "09_ir_resummation",
    "10_ap_projection",
    "11_angular_convergence",
    "12_zero_selection_regression",
    "13_jax_finite_difference"
]


def main():
    notebook = json.loads(NB.read_text(encoding="utf-8"))
    images = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            data = output.get("data", {}).get("image/png")
            if data:
                images.append("".join(data) if isinstance(data, list) else data)
    if len(images) != len(NAMES):
        raise ValueError(f"Expected {len(NAMES)} embedded PNGs; found {len(images)}; inspect notebook before exporting")
    DEST.mkdir(parents=True, exist_ok=True)
    for name, image in zip(NAMES, images):
        path = DEST / f"{name}.png"
        path.write_bytes(base64.b64decode(image))
        print(path.relative_to(ROOT), path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
