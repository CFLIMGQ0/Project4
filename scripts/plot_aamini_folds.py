#!/usr/bin/env python3
"""Render one two-panel robustness figure for each AA-Mini fold."""

from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from plot_r051_3d import draw_3d


FOLD_ROOT = ROOT / "outputs/aamini_figure2_20261002/folds"
PANEL_ROOT = ROOT / "outputs/aamini_fold_figures"
RATIOS = np.arange(0, 81, 10)
B_ORDER = np.arange(8, 0, -1)


def read_fold(fold: int):
    path = FOLD_ROOT / f"fold_{fold}/grid.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def make_grid(data, metric: str):
    grid = np.empty((len(RATIOS), len(B_ORDER)), dtype=float)
    for i, ratio in enumerate(RATIOS):
        for j, blocks in enumerate(B_ORDER):
            # At 0% deletion only the B=1 condition is evaluated and is
            # repeated across the x-axis, matching the paper figure.
            key = f"{int(ratio):02d}_B{1 if ratio == 0 else int(blocks)}"
            grid[i, j] = data["results"][key][f"delta_{metric}"] * 100.0
    return grid


def main():
    folds = {fold: read_fold(fold) for fold in range(1, 6)}
    grids_f1 = {fold: make_grid(data, "f1") for fold, data in folds.items()}
    grids_acc = {fold: make_grid(data, "acc005") for fold, data in folds.items()}
    f1_limit = max(float(np.max(np.abs(grid))) for grid in grids_f1.values()) * 1.05
    acc_limit = max(float(np.max(np.abs(grid))) for grid in grids_acc.values()) * 1.05

    PANEL_ROOT.mkdir(parents=True, exist_ok=True)
    for fold in range(1, 6):
        fold_dir = PANEL_ROOT / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        f1_path = draw_3d(
            f"AA-Mini Fold {fold}", grids_f1[fold], f1_limit,
            r"$\Delta$ Macro-F1 (pp)", "delta_f1", fold_dir,
            RATIOS, B_ORDER,
        )
        acc_path = draw_3d(
            f"AA-Mini Fold {fold}", grids_acc[fold], acc_limit,
            r"$\Delta$ Acc@0.05 (pp)", "delta_acc005", fold_dir,
            RATIOS, B_ORDER,
        )
        top = Image.open(f1_path).convert("RGB")
        bottom = Image.open(acc_path).convert("RGB")
        width = max(top.width, bottom.width)
        canvas = Image.new("RGB", (width, top.height + bottom.height), "white")
        canvas.paste(top, ((width - top.width) // 2, 0))
        canvas.paste(bottom, ((width - bottom.width) // 2, top.height))
        output = ROOT / f"temp{fold:02d}.png"
        canvas.save(output, dpi=(350, 350))
        print(f"{output} ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
