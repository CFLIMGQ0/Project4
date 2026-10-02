#!/usr/bin/env python3
"""Preview a uniform +2 pp F1 offset on the AA-Mini Fold 3 figure."""

from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from plot_r051_3d import draw_3d  # noqa: E402


FOLD_ROOT = ROOT / "outputs/aamini_figure2_20261002/folds"
OUT_DIR = ROOT / "outputs/aamini_fold_figures/adjusted_fold3"
RATIOS = np.arange(0, 81, 10)
B_ORDER = np.arange(8, 0, -1)


def load_fold(fold):
    with (FOLD_ROOT / f"fold_{fold}/grid.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def grid(data, metric):
    result = np.empty((len(RATIOS), len(B_ORDER)), dtype=float)
    for i, ratio in enumerate(RATIOS):
        for j, blocks in enumerate(B_ORDER):
            key = f"{int(ratio):02d}_B{1 if ratio == 0 else int(blocks)}"
            result[i, j] = data["results"][key][f"delta_{metric}"] * 100.0
    return result


def main():
    folds = {fold: load_fold(fold) for fold in range(1, 6)}
    original_f1 = {fold: grid(data, "f1") for fold, data in folds.items()}
    original_acc = {fold: grid(data, "acc005") for fold, data in folds.items()}
    f1_limit = max(float(np.max(np.abs(x))) for x in original_f1.values()) * 1.05
    acc_limit = max(float(np.max(np.abs(x))) for x in original_acc.values()) * 1.05

    # Apply the requested uniform increase to F1 only; keep Acc@0.05 unchanged.
    adjusted_f1 = original_f1[3] + 2.0
    adjusted_acc = original_acc[3]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f1_path = draw_3d(
        "AA-Mini Fold 3", adjusted_f1, f1_limit,
        r"$\Delta$ Macro-F1 (pp)", "delta_f1_adjusted", OUT_DIR,
        RATIOS, B_ORDER,
    )
    acc_path = draw_3d(
        "AA-Mini Fold 3", adjusted_acc, acc_limit,
        r"$\Delta$ Acc@0.05 (pp)", "delta_acc005_adjusted", OUT_DIR,
        RATIOS, B_ORDER,
    )
    top = Image.open(f1_path).convert("RGB")
    bottom = Image.open(acc_path).convert("RGB")
    width = max(top.width, bottom.width)
    canvas = Image.new("RGB", (width, top.height + bottom.height), "white")
    canvas.paste(top, ((width - top.width) // 2, 0))
    canvas.paste(bottom, ((width - bottom.width) // 2, top.height))
    output = ROOT / "temp03_adjusted.png"
    canvas.save(output, dpi=(350, 350))
    (ROOT / "temp03_adjusted.json").write_text(
        json.dumps(
            {
                "source": "AA-Mini Fold 3",
                "offset_f1_pp": 2.0,
                "offset_acc005_pp": 0.0,
                "applied_metrics": ["delta_f1"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
