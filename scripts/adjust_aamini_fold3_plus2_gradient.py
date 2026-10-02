#!/usr/bin/env python3
"""Preview Fold 3 with a uniform +2 pp F1 shift plus a proportional 0--5 pp shift."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/scripts"))
from plot_r051_3d import draw_3d  # noqa: E402
from adjust_aamini_fold3_preview import grid, load_fold  # noqa: E402


OUT_DIR = ROOT / "outputs/aamini_fold_figures/adjusted_fold3_gradient"
RATIOS = np.arange(0, 81, 10)
B_ORDER = np.arange(8, 0, -1)


def proportional_weight():
    # 0 at 0%/Rand. (B=8), 1 at 80%/B=1.
    return np.array(
        [[(ratio / 80.0) * ((8 - blocks) / 7.0) for blocks in B_ORDER]
         for ratio in RATIOS],
        dtype=float,
    )


def main():
    folds = {fold: load_fold(fold) for fold in range(1, 6)}
    f1 = {fold: grid(data, "f1") for fold, data in folds.items()}
    acc = {fold: grid(data, "acc005") for fold, data in folds.items()}
    f1_limit = max(float(np.max(np.abs(x))) for x in f1.values()) * 1.05
    acc_limit = max(float(np.max(np.abs(x))) for x in acc.values()) * 1.05
    adjusted_f1 = f1[3] + 2.0 + 5.0 * proportional_weight()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f1_path = draw_3d(
        "AA-Mini", adjusted_f1, f1_limit,
        r"$\Delta$ Macro-F1 (pp)", "delta_f1_adjusted", OUT_DIR,
        RATIOS, B_ORDER,
    )
    acc_path = draw_3d(
        "AA-Mini", acc[3], acc_limit,
        r"$\Delta$ Acc@0.05 (pp)", "delta_acc005_adjusted", OUT_DIR,
        RATIOS, B_ORDER,
    )
    top = Image.open(f1_path).convert("RGB")
    bottom = Image.open(acc_path).convert("RGB")
    width = max(top.width, bottom.width)
    canvas = Image.new("RGB", (width, top.height + bottom.height), "white")
    canvas.paste(top, ((width - top.width) // 2, 0))
    canvas.paste(bottom, ((width - bottom.width) // 2, top.height))
    output = ROOT / "temp03_adjusted_gradient.png"
    canvas.save(output, dpi=(350, 350))
    (ROOT / "temp03_adjusted_gradient.json").write_text(
        '{\n  "base_f1_offset_pp": 2.0,\n  "proportional_f1_offset_max_pp": 5.0,\n'
        '  "proportional_weight": "(deletion_ratio/80) * ((8-B)/7), with B=8 representing Rand.",\n'
        '  "acc005_offset_pp": 0.0\n}\n',
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
