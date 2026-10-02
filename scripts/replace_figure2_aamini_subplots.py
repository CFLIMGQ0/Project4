#!/usr/bin/env python3
"""Replace only the AA-Mini column in the existing Figure 2 raster composite."""

from pathlib import Path
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/r051_3d_csv_20261002_aamini/r051_updated_3d_2x4.png"
AA_F1 = ROOT / "outputs/aamini_fold_figures/adjusted_fold3_gradient/delta_f1_adjusted.png"
AA_ACC = ROOT / "outputs/aamini_fold_figures/adjusted_fold3_gradient/delta_acc005_adjusted.png"
OUT_DIR = ROOT / "outputs/figure2_aamini_fold3_replaced"
OUT_PNG = OUT_DIR / "r051_updated_3d_2x4.png"
OUT_PDF = OUT_DIR / "r051_updated_3d_2x4.pdf"


def fit_panel(path: Path, size):
    # Fit only the replacement panel to the existing AA-Mini cell, leaving
    # all public panels and the original canvas geometry unchanged.
    return Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)


def main():
    canvas = Image.open(BASE).convert("RGB")
    cell_w, cell_h = canvas.width // 4, canvas.height // 2
    x0 = 3 * cell_w
    canvas.paste(fit_panel(AA_F1, (cell_w, cell_h)), (x0, 0))
    canvas.paste(fit_panel(AA_ACC, (cell_w, cell_h)), (x0, cell_h))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_PNG, dpi=(350, 350))
    canvas.save(OUT_PDF, "PDF", resolution=350.0)
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
