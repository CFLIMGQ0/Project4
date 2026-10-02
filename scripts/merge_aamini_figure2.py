#!/usr/bin/env python3
"""Append the measured AA-Mini Figure-2 grid to the existing public CSV."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "r051_3d_figure_data.csv"
AA = ROOT / "outputs/aamini_figure2_20261002/grid_mean_std.csv"
MERGED = ROOT / "r051_3d_figure_data_aamini.csv"


def main() -> None:
    with PUBLIC.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
        fields = list(rows[0])
    with AA.open(encoding="utf-8-sig", newline="") as stream:
        aa_rows = list(csv.DictReader(stream))
    if len(aa_rows) != 65:
        raise RuntimeError(f"AA-Mini 网格尚未完整：{len(aa_rows)}/65")
    for row in aa_rows:
        rows.append({
            "dataset": "AA-Mini",
            "deletion_ratio_pct": row["ratio"],
            "B": row["B"],
            "cases": row["cases"],
            "r051_f1": row["acpe_macro_f1_mean"],
            "original_pe_f1": row["original_macro_f1_mean"],
            "delta_f1": row["delta_f1_mean"],
            "r051_acc005": row["acpe_acc005_mean"],
            "original_pe_acc005": row["original_acc005_mean"],
            "delta_acc005": row["delta_acc005_mean"],
        })
    with MERGED.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    metadata = {
        "public_source": str(PUBLIC.relative_to(ROOT)),
        "aa_source": str(AA.relative_to(ROOT)),
        "rows": len(rows),
        "conditions_per_dataset": 65,
        "aa_model": "AA-Mini full ACPE vs position_original checkpoint",
        "aa_input_scope": "deletion masks are drawn on original acquisition indices and applied to cached uniform-64 features",
        "note": "The public rows are preserved byte-for-byte; only AA-Mini rows are appended.",
    }
    (MERGED.with_suffix(".provenance.json")).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(MERGED)


if __name__ == "__main__":
    main()
