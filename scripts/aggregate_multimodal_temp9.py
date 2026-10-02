#!/usr/bin/env python3
"""Aggregate five-fold multimodal intervention matrices for temp9."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/lccf_public_evidence_20261001"
RAW = OUT / "multimodal_interventions" / "raw"
DATASETS = {
    "ct_rate": ["Emph.", "Atel.", "Fibrotic"],
    "mr_rate": ["Uns.", "Neuro.", "Cerebro.", "Neo."],
    "amos_mm": ["L", "K", "G", "S", "B", "P", "T"],
}
METHODS = ["mmfnet", "radfuse", "saif", "mmtf", "camchex", "med3dvlm",
           "m3fm", "unified_mm", "adaptive_fusion"]


def collapse(case: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.stack([values[case == c].mean(axis=0) for c in np.unique(case)])


def main() -> None:
    rows = []
    for ds, labels in DATASETS.items():
        for method in METHODS:
            parts = []
            for fold in range(1, 6):
                with np.load(RAW / ds / method / f"fold_{fold}.npz", allow_pickle=False) as z:
                    parts.append({k: z[k] for k in z.files})
            case = np.concatenate([p["case"] for p in parts])
            y = np.concatenate([p["labels"] for p in parts])
            image_count = np.concatenate([p["image_count"] for p in parts])
            signed = np.concatenate([p["deletion_signed"] for p in parts])
            n = len(labels)
            matrix = np.zeros((n, n), dtype=float)
            for source in range(n):
                selected = (y[:, source] > 0) & (image_count > 1)
                changes = np.abs(signed[selected, source, :])
                means = collapse(case[selected], changes).mean(axis=0)
                matrix[source] = 100 * means / means.sum() if means.sum() > 0 else np.nan
            for source, source_name in enumerate(labels):
                for target, target_name in enumerate(labels):
                    rows.append({
                        "dataset": ds, "method": method, "source": source_name,
                        "target": target_name, "value": float(matrix[source, target]),
                    })
    out = OUT / "multimodal_intervention_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "method", "source", "target", "value"])
        writer.writeheader(); writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
