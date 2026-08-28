#!/usr/bin/env python3
"""Draw the standalone WLE attention-deletion faithfulness figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT_ROOT / "temp_img/wle_label_attention_mechanism.json"
OUTPUT_STEM = PROJECT_ROOT / "temp_img/wle_attention_deletion_faithfulness"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def main() -> None:
    configure_style()
    source_payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    payload = source_payload["deletion"]
    fractions = 100 * np.asarray(payload["fractions"], dtype=float)
    styles = {
        "top": ("High-attention images", "#C23B3B", "o"),
        "random": ("Random images", "#6F6F6F", "s"),
        "bottom": ("Low-attention images", "#2878B5", "^"),
    }

    figure, axis = plt.subplots(figsize=(8.4, 5.7))
    axis.axhspan(0, 0.13, color="#C23B3B", alpha=0.045, zorder=0)
    axis.axhspan(-0.11, 0, color="#2878B5", alpha=0.045, zorder=0)
    for key, (label, color, marker) in styles.items():
        means = 100 * np.asarray(payload["curves"][key]["mean"], dtype=float)
        cis = 100 * np.asarray(payload["curves"][key]["ci95"], dtype=float)
        axis.plot(
            fractions,
            means,
            color=color,
            marker=marker,
            markersize=6.5,
            linewidth=2.2,
            label=label,
            zorder=3,
        )
        axis.fill_between(fractions, means - cis, means + cis, color=color, alpha=0.14, zorder=2)
        end_value = float(means[-1])
        offsets = {"top": 0.005, "random": 0.006, "bottom": -0.008}
        vertical_alignment = "bottom" if key != "bottom" else "top"
        axis.text(
            fractions[-1] + 0.8,
            end_value + offsets[key],
            f"{end_value:+.3f} pp",
            color=color,
            ha="left",
            va=vertical_alignment,
            fontweight="bold",
        )

    axis.axhline(0, color="#333333", linewidth=1.0, zorder=4)
    axis.text(
        0.02,
        0.965,
        r"$\Delta c=c_{original}-c_{deleted}$",
        transform=axis.transAxes,
        ha="left",
        va="top",
        color="#444444",
    )
    axis.text(
        0.98,
        0.92,
        "Positive: confidence loss",
        transform=axis.transAxes,
        ha="right",
        va="top",
        color="#9E3030",
    )
    axis.text(
        0.02,
        0.08,
        "Negative: confidence gain",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        color="#246A9B",
    )
    axis.set_xlim(-2, 47)
    axis.set_ylim(-0.105, 0.125)
    axis.set_xlabel("Deleted images per label (%)")
    axis.set_ylabel(r"True-class confidence decrease, $\Delta c$ (pp)")
    axis.set_title("WLE: label-attention deletion faithfulness test", fontweight="bold")
    axis.grid(color="#E3E3E3", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.86))
    figure.text(
        0.5,
        0.015,
        f"Held-out fold {payload['fold']} examinations (n={payload['num_held_out_examinations']}); mean and 95% CI; random deletion averaged over {payload['random_repeats']} draws",
        ha="center",
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    output_payload = {
        **payload,
        "definition": "delta_c = original true-class confidence - confidence after deletion",
        "interpretation": {
            "positive": "confidence loss after deletion",
            "negative": "confidence gain after deletion",
        },
        "endpoint_40_percent_pp": {
            key: 100 * float(payload["curves"][key]["mean"][-1]) for key in styles
        },
    }
    OUTPUT_STEM.with_suffix(".json").write_text(
        json.dumps(output_payload, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
