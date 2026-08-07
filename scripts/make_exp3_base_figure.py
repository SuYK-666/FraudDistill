# -*- coding: utf-8 -*-
"""Exp3 Student capability-chain figure (Base-1.5B -> Trained Student).

Reads outputs/neural_student/base_chain_500.json (from compare_exp3_base_chain.py)
and renders fig9_student_capability_chain.png: Macro-F1 / Recall / FPR across
Base-1.5B-ZeroShot, Random Head, Neural-Gold, Neural-SoftDistill, Neural-FullDistill
on the same fixed 500-row test subset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "experiments" / "exp3_agent_distillation_ablation" / "outputs"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_ORDER = [
    "Base-1.5B-ZeroShot",
    "Neural-ZeroShot (random head)",
    "Neural-Gold",
    "Neural-SoftDistill",
    "Neural-FullDistill",
]


def main():
    data = json.loads((OUT_ROOT / "neural_student" / "base_chain_500.json").read_text(encoding="utf-8"))
    names = [m for m in MODEL_ORDER if m in data]
    labels = [n.replace("Neural-", "").replace(" (random head)", " RandomHead") for n in names]
    mf1 = [data[n]["macro_f1"] for n in names]
    rec = [data[n]["recall"] for n in names]
    fpr = [data[n]["fpr"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = ["#c0392b", "#e67e22", "#f1c40f", "#27ae60", "#2980b9"]
    for ax, vals, title, fmt in (
            (axes[0], mf1, "Macro-F1", "{:.3f}"),
            (axes[1], rec, "Recall", "{:.3f}"),
            (axes[2], fpr, "FPR", "{:.3f}")):
        bars = ax.bar(labels, vals, color=colors[:len(names)], edgecolor="black", linewidth=0.6)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, max(max(vals) * 1.25, 0.2))
        ax.tick_params(axis="x", rotation=18, labelsize=8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, fmt.format(v),
                    ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Exp3 Student capability chain (fixed 500-row test subset)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_ROOT / "figures" / "fig9_student_capability_chain.png"
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()