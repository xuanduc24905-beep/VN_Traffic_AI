#!/usr/bin/env python3
"""Hình 4.1 — box_loss, cls_loss, dfl_loss, mAP curves from results.csv of the
V5+ baseline (exp_baseline — Hướng B).
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("/home/xuand/vn_traffic_ai")


def read_csv(p: Path):
    rows = []
    with p.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k.strip(): float(v) if v.strip() else None for k, v in row.items() if k.strip() != ""})
    return rows


def main():
    src = ROOT / "runs/detect/exp_baseline/results.csv"
    rows = read_csv(src)
    epochs = [r["epoch"] for r in rows]
    train_box = [r["train/box_loss"] for r in rows]
    train_cls = [r["train/cls_loss"] for r in rows]
    train_dfl = [r["train/dfl_loss"] for r in rows]
    val_box = [r["val/box_loss"] for r in rows]
    val_cls = [r["val/cls_loss"] for r in rows]
    val_dfl = [r["val/dfl_loss"] for r in rows]
    map50 = [r["metrics/mAP50(B)"] for r in rows]
    map5095 = [r["metrics/mAP50-95(B)"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (train_v, val_v, name) in zip(axes.flat[:3], [
        (train_box, val_box, "Box loss"),
        (train_cls, val_cls, "Classification loss"),
        (train_dfl, val_dfl, "DFL loss"),
    ]):
        ax.plot(epochs, train_v, label="train", linewidth=2)
        ax.plot(epochs, val_v, label="val", linewidth=2, linestyle="--")
        ax.set_title(name)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend()

    ax = axes.flat[3]
    ax.plot(epochs, map50, label="mAP@0.5", linewidth=2)
    ax.plot(epochs, map5095, label="mAP@0.5:0.95", linewidth=2, linestyle="--")
    ax.set_title("Validation mAP")
    ax.set_xlabel("Epoch"); ax.set_ylabel("mAP")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.suptitle("Hình 4.1 — Training curves cho V5+ baseline (exp_baseline, 40 epoch)")
    plt.tight_layout()
    out_dir = ROOT / "outputs/verify"
    plt.savefig(out_dir / "fig_4_1_training_curves_exp_baseline.png", dpi=120)
    plt.savefig(out_dir / "fig_4_1_training_curves_exp_baseline.pdf")
    print(f"Wrote {out_dir}/fig_4_1_training_curves_exp_baseline.{{png,pdf}}")

    # Also compare exp_baseline vs r_clspw_v2
    src2 = ROOT / "runs/detect/r_clspw_v2/results.csv"
    rows2 = read_csv(src2)
    epochs2 = [r["epoch"] for r in rows2]
    map50_2 = [r["metrics/mAP50(B)"] for r in rows2]

    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, map50, label="V5+ baseline (exp_baseline)", linewidth=2)
    ax.plot(epochs2, map50_2, label="r_clspw_v2 (fine-tune)", linewidth=2, linestyle="--")
    ax.set_title("Validation mAP@0.5 — baseline vs mô hình cuối")
    ax.set_xlabel("Epoch"); ax.set_ylabel("mAP@0.5")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_4_1b_baseline_vs_r_clspw_v2.png", dpi=120)
    print(f"Wrote {out_dir}/fig_4_1b_baseline_vs_r_clspw_v2.png")


if __name__ == "__main__":
    main()
