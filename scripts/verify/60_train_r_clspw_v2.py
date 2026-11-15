#!/usr/bin/env python3
"""Retrain r_clspw with warm-start from exp_baseline (the REAL V5+ baseline).

Only difference from r_clspw v1: `model` points to exp_baseline/best.pt instead
of train_v8s_v5plus_960/best.pt. All other hyperparams identical.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path("/home/xuand/vn_traffic_ai")


def main():
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "runs/detect/exp_baseline/weights/best.pt"))
    model.train(
        data=str(ROOT / "data/data_v5plus.yaml"),
        epochs=15,
        patience=8,
        batch=16,
        imgsz=960,
        optimizer="SGD",
        cos_lr=True,
        close_mosaic=5,
        lr0=0.001,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        cls_pw=0.5,
        dfl=1.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5,
        mosaic=1.0, mixup=0.1, copy_paste=0.0,
        name="r_clspw_v2",
        project=str(ROOT / "runs/detect"),
        exist_ok=True,
        device=0,
        half=False,
        verbose=True,
    )


if __name__ == "__main__":
    main()
