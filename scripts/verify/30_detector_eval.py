#!/usr/bin/env python3
"""BƯỚC 3 — Re-run YOLOv8 model.val() for 6 detectors × 3 test sets.

Checkpoints:
  V5+_baseline  = runs/detect/train_v8s_v5plus_960/weights/best.pt    (imgsz 960)
  p4_scale      = runs/detect/p4_scale_from_baseline/weights/best.pt  (imgsz 960)
  p4_imgsz1280  = runs/detect/p4_imgsz1280_from_baseline/weights/best.pt (imgsz 1280)
  r_p2          = runs/detect/r_p2/weights/best.pt                    (imgsz 960)
  r_clspw       = runs/detect/r_clspw/weights/best.pt                 (imgsz 960)
  r_progressive = runs/detect/r_progressive/weights/best.pt           (imgsz 1280)

Test sets (dataset YAMLs with 'test' pointing to the right split):
  V5+ test        = data/data_v5plus.yaml           (split='test', 1500 img)
  VN Aerial val   = data/data_vn_aerial.yaml        (split='val',  486 img)
  VN Aerial test  = data/data_vn_aerial.yaml        (split='test', 149 img)

Val params: conf=0.001, iou=0.7, batch=16, half=True (FP16).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "False")

ROOT = Path("/home/xuand/vn_traffic_ai")

CKPTS = {
    "V5+_baseline":   ("runs/detect/train_v8s_v5plus_960/weights/best.pt", 960),
    "p4_scale":       ("runs/detect/p4_scale_from_baseline/weights/best.pt", 960),
    "p4_imgsz1280":   ("runs/detect/p4_imgsz1280_from_baseline/weights/best.pt", 1280),
    "r_p2":           ("runs/detect/r_p2/weights/best.pt", 960),
    "r_clspw":        ("runs/detect/r_clspw/weights/best.pt", 960),
    "r_progressive":  ("runs/detect/r_progressive/weights/best.pt", 1280),
}

DATASETS = [
    ("V5+_test",        "data/data_v5plus.yaml",   "test"),
    ("VN_Aerial_val",   "data/data_vn_aerial.yaml", "val"),
    ("VN_Aerial_test",  "data/data_vn_aerial.yaml", "test"),
]


def main():
    from ultralytics import YOLO
    import torch

    out_path = ROOT / "outputs/verify/detector_eval.json"
    results = {}
    if out_path.exists():
        results = json.loads(out_path.read_text())

    for model_name, (ckpt_rel, imgsz) in CKPTS.items():
        ckpt = ROOT / ckpt_rel
        if not ckpt.exists():
            print(f"[SKIP] {model_name}: ckpt not found")
            continue
        model_entry = results.setdefault(model_name, {"ckpt": ckpt_rel, "imgsz": imgsz, "per_dataset": {}})
        model = YOLO(str(ckpt))
        for ds_name, data_rel, split in DATASETS:
            if ds_name in model_entry["per_dataset"]:
                print(f"[CACHED] {model_name} / {ds_name}")
                continue
            data = ROOT / data_rel
            t0 = time.time()
            print(f"[VAL] {model_name} on {ds_name}  imgsz={imgsz}  split={split}  ckpt={ckpt_rel}")
            try:
                r = model.val(
                    data=str(data),
                    split=split,
                    imgsz=imgsz,
                    conf=0.001,
                    iou=0.7,
                    batch=16,
                    half=True,
                    device=0,
                    plots=False,
                    save_json=False,
                    verbose=False,
                    name=f"verify_{model_name}_{ds_name}",
                )
                b = r.box
                # per-class AP@0.5
                ap50_per_class = {}
                names = model.names
                if hasattr(b, "ap50") and b.ap50 is not None:
                    for i, cid in enumerate(b.ap_class_index):
                        ap50_per_class[names[int(cid)]] = float(b.ap50[i])
                model_entry["per_dataset"][ds_name] = {
                    "data": data_rel,
                    "split": split,
                    "mAP50": float(b.map50),
                    "mAP50_95": float(b.map),
                    "mAP75": float(b.map75),
                    "precision": float(b.mp),
                    "recall": float(b.mr),
                    "AP50_per_class": ap50_per_class,
                    "seconds": round(time.time() - t0, 2),
                    "imgsz": imgsz,
                    "conf": 0.001,
                    "iou": 0.7,
                    "batch": 16,
                    "half": True,
                }
            except Exception as e:
                model_entry["per_dataset"][ds_name] = {"ERROR": str(e)}
                print(f"  ERROR: {e}")
            # Save progressively
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
