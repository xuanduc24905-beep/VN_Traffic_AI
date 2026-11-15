#!/usr/bin/env python3
"""Re-val `runs/detect/exp_baseline/weights/best.pt` on V5+ test + VN Aerial val/test
with the same conditions as 30_detector_eval.py.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "False")
ROOT = Path("/home/xuand/vn_traffic_ai")


def main():
    from ultralytics import YOLO

    ckpt = ROOT / "runs/detect/exp_baseline/weights/best.pt"
    model = YOLO(str(ckpt))

    datasets = [
        ("V5+_test",       "data/data_v5plus.yaml",   "test"),
        ("VN_Aerial_val",  "data/data_vn_aerial.yaml", "val"),
        ("VN_Aerial_test", "data/data_vn_aerial.yaml", "test"),
    ]
    out = {"ckpt": "runs/detect/exp_baseline/weights/best.pt", "imgsz": 960, "per_dataset": {}}
    for ds_name, data_rel, split in datasets:
        print(f"[VAL] exp_baseline on {ds_name}", flush=True)
        t0 = time.time()
        r = model.val(
            data=str(ROOT / data_rel), split=split, imgsz=960,
            conf=0.001, iou=0.7, batch=16, half=True, device=0,
            plots=False, save_json=False, verbose=False,
            name=f"verify_exp_baseline_{ds_name}",
        )
        b = r.box
        ap50 = {}
        names = model.names
        for i, cid in enumerate(b.ap_class_index):
            ap50[names[int(cid)]] = float(b.ap50[i])
        out["per_dataset"][ds_name] = {
            "data": data_rel, "split": split,
            "mAP50": float(b.map50), "mAP50_95": float(b.map),
            "mAP75": float(b.map75),
            "precision": float(b.mp), "recall": float(b.mr),
            "AP50_per_class": ap50,
            "seconds": round(time.time() - t0, 2),
            "imgsz": 960, "conf": 0.001, "iou": 0.7, "batch": 16, "half": True,
        }
        print(f"  mAP50={float(b.map50):.4f}  mAP50-95={float(b.map):.4f}  P={float(b.mp):.4f}  R={float(b.mr):.4f}")
    (ROOT / "outputs/verify/exp_baseline_eval.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("Wrote outputs/verify/exp_baseline_eval.json")


if __name__ == "__main__":
    main()
