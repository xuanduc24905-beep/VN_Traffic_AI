#!/usr/bin/env python3
"""Danh gia checkpoint train tren ban du lieu SACH (variant full) tren 3 bo: V5 test, VN Aerial val/test.

Dung SAU khi chay:  python scripts/training/v5plus_clean_train.py --variant full --seed 42
(script do da tu danh gia V5 test dung 1 lan; script nay bo sung VN Aerial va cung dieu kien
 voi 61_eval_r_clspw_v2.py: imgsz 960, conf 0.001, iou 0.7, batch 16, FP16)

Chay:
    python scripts/verify/63_eval_clean_full.py
    python scripts/verify/63_eval_clean_full.py --ckpt runs/detect/train_v8s_v5pc_full_s42/weights/best.pt
Ket qua: outputs/verify/clean_full_eval.json  -> dien vao cac o [?] trong docs/paper/main.tex
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "False")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = ROOT / "runs/detect/train_v8s_v5pc_full_s42/weights/best.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    if not args.ckpt.exists():
        raise SystemExit(f"Khong thay checkpoint: {args.ckpt}")

    from ultralytics import YOLO
    import ultralytics

    model = YOLO(str(args.ckpt))
    datasets = [
        ("V5_test",        "data/data_v5plus_clean.yaml", "test"),   # test giu nguyen, val sach
        ("VN_Aerial_val",  "data/data_vn_aerial.yaml",    "val"),
        ("VN_Aerial_test", "data/data_vn_aerial.yaml",    "test"),
    ]
    out = {"ckpt": str(args.ckpt.relative_to(ROOT)) if args.ckpt.is_relative_to(ROOT) else str(args.ckpt),
           "ultralytics": ultralytics.__version__, "imgsz": 960, "per_dataset": {}}
    for name, data_rel, split in datasets:
        print(f"[VAL] {name}", flush=True)
        t0 = time.time()
        r = model.val(data=str(ROOT / data_rel), split=split, imgsz=960, conf=0.001, iou=0.7,
                      batch=16, half=True, device=args.device, plots=False, save_json=False,
                      verbose=False, name=f"verify_clean_full_{name}")
        b = r.box
        out["per_dataset"][name] = {
            "mAP50": float(b.map50), "mAP50_95": float(b.map),
            "precision": float(b.mp), "recall": float(b.mr),
            "AP50_per_class": {model.names[int(c)]: float(b.ap50[i]) for i, c in enumerate(b.ap_class_index)},
            "seconds": round(time.time() - t0, 1),
        }
        d = out["per_dataset"][name]
        print(f"  mAP50={d['mAP50']:.4f} mAP50-95={d['mAP50_95']:.4f} P={d['precision']:.4f} R={d['recall']:.4f}")
        print(f"  AP50 per class: {d['AP50_per_class']}")
    dst = ROOT / "outputs/verify/clean_full_eval.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
