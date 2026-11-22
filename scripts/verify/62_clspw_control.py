#!/usr/bin/env python3
"""Doi chung cls_pw=0 vs cls_pw=0.5, cung diem khoi dau + cung ngan sach fine-tune, nhieu seed.

Muc dich (bai bao, muc Thao luan): tach tac dung cua trong so lop khoi tac dung cua
viec fine-tune them tu exp_baseline, va do dao dong giua cac seed.

- Khoi tao: runs/detect/exp_baseline/weights/best.pt (giong r_clspw_v2)
- Hyper-param giong 60_train_r_clspw_v2.py, CHI doi: cls_pw, seed, epochs=10 co dinh, patience=100
  (khong early-stop, de moi run co cung ngan sach)
- Checkpoint = best.pt theo fitness cua VALIDATION (khong dung test de chon)
- Eval: V5+ test, VN Aerial val/test (cung dieu kien 61_eval_r_clspw_v2.py)

Chay:  python scripts/verify/62_clspw_control.py            (tat ca)
       python scripts/verify/62_clspw_control.py --only cw05_s0
Co the chay lai: run da co json se duoc bo qua.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "False")
ROOT = Path("/home/xuand/vn_traffic_ai")
OUT = ROOT / "outputs/verify/clspw_control"
INIT = ROOT / "runs/detect/exp_baseline/weights/best.pt"

CONDITIONS = {"cw0": 0.0, "cw05": 0.5}
SEEDS = [0, 1, 2]
EPOCHS = 10


def train(name: str, cls_pw: float, seed: int):
    from ultralytics import YOLO

    model = YOLO(str(INIT))
    model.train(
        data=str(ROOT / "data/data_v5plus.yaml"),
        epochs=EPOCHS, patience=100, batch=16, imgsz=960,
        optimizer="SGD", cos_lr=True, close_mosaic=5,
        lr0=0.001, lrf=0.1, momentum=0.937, weight_decay=0.0005,
        warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
        box=7.5, cls=0.5, cls_pw=cls_pw, dfl=1.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.1, copy_paste=0.0,
        seed=seed, deterministic=True,
        name=f"ctrl_{name}", project=str(ROOT / "runs/detect"), exist_ok=True,
        device=0, half=False, verbose=True, plots=False,
    )
    return ROOT / f"runs/detect/ctrl_{name}/weights/best.pt"


def evaluate(ckpt: Path, name: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(ckpt))
    res = {}
    for ds, data_rel, split in [("V5+_test", "data/data_v5plus.yaml", "test"),
                                ("VN_Aerial_val", "data/data_vn_aerial.yaml", "val"),
                                ("VN_Aerial_test", "data/data_vn_aerial.yaml", "test")]:
        r = model.val(data=str(ROOT / data_rel), split=split, imgsz=960, conf=0.001, iou=0.7,
                      batch=16, half=True, device=0, plots=False, save_json=False, verbose=False,
                      name=f"ctrl_eval_{name}_{ds}")
        b = r.box
        res[ds] = {
            "mAP50": float(b.map50), "mAP50_95": float(b.map),
            "precision": float(b.mp), "recall": float(b.mr),
            "AP50_per_class": {model.names[int(c)]: float(b.ap50[i]) for i, c in enumerate(b.ap_class_index)},
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="vd: cw05_s0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # xen ke cw0/cw05 de neu dung giua chung van co du cap
    plan = [(f"{c}_s{s}", CONDITIONS[c], s) for s in SEEDS for c in ("cw0", "cw05")]
    for name, cls_pw, seed in plan:
        if args.only and name != args.only:
            continue
        out = OUT / f"{name}.json"
        if out.exists():
            print(f"[skip] {name}", flush=True)
            continue
        t0 = time.time()
        print(f"[TRAIN] {name} cls_pw={cls_pw} seed={seed}", flush=True)
        ckpt = train(name, cls_pw, seed)
        t1 = time.time()
        res = evaluate(ckpt, name)
        out.write_text(json.dumps({"name": name, "cls_pw": cls_pw, "seed": seed, "epochs": EPOCHS,
                                   "ckpt": str(ckpt.relative_to(ROOT)),
                                   "train_seconds": round(t1 - t0, 1), "results": res},
                                  indent=2, ensure_ascii=False))
        print(f"[DONE] {name}: test mAP50={res['V5+_test']['mAP50']:.4f}", flush=True)


if __name__ == "__main__":
    main()
