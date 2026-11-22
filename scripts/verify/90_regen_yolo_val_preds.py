#!/usr/bin/env python3
"""Sinh lai du doan YOLO bang model.val(save_json=True) (NMS nhieu nhan, tien xu ly chuan cua Ultralytics),
thay cho du doan predict() mot nhan ma 80_unified_coco_eval.py da luu. Ly do: Faster R-CNN cua TorchVision
von cho nhieu nhan moi vung, nen so sanh voi YOLO mot nhan bi lech (xem multilabel_isolation.json).

Ghi de pred_{tap}_{mo_hinh}.npz (ban cu chuyen vao unified_eval/predict_single_label/), roi chay
80_unified_coco_eval.py --rescore de cham lai.
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np

os.environ.setdefault("YOLO_VERBOSE", "False")
from ultralytics import YOLO

ROOT = Path("/home/xuand/vn_traffic_ai")
D = ROOT / "outputs/verify/unified_eval"
BK = D / "predict_single_label"
BK.mkdir(exist_ok=True)
SETS = {"v5plus_test": ("data/data_v5plus.yaml", "test"),
        "aerial_val": ("data/data_vn_aerial.yaml", "val"),
        "aerial_test": ("data/data_vn_aerial.yaml", "test")}
MODELS = {"exp_baseline": "runs/detect/exp_baseline/weights/best.pt",
          "r_clspw_v2": "runs/detect/r_clspw_v2/weights/best.pt",
          "ctrl_cw0_s0": "runs/detect/ctrl_cw0_s0_partial/weights/best.pt"}
for c in ("cw0", "cw05"):
    for s in range(3):
        MODELS[f"{c}_s{s}"] = f"runs/detect/ctrl_{c}_s{s}/weights/best.pt"

log = {}
for sname, (yaml, split) in SETS.items():
    gt = json.loads((D / f"gt_{sname}.json").read_text())
    by_stem = {Path(i["file_name"]).stem: i["id"] for i in gt["images"]}
    for name, ck in MODELS.items():
        out = D / f"pred_{sname}_{name}.npz"
        if out.exists() and not (BK / out.name).exists():
            shutil.move(str(out), BK / out.name)  # giu ban predict() mot nhan de doi chieu
        proj = f"/tmp/claude-1000/valpred/{sname}_{name}"
        r = YOLO(str(ROOT / ck)).val(data=str(ROOT / yaml), split=split, imgsz=960, batch=16, conf=0.001, iou=0.7,
                                     max_det=300, half=True, device=0, plots=False, verbose=False, save_json=True,
                                     project=proj, name="v", exist_ok=True)
        preds = json.loads((Path(proj) / "v/predictions.json").read_text())
        per = {i: [] for i in range(len(gt["images"]))}
        for p in preds:
            x, y, w, h = p["bbox"]
            per[by_stem[str(p["image_id"])]].append([x, y, x + w, y + h, p["score"], p["category_id"] - 1])
        np.savez_compressed(out, **{f"b{i}": np.array(per[i], dtype=np.float64).reshape(-1, 6) for i in per})
        log[f"{sname}/{name}"] = {"ultralytics_mAP50": float(r.box.map50), "ultralytics_mAP50_95": float(r.box.map),
                                  "P": float(r.box.mp), "R": float(r.box.mr), "n_preds": len(preds)}
        print(sname, name, log[f"{sname}/{name}"], flush=True)
(D / "yolo_val_preds_log.json").write_text(json.dumps(log, indent=1))
