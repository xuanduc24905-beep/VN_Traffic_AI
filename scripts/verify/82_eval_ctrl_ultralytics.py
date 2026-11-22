#!/usr/bin/env python3
"""Cham luot doi chung cls_pw=0 (ctrl_cw0_s0, dung o epoch 3/10) bang model.val() cua Ultralytics,
cung dieu kien voi Bang III/IV/V (conf 0,001, IoU 0,7, batch 16, FP16, imgsz 960)."""
import json
import os
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "False")
from ultralytics import YOLO

ROOT = Path("/home/xuand/vn_traffic_ai")
CK = ROOT / "runs/detect/ctrl_cw0_s0_partial/weights/best.pt"
OUT = ROOT / "outputs/verify/unified_eval/ctrl_cw0_s0_ultralytics.json"
res = {}
for name, yaml, split in [("v5plus_test", "data/data_v5plus.yaml", "test"),
                          ("aerial_val", "data/data_vn_aerial.yaml", "val"),
                          ("aerial_test", "data/data_vn_aerial.yaml", "test")]:
    m = YOLO(str(CK))
    r = m.val(data=str(ROOT / yaml), split=split, imgsz=960, batch=16, conf=0.001, iou=0.7, half=True,
              device=0, plots=False, verbose=False, project="/tmp/claude-1000/ctrl_val", name=name, exist_ok=True)
    res[name] = {"mAP50": float(r.box.map50), "mAP50_95": float(r.box.map), "P": float(r.box.mp), "R": float(r.box.mr),
                 "AP50_per_class": [float(x) for x in r.box.ap50]}
    print(name, res[name], flush=True)
import torch
ck = torch.load(CK, map_location="cpu", weights_only=False)
res["best_epoch_index"] = ck.get("epoch")
OUT.write_text(json.dumps(res, indent=1))
