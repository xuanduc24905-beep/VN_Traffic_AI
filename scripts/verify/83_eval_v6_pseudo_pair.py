#!/usr/bin/env python3
"""Cham lai cap V5 (2.833 anh that) va V6 cu (2.833 anh that + 402 anh nhan gia vn_cctv) tren V5+ test,
cung dieu kien Bang III (imgsz 960, conf 0,001, IoU 0,7, batch 16, FP16)."""
import json, os
from pathlib import Path
os.environ.setdefault("YOLO_VERBOSE", "False")
from ultralytics import YOLO
ROOT = Path("/home/xuand/vn_traffic_ai")
res = {}
for name in ["train_v8s_v5plus_960", "train_v8s_v6_pseudo"]:
    m = YOLO(str(ROOT / f"runs/detect/{name}/weights/best.pt"))
    r = m.val(data=str(ROOT / "data/data_v5plus.yaml"), split="test", imgsz=960, batch=16, conf=0.001, iou=0.7,
              half=True, device=0, plots=False, verbose=False, project="/tmp/claude-1000/v6pair", name=name, exist_ok=True)
    res[name] = {"mAP50": float(r.box.map50), "mAP50_95": float(r.box.map)}
    print(name, res[name], flush=True)
(ROOT / "outputs/verify/unified_eval/v6_pseudo_pair.json").write_text(json.dumps(res, indent=1))
