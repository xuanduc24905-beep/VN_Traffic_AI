#!/usr/bin/env python3
"""Kiem tra Faster R-CNN train tren Colab sau khi keo ve may, roi cham cung pycocotools voi YOLO.

Dat thu muc tai tu Drive vao weights/trained/frcnn_v5plus/ (best.pt, last.pt, train_log.csv, env.json, config.json).
Chay bang moi truong vehicle_env:
    ~/miniconda3/envs/vehicle_env/bin/python scripts/colab/verify_frcnn_v5plus.py
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch
import torchvision

ROOT = Path("/home/xuand/vn_traffic_ai")
D = ROOT / "weights/trained/frcnn_v5plus"
for f in ("best.pt", "last.pt", "train_log.csv", "env.json", "config.json"):
    assert (D / f).exists(), f"thieu {D / f}"

env = json.loads((D / "env.json").read_text())
print("Colab :", env)
print("Local :", {"torch": torch.__version__, "torchvision": torchvision.__version__})
assert env["torch"] == torch.__version__ and env["torchvision"] == torchvision.__version__, \
    "phien ban torch/torchvision tren Colab khac may local"

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
ck = torch.load(D / "best.pt", map_location="cpu", weights_only=False)
m = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 5)
m.load_state_dict(ck["model"])  # strict: loi neu kien truc khac
print(f"best.pt: epoch {ck['epoch'] + 1}, val mAP50 {ck['val_map50']:.4f}, val mAP {ck['val_map']:.4f}")
cfg = json.loads((D / "config.json").read_text())
print("config:", cfg)
rows = list(csv.DictReader(open(D / "train_log.csv")))
print(f"train_log: {len(rows)}/{cfg['epochs']} epoch")
for r in rows:
    print("  ", r)
assert len(rows) == cfg["epochs"], "chua train du so epoch"

print("\nCham bang pycocotools (V5+ test, VN Aerial val/test)...")
subprocess.check_call([sys.executable, str(ROOT / "scripts/verify/80_unified_coco_eval.py")])
res = json.loads((ROOT / "outputs/verify/unified_eval/unified_eval.json").read_text())
for s in ("v5plus_test", "aerial_val", "aerial_test"):
    for k in ("r_clspw_v2", "frcnn_300", "frcnn_v5plus_300"):
        v = res.get(f"{s}/{k}")
        if v:
            print(f"{s:12s} {k:18s} mAP50={v['mAP50']:.4f} mAP50-95={v['mAP50_95']:.4f}")
