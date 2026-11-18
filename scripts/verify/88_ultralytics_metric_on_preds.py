#!/usr/bin/env python3
"""Ap dung cong thuc tinh chi so cua Ultralytics (ghep du doan voi nhan, ap_per_class, P/R tai nguong F1 cao nhat)
len du doan da luu boi 80_unified_coco_eval.py, de dua Faster R-CNN vao cung bang voi cac mo hinh YOLO (Bang III).

Kiem tra tinh dung dan: chay tren du doan YOLO va doi chieu voi ket qua model.val() da biet.
"""
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import DetMetrics, box_iou

ROOT = Path("/home/xuand/vn_traffic_ai")
D = ROOT / "outputs/verify/unified_eval"
NAMES = {0: "motorcycle", 1: "car", 2: "bus", 3: "truck"}
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "v5plus_test"


def score(sname: str, model: str):
    gt = json.loads((D / f"gt_{sname}.json").read_text())
    z = np.load(D / f"pred_{sname}_{model}.npz")
    anns = {}
    for a in gt["annotations"]:
        x, y, w, h = a["bbox"]
        anns.setdefault(a["image_id"], []).append([x, y, x + w, y + h, a["category_id"] - 1])
    ns = types.SimpleNamespace(iouv=torch.linspace(0.5, 0.95, 10))
    m = DetMetrics(names=NAMES)
    for i in range(len(gt["images"])):
        g = np.array(anns.get(i, []), dtype=np.float32).reshape(-1, 5)
        p = z[f"b{i}"].astype(np.float32).reshape(-1, 6)
        tp = np.zeros((len(p), 10), dtype=bool)
        if len(g) and len(p):
            iou = box_iou(torch.from_numpy(g[:, :4]), torch.from_numpy(p[:, :4]))
            tp = BaseValidator.match_predictions(ns, torch.from_numpy(p[:, 5]), torch.from_numpy(g[:, 4]), iou).cpu().numpy() \
                if hasattr(BaseValidator, "match_predictions") else tp
        m.update_stats({"tp": tp, "target_cls": g[:, 4], "target_img": np.unique(g[:, 4]), "conf": p[:, 4],
                        "pred_cls": p[:, 5], "im_name": str(i)})
    m.process()
    r = m.results_dict
    return {"mAP50": float(r["metrics/mAP50(B)"]), "mAP50_95": float(r["metrics/mAP50-95(B)"]),
            "P": float(r["metrics/precision(B)"]), "R": float(r["metrics/recall(B)"]),
            "AP50_per_class": [float(x) for x in m.box.all_ap[:, 0]]}


if __name__ == "__main__":
    models = [m.stem.replace(f"pred_{SPLIT}_", "") for m in sorted(D.glob(f"pred_{SPLIT}_*.npz"))]
    out = {}
    for mod in models:
        out[mod] = score(SPLIT, mod)
        print(f"{SPLIT}/{mod:20s}", {k: (round(v, 4) if not isinstance(v, list) else [round(x, 4) for x in v]) for k, v in out[mod].items()}, flush=True)
    (D / f"ultralytics_metric_{SPLIT}.json").write_text(json.dumps(out, indent=1))
