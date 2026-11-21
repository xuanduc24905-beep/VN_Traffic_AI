"""Cham diem V5fix: test SOICT (1.500 anh) + tap ngoai data/ext_benchmark (nhan NHAP yolov8x).

Chi so tap ngoai (theo nhom india / vncctv):
  - recall dinh vi theo lop: % hop nhan nhap duoc 1 du doan bat trung (IoU>=0.5, khong xet lop),
    conf=0.25, imgsz=960 — dung duoc cho ca yolov8s COCO
  - so hop du doan chiem > 25% khung hinh ("hop khong lo")
  - mAP50 (Ultralytics val) cho model 4 lop

Chay: python scripts/evaluation/eval_v5fix.py [--models ten1 ten2 ...]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "v5fix" / "results.json"
MODELS = {
    "coco_yolov8s": "yolov8s.pt",
    "r_clspw_v2": "runs/detect/r_clspw_v2/weights/best.pt",
    "v5pc_full_s42": "runs/detect/train_v8s_v5pc_full_s42/weights/best.pt",
    "v5fix_d1_freeze10": "runs/detect/v5fix_d1_freeze10/weights/best.pt",
    "v5fix_d2_unfreeze": "runs/detect/v5fix_d2_unfreeze/weights/best.pt",
    "v5fix_c_nofreeze": "runs/detect/v5fix_c_nofreeze/weights/best.pt",
    "v5fixmio_d1_freeze10": "runs/detect/v5fixmio_d1_freeze10/weights/best.pt",
    "v5fixmio_mask_e30": "runs/detect/v5fixmio_mask_e30/weights/best.pt",
    "v5fixmio_mask_e15c": "runs/detect/v5fixmio_mask_e15c/weights/best.pt",
    "v5fixmio_mask_e60": "runs/detect/v5fixmio_mask_e60/weights/best.pt",
}
COCO_TO_V5 = {3: 0, 2: 1, 5: 2, 7: 3}
NAMES = ["motorcycle", "car", "bus", "truck"]


def ext_recall(weights: str, is_coco: bool) -> dict:
    from ultralytics import YOLO
    from ultralytics.utils.metrics import box_iou

    ims = sorted((ROOT / "data" / "ext_benchmark" / "images").glob("*.jpg"))
    m = YOLO(str(ROOT / weights))
    tp, ngt = defaultdict(lambda: np.zeros(4)), defaultdict(lambda: np.zeros(4))
    giant, npred = defaultdict(int), defaultdict(int)
    for i in range(0, len(ims), 16):
        batch = [str(p) for p in ims[i:i + 16]]
        res = m.predict(batch, imgsz=960, conf=0.25, iou=0.5, half=True, verbose=False,
                        classes=list(COCO_TO_V5) if is_coco else None)
        for p, r in zip(batch, res):
            grp = Path(p).name.split("_")[0]
            h, w = r.orig_shape
            lp = Path(p.replace("/images/", "/labels/")).with_suffix(".txt")
            a = np.loadtxt(lp, ndmin=2) if lp.stat().st_size else np.zeros((0, 5))
            gb = torch.tensor(np.c_[(a[:, 1] - a[:, 3] / 2) * w, (a[:, 2] - a[:, 4] / 2) * h,
                                    (a[:, 1] + a[:, 3] / 2) * w, (a[:, 2] + a[:, 4] / 2) * h]).float().reshape(-1, 4)
            gc = a[:, 0].astype(int)
            d = r.boxes.xyxy.cpu().float()
            area = ((d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])) / (w * h)
            giant[grp] += int((area > 0.25).sum())
            npred[grp] += len(d)
            for c in range(4):
                ngt[grp][c] += (gc == c).sum()
            if len(gb) and len(d):
                io, used = box_iou(gb, d), set()
                for j in range(len(gb)):
                    k = int(io[j].argmax())
                    if io[j, k] >= 0.5 and k not in used:
                        used.add(k)
                        tp[grp][gc[j]] += 1
    out = {}
    for g in sorted(ngt):
        out[g] = {"recall": {NAMES[c]: round(float(tp[g][c] / ngt[g][c]), 4) for c in range(4) if ngt[g][c]},
                  "recall_all": round(float(tp[g].sum() / ngt[g].sum()), 4),
                  "n_gt": {NAMES[c]: int(ngt[g][c]) for c in range(4)},
                  "n_pred": npred[g], "giant_boxes": giant[g]}
    return out


def map_val(weights: str, data: str, split: str) -> dict:
    from ultralytics import YOLO

    r = YOLO(str(ROOT / weights)).val(data=str(ROOT / data), split=split, imgsz=960, batch=16, conf=0.001,
                                     iou=0.7, plots=False, verbose=False, project="/tmp", name="v5fix_eval",
                                     exist_ok=True)
    return {"mAP50": round(float(r.box.map50), 4), "mAP50_95": round(float(r.box.map), 4),
            "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4),
            "AP50": {NAMES[c]: round(float(r.box.ap50[i]), 4) for i, c in enumerate(r.box.ap_class_index)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    args = ap.parse_args()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in args.models:
        w = MODELS[name]
        if not (ROOT / w).exists():
            print(f"[SKIP] {name}: chua co {w}")
            continue
        is_coco = name.startswith("coco")
        entry = {"weights": w, "ext": ext_recall(w, is_coco)}
        if not is_coco:
            entry["soict_test"] = map_val(w, "data/data_v5fix.yaml", "test")
            entry["ext_map"] = map_val(w, "data/data_ext_benchmark.yaml", "test")
        res[name] = entry
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        e = entry["ext"]
        print(f"{name}: india car={e['india']['recall'].get('car')} all={e['india']['recall_all']} "
              f"| vncctv all={e['vncctv']['recall_all']} | giant={sum(v['giant_boxes'] for v in e.values())} "
              f"| SOICT mAP50={entry.get('soict_test', {}).get('mAP50')}", flush=True)


if __name__ == "__main__":
    main()
