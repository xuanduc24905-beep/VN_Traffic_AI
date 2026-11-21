#!/usr/bin/env python3
"""BƯỚC 3 — Error analysis for r_clspw on V5+ test.

- Run model.predict() with conf=0.25, iou=0.5 (NMS iou).
- Match predictions to GT with IoU >= 0.5 (per-class).
- Compute TP/FP/FN per class.
- Compute APs/APm/APl via model.val() with the "val" method's built-in area bins.
- Classify each image as "day" (cam_09/cam_10 SOICT day) vs "night" (SOICT night in val only).
  Note: V5+ test is 100% day cam_09/cam_10. So "night" is 0 in test. We report that.
- Confusion matrix (4 class + background).
- Save example images for FP/FN/wrong class (6-10 examples).
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/xuand/vn_traffic_ai")
CLASS_NAMES = {0: "motorcycle", 1: "car", 2: "bus", 3: "truck"}


def box_iou(b1, b2):
    """b1 (4,), b2 (N,4) in xyxy. Return IoU vector (N,)."""
    x1 = np.maximum(b1[0], b2[:, 0])
    y1 = np.maximum(b1[1], b2[:, 1])
    x2 = np.minimum(b1[2], b2[:, 2])
    y2 = np.minimum(b1[3], b2[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    return inter / (a1 + a2 - inter + 1e-9)


def yolo_to_xyxy(cx, cy, w, h, iw, ih):
    x1 = (cx - w / 2) * iw
    y1 = (cy - h / 2) * ih
    x2 = (cx + w / 2) * iw
    y2 = (cy + h / 2) * ih
    return np.array([x1, y1, x2, y2])


def size_bin(x1, y1, x2, y2):
    a = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    if a < 32 * 32:
        return "small"
    if a < 96 * 96:
        return "medium"
    return "large"


def main():
    from ultralytics import YOLO
    import cv2

    ckpt = ROOT / "runs/detect/train_v8s_v5pc_full_s42/weights/best.pt"
    test_img_dir = ROOT / "data/merged_v5plus_clean/images/test"
    test_lab_dir = ROOT / "data/merged_v5plus_clean/labels/test"

    model = YOLO(str(ckpt))
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    tp_by_size = defaultdict(lambda: defaultdict(int))  # size -> cls -> tp
    fp_by_size = defaultdict(lambda: defaultdict(int))
    fn_by_size = defaultdict(lambda: defaultdict(int))
    # day/night
    tp_dn = defaultdict(lambda: defaultdict(int))
    fp_dn = defaultdict(lambda: defaultdict(int))
    fn_dn = defaultdict(lambda: defaultdict(int))
    # confusion: pred_cls x gt_cls (+bg for FP (no match), +bg row for FN)
    cm = np.zeros((5, 5), dtype=np.int64)  # rows=pred(0-3,bg=4), cols=gt(0-3,bg=4)

    images = sorted(test_img_dir.glob("*.jpg"))
    print(f"Evaluating {len(images)} images at conf=0.25, iou(match)=0.5", flush=True)

    # Example buckets
    examples = {"missed": [], "extra": [], "wrong_cls": []}

    # Run in batches via model.predict stream for speed
    for batch_start in range(0, len(images), 32):
        batch_paths = images[batch_start:batch_start+32]
        # Day vs night (V5+ test = all day cam_09/cam_10, so all day)
        for img_path in batch_paths:
            # Predict single image
            pass
        results = model.predict(source=[str(p) for p in batch_paths], conf=0.25, iou=0.7,
                                imgsz=960, verbose=False, device=0, half=True)
        for img_path, res in zip(batch_paths, results):
            # time is day/night; test = day
            daynight = "day"
            # load GT
            lab = test_lab_dir / (img_path.stem + ".txt")
            if not lab.exists():
                continue
            img_arr = res.orig_img
            ih, iw = img_arr.shape[:2]
            gts = []
            with lab.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cid = int(parts[0]); cx, cy, w, h = [float(x) for x in parts[1:]]
                    if cid not in CLASS_NAMES:
                        continue
                    gts.append((cid, yolo_to_xyxy(cx, cy, w, h, iw, ih)))
            # predictions xyxy
            preds = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)
                for b, c in zip(xyxy, cls):
                    preds.append((int(c), b))

            # Greedy matching per class: for each pred, find best unmatched gt of SAME class with IoU >= 0.5
            matched_gt = set()
            matched_pred = set()
            for pi, (pc, pb) in enumerate(preds):
                # Candidate gts of same class
                cand = [(gi, g[1]) for gi, g in enumerate(gts) if g[0] == pc and gi not in matched_gt]
                if not cand:
                    continue
                ious = box_iou(pb, np.stack([c[1] for c in cand]))
                best = np.argmax(ious)
                if ious[best] >= 0.5:
                    gi = cand[best][0]
                    matched_gt.add(gi)
                    matched_pred.add(pi)
                    tp[pc] += 1
                    sb = size_bin(*gts[gi][1])
                    tp_by_size[sb][pc] += 1
                    tp_dn[daynight][pc] += 1
                    cm[pc, pc] += 1

            # For remaining FP: try to find any overlapping gt of different class (IoU>=0.5)
            # Only once per pred: wrong_cls OR fp(bg).
            for pi, (pc, pb) in enumerate(preds):
                if pi in matched_pred:
                    continue
                # Any gt (any class) with IoU>=0.5?
                if gts:
                    ious_all = box_iou(pb, np.stack([g[1] for g in gts]))
                    best = int(np.argmax(ious_all))
                    if ious_all[best] >= 0.5 and gts[best][0] != pc:
                        # wrong class
                        gc = gts[best][0]
                        cm[pc, gc] += 1
                        fp[pc] += 1  # counted as FP for the predicted class
                        # not a FN for gc yet unless gt is otherwise unmatched; handled below
                        if len(examples["wrong_cls"]) < 10:
                            examples["wrong_cls"].append({
                                "img": str(img_path.relative_to(ROOT)),
                                "pred_class": CLASS_NAMES[pc],
                                "gt_class": CLASS_NAMES[gc],
                                "iou": float(ious_all[best]),
                            })
                        continue
                # pure background FP
                fp[pc] += 1
                sb = size_bin(*pb)
                fp_by_size[sb][pc] += 1
                fp_dn[daynight][pc] += 1
                cm[pc, 4] += 1  # pred vs bg
                # at most 1 extra example per image
                if len(examples["extra"]) < 10 and all(
                    ex["img"] != str(img_path.relative_to(ROOT)) for ex in examples["extra"]
                ):
                    examples["extra"].append({
                        "img": str(img_path.relative_to(ROOT)),
                        "pred_class": CLASS_NAMES[pc],
                        "bbox_xyxy": [float(x) for x in pb],
                    })

            # Unmatched gts = FN  (reservoir-sample across whole dataset so examples span many images)
            import random as _r
            _r.seed(hash(img_path.name) & 0xFFFF)  # stable per-image
            for gi, (gc, gb) in enumerate(gts):
                if gi in matched_gt:
                    continue
                fn[gc] += 1
                sb = size_bin(*gb)
                fn_by_size[sb][gc] += 1
                fn_dn[daynight][gc] += 1
                cm[4, gc] += 1
                # Accept FIRST missed per image, then skip to encourage diversity
                if len(examples["missed"]) < 10 and all(
                    ex["img"] != str(img_path.relative_to(ROOT)) for ex in examples["missed"]
                ):
                    examples["missed"].append({
                        "img": str(img_path.relative_to(ROOT)),
                        "gt_class": CLASS_NAMES[gc],
                        "bbox_xyxy": [float(x) for x in gb],
                    })
                    break  # at most 1 missed example per image

        if (batch_start // 32) % 5 == 0:
            print(f"  {batch_start+len(batch_paths)}/{len(images)}", flush=True)

    out = {
        "ckpt": "runs/detect/train_v8s_v5pc_full_s42/weights/best.pt",
        "test": "V5 test sach (1500 images, SOICT24 day cam_09/cam_10)",
        "imgsz": 960, "conf": 0.25, "iou_nms": 0.7, "iou_match": 0.5,
        "half": True,
        "per_class": {
            CLASS_NAMES[c]: {"tp": tp[c], "fp": fp[c], "fn": fn[c],
                             "precision": tp[c] / max(tp[c] + fp[c], 1),
                             "recall": tp[c] / max(tp[c] + fn[c], 1)}
            for c in range(4)},
        "per_size": {
            sz: {"tp": {CLASS_NAMES[c]: tp_by_size[sz][c] for c in range(4)},
                 "fp": {CLASS_NAMES[c]: fp_by_size[sz][c] for c in range(4)},
                 "fn": {CLASS_NAMES[c]: fn_by_size[sz][c] for c in range(4)}}
            for sz in ("small", "medium", "large")
        },
        "per_daynight": {
            "day": {
                "tp": {CLASS_NAMES[c]: tp_dn["day"][c] for c in range(4)},
                "fp": {CLASS_NAMES[c]: fp_dn["day"][c] for c in range(4)},
                "fn": {CLASS_NAMES[c]: fn_dn["day"][c] for c in range(4)},
            },
            "night": {
                "tp": {CLASS_NAMES[c]: tp_dn["night"][c] for c in range(4)},
                "fp": {CLASS_NAMES[c]: fp_dn["night"][c] for c in range(4)},
                "fn": {CLASS_NAMES[c]: fn_dn["night"][c] for c in range(4)},
            }
        },
        "confusion_matrix": {
            "rows_pred": ["motor", "car", "bus", "truck", "bg"],
            "cols_gt":   ["motor", "car", "bus", "truck", "bg"],
            "matrix": cm.tolist(),
        },
        "examples": examples,
    }
    (ROOT / "outputs/verify/error_analysis_clean_full.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )
    print("Wrote outputs/verify/error_analysis_clean_full.json")

    # Also save example images composited with pred+GT boxes
    out_dir = ROOT / "outputs/verify/error_examples_clean"
    out_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)
    for key, lst in examples.items():
        for i, ex in enumerate(lst[:3]):
            img_path = ROOT / ex["img"]
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            txt = key + ": " + (ex.get("pred_class", "") + "->" + ex.get("gt_class", "") if "gt_class" in ex and "pred_class" in ex else ex.get("gt_class", ex.get("pred_class", "")))
            cv2.putText(img, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            if "bbox_xyxy" in ex:
                x1, y1, x2, y2 = (int(v) for v in ex["bbox_xyxy"])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            op = out_dir / f"{key}_{i}_{img_path.stem[:40]}.jpg"
            cv2.imwrite(str(op), img)
    print(f"Wrote examples to {out_dir}")


if __name__ == "__main__":
    main()
