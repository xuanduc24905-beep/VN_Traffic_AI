"""Eval + so sánh 6 models cuối Tuần 2 → bảng + chart cho báo cáo.

Models:
    1. baseline (train_v8s_1round_merged_v2)
    2. V1 fix small_object (train_v8s_fix_small_object)
    3. V2 fix small_car (train_v8s_fix_v2_small_car)
    4. freeze=10 (train_v8s_freeze10)
    5. cls boost (train_v8s_cls_boost)
    6. YOLOv8m (train_v8m_merged)

Chạy sau khi tất cả 3 experiment Tuần 2 xong. Output:
    - results/error_analysis/all_models_summary.json
    - results/error_analysis/all_models_comparison.csv
    - results/error_analysis/all_models_chart.png
"""
import json, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO

PROJECT_ROOT = Path("/home/xuand/vn_traffic_ai").resolve()
RUNS_DIR     = PROJECT_ROOT / "runs" / "detect"
DATA_YAML    = PROJECT_ROOT / "data" / "data.yaml"
MERGED_DIR   = PROJECT_ROOT / "data" / "merged"
VAL_IMG_DIR  = MERGED_DIR / "images" / "val"
VAL_LBL_DIR  = MERGED_DIR / "labels" / "val"
ERR_DIR      = PROJECT_ROOT / "results" / "error_analysis"

class_names = ["motorcycle", "car", "bus", "truck"]
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.5

MODELS = [
    ("baseline",         RUNS_DIR / "train_v8s_1round_merged_v2" / "weights" / "best.pt"),
    ("v1_small_object",  RUNS_DIR / "train_v8s_fix_small_object" / "weights" / "best.pt"),
    ("v2_small_car",     RUNS_DIR / "train_v8s_fix_v2_small_car"  / "weights" / "best.pt"),
    ("freeze10",         RUNS_DIR / "train_v8s_freeze10"         / "weights" / "best.pt"),
    ("cls_boost",        RUNS_DIR / "train_v8s_cls_boost"        / "weights" / "best.pt"),
    ("yolov8m",          RUNS_DIR / "train_v8m_merged"           / "weights" / "best.pt"),
]


def iou_xyxy(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def match_image(preds, gts, conf_thresh=CONF_THRESHOLD, iou_thresh=IOU_THRESHOLD):
    preds_f = [p for p in preds if p["conf"] >= conf_thresh]
    preds_sorted = sorted(enumerate(preds_f), key=lambda x: -x[1]["conf"])
    gt_matched = [False] * len(gts)
    pred_tp = [False] * len(preds_f)
    pred_cc = [False] * len(preds_f)
    for pi, p in preds_sorted:
        best_iou = iou_thresh; best_gi = -1
        for gi, g in enumerate(gts):
            if gt_matched[gi]: continue
            io = iou_xyxy(p["xyxy"], g["xyxy"])
            if io >= best_iou:
                best_iou = io; best_gi = gi
        if best_gi >= 0:
            gt_matched[best_gi] = True
            if gts[best_gi]["cls"] == p["cls"]: pred_tp[pi] = True
            else: pred_cc[pi] = True
    return {
        "n_tp": sum(pred_tp),
        "n_fp": sum(1 for pi in range(len(preds_f)) if not pred_tp[pi] and not pred_cc[pi]),
        "n_cc": sum(pred_cc),
        "n_fn": sum(1 for g in gt_matched if not g),
    }


def inference(weights, val_images):
    model = YOLO(str(weights))
    preds = {}
    for i in tqdm(range(0, len(val_images), 64), desc="Infer"):
        batch = val_images[i:i+64]
        results = model.predict(
            [str(p) for p in batch],
            imgsz=512, conf=0.001, iou=0.5, verbose=False, half=True
        )
        for p, r in zip(batch, results):
            boxes = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls  = r.boxes.cls.cpu().numpy().astype(int)
                conf = r.boxes.conf.cpu().numpy()
                for b, c, cf in zip(xyxy, cls, conf):
                    boxes.append({"xyxy": b.tolist(), "cls": int(c), "conf": float(cf)})
            preds[p.name] = boxes
    return preds


def main():
    # Load GT metadata (phải đã có từ error_analysis.ipynb)
    meta_cache = ERR_DIR / "image_metadata.pkl"
    assert meta_cache.exists(), "Thiếu image_metadata.pkl → chạy error_analysis.ipynb trước"
    with open(meta_cache, "rb") as f:
        image_meta, gt_data = pickle.load(f)

    val_images = sorted([p for p in VAL_IMG_DIR.iterdir()
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png")])

    summary = []
    for name, weights_path in MODELS:
        if not weights_path.exists():
            print(f"[SKIP] {name}: không có {weights_path}")
            continue
        print(f"\n=== {name} ===")
        print(f"  weights: {weights_path}")

        # Load or create predictions cache
        pred_cache = ERR_DIR / f"predictions_{weights_path.parent.parent.name}.pkl"
        if pred_cache.exists():
            print(f"  [CACHE HIT] {pred_cache}")
            with open(pred_cache, "rb") as f:
                preds = pickle.load(f)
        else:
            print(f"  [CACHE MISS] Running inference ...")
            preds = inference(weights_path, val_images)
            with open(pred_cache, "wb") as f:
                pickle.dump(preds, f)

        # yolo val for official metrics
        print(f"  Running yolo val ...")
        model = YOLO(str(weights_path))
        r = model.val(data=str(DATA_YAML), imgsz=512, batch=32,
                       verbose=False, plots=False)

        # Count TP/FP/FN using matching
        tp = fp = fn = cc = total_gt = 0
        for img_name, gts in gt_data.items():
            m = match_image(preds.get(img_name, []), gts)
            tp += m["n_tp"]; fp += m["n_fp"]; cc += m["n_cc"]
            fn += m["n_fn"]; total_gt += len(gts)

        row = {
            "model": name,
            "mAP50": round(r.box.map50, 4),
            "mAP50-95": round(r.box.map, 4),
            "P": round(r.box.mp, 4),
            "R": round(r.box.mr, 4),
            "motorcycle_mAP50": round(r.box.ap50[0], 4),
            "car_mAP50":        round(r.box.ap50[1], 4),
            "bus_mAP50":        round(r.box.ap50[2], 4),
            "truck_mAP50":      round(r.box.ap50[3], 4),
            "TP": tp, "FP": fp, "FN": fn, "class_confused": cc,
            "precision": round(tp / max(tp + fp + cc, 1), 4),
            "recall":    round(tp / max(total_gt, 1), 4),
            "f1":        round(2 * (tp/max(tp+fp+cc,1)) * (tp/max(total_gt,1)) /
                               max((tp/max(tp+fp+cc,1)) + (tp/max(total_gt,1)), 1e-9), 4),
        }
        summary.append(row)
        print(f"  mAP@0.5 = {row['mAP50']}, mAP@0.5:0.95 = {row['mAP50-95']}, "
              f"F1 = {row['f1']}, Recall = {row['recall']}")

    # Save comparison table
    df = pd.DataFrame(summary)
    df.to_csv(ERR_DIR / "all_models_comparison.csv", index=False)
    print(f"\n[SAVED] {ERR_DIR / 'all_models_comparison.csv'}")
    print(df.to_string(index=False))

    # Chart: mAP@0.5 + Recall per model
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(df))
    w = 0.35
    axes[0].bar(x - w/2, df["mAP50"], w, label="mAP@0.5", color="#3498db")
    axes[0].bar(x + w/2, df["mAP50-95"], w, label="mAP@0.5:0.95", color="#e74c3c")
    axes[0].set_xticks(x); axes[0].set_xticklabels(df["model"], rotation=20, ha="right")
    axes[0].set_ylabel("mAP")
    axes[0].set_title("mAP comparison across models")
    axes[0].legend(); axes[0].grid(alpha=0.3, axis="y")
    axes[0].set_ylim(0.5, 1.0)

    axes[1].bar(x - w/2, df["precision"], w, label="Precision", color="#9b59b6")
    axes[1].bar(x + w/2, df["recall"], w, label="Recall", color="#27ae60")
    axes[1].set_xticks(x); axes[1].set_xticklabels(df["model"], rotation=20, ha="right")
    axes[1].set_title("Precision / Recall comparison")
    axes[1].legend(); axes[1].grid(alpha=0.3, axis="y")
    axes[1].set_ylim(0.5, 1.0)

    plt.tight_layout()
    plt.savefig(ERR_DIR / "all_models_chart.png", dpi=120, bbox_inches="tight")
    print(f"[SAVED] {ERR_DIR / 'all_models_chart.png'}")

    # Export JSON summary
    with open(ERR_DIR / "all_models_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {ERR_DIR / 'all_models_summary.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
