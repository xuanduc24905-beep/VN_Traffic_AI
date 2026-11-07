"""So sánh 3 model trên cùng test set V5 gold:
  1. YOLOv8s vanilla (COCO pretrained)
  2. V5 (fine-tune từ YOLOv8s, 4 classes VN)
  3. V6 (train với pseudo-label HCM, 4 classes VN)

Output: bảng markdown mAP, per-class AP cho tiểu luận.

Chạy:
  python scripts/pseudo/10_compare_models.py
  python scripts/pseudo/10_compare_models.py --split val   # eval trên val thay vì test
"""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent.parent

# Mapping COCO class → V5 class (cho yolov8s.pt vanilla)
# COCO: 2=car, 3=motorcycle, 5=bus, 7=truck
# V5:   0=motorcycle, 1=car, 2=bus, 3=truck
COCO_TO_V5 = {3: 0, 2: 1, 5: 2, 7: 3}

MODELS = {
    "YOLOv8s (COCO, vanilla)": {
        "path": "yolov8s.pt",
        "coco_remap": True,   # cần remap COCO → V5 classes
    },
    "V5 (fine-tuned)": {
        "path": "runs/detect/train_v8s_v5_honest/weights/best.pt",
        "coco_remap": False,
    },
    "V6 (with pseudo-label)": {
        "path": "runs/detect/train_v8s_v6_pseudo/weights/best.pt",
        "coco_remap": False,
    },
}

DATA_YAML = ROOT / "data" / "data_v5.yaml"  # test set gold (không đổi)


def eval_model(name: str, model_info: dict, split: str) -> dict:
    """Chạy val trên 1 model, trả dict metrics."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {name}")
    print(f"  Model: {model_info['path']}")
    print(f"{'='*60}")

    model = YOLO(model_info["path"])

    if model_info["coco_remap"]:
        # YOLOv8s vanilla: chỉ eval 4 class COCO tương đương
        # Dùng param `classes` để filter COCO output
        print("  [COCO mode] filter classes {2,3,5,7} + remap to V5 schema")
        results = model.val(
            data=str(DATA_YAML),
            split=split,
            classes=[2, 3, 5, 7],   # COCO classes tương đương V5
            verbose=False,
            device=0,
            batch=8,
            plots=False,
        )
    else:
        results = model.val(
            data=str(DATA_YAML),
            split=split,
            verbose=False,
            device=0,
            batch=8,
            plots=False,
        )

    # Extract metrics
    metrics = {
        "model": name,
        "mAP@50":    float(results.box.map50),
        "mAP@50-95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall":    float(results.box.mr),
        "per_class_AP50": {},
    }

    # Per-class AP50
    # results.box.ap50 = numpy array shape (num_classes,)
    class_names = results.names  # dict idx -> name
    ap50_arr = results.box.ap50 if hasattr(results.box, 'ap50') else results.box.ap

    for idx, ap in enumerate(ap50_arr):
        if idx in class_names:
            metrics["per_class_AP50"][class_names[idx]] = float(ap)

    return metrics


def print_table(all_metrics: list[dict]):
    """In bảng markdown so sánh."""
    print(f"\n\n{'='*80}")
    print("  BẢNG SO SÁNH")
    print(f"{'='*80}\n")

    # Header
    model_names = [m["model"] for m in all_metrics]
    print("| Metric | " + " | ".join(model_names) + " |")
    print("|" + "---|" * (len(model_names) + 1))

    # Row: mAP@50
    row = ["mAP@50"] + [f"{m['mAP@50']:.4f}" for m in all_metrics]
    print("| " + " | ".join(row) + " |")

    # Row: mAP@50-95
    row = ["mAP@50-95"] + [f"{m['mAP@50-95']:.4f}" for m in all_metrics]
    print("| " + " | ".join(row) + " |")

    # Row: precision
    row = ["Precision"] + [f"{m['precision']:.4f}" for m in all_metrics]
    print("| " + " | ".join(row) + " |")

    # Row: recall
    row = ["Recall"] + [f"{m['recall']:.4f}" for m in all_metrics]
    print("| " + " | ".join(row) + " |")

    # Per-class AP50
    print("\n**Per-class AP@50:**\n")
    print("| Class | " + " | ".join(model_names) + " |")
    print("|" + "---|" * (len(model_names) + 1))

    all_classes = set()
    for m in all_metrics:
        all_classes.update(m["per_class_AP50"].keys())

    for cls in sorted(all_classes):
        row = [cls]
        for m in all_metrics:
            ap = m["per_class_AP50"].get(cls)
            row.append(f"{ap:.4f}" if ap is not None else "-")
        print("| " + " | ".join(row) + " |")


def main(split: str):
    all_metrics = []
    for name, info in MODELS.items():
        if not Path(info["path"]).exists() and not info["path"].startswith("yolov8"):
            print(f"[skip] {name}: không thấy weights")
            continue
        try:
            m = eval_model(name, info, split)
            all_metrics.append(m)
        except Exception as e:
            print(f"[err] {name}: {e}")

    print_table(all_metrics)

    # Save JSON cho dễ parse sau
    out = ROOT / "runs" / f"compare_v5_v6_yolov8s_{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "val"],
                    help="eval trên split nào (mặc định test)")
    args = ap.parse_args()
    main(args.split)
