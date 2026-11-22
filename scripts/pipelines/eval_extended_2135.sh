#!/usr/bin/env bash
# Đánh giá các checkpoint hiện có trên tập mở rộng 2.135 ảnh.
# Không huấn luyện lại: 1.500 ảnh V5+ test + 486 VN Aerial val + 149 VN Aerial test.

set -euo pipefail

PROJECT_ROOT="/home/xuand/vn_traffic_ai"
cd "$PROJECT_ROOT"

DEVICE="${DEVICE:-0}"
DATA_YAML="data/data_eval_extended.yaml"
OUT_JSON="docs/report/assets/extended_eval_results.json"
OUT_MD="docs/report/assets/extended_eval_table.md"

python3 - <<'PYEOF'
from pathlib import Path

parts = {"Extended test": Path("data/eval_extended_2135/images/test")}
exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
counts = {name: sum(p.suffix.lower() in exts for p in folder.iterdir()) for name, folder in parts.items()}
assert counts == {"Extended test": 2135}, counts
assert sum(counts.values()) == 2135
print("[AUDIT]", counts, "total=2135")
PYEOF

python3 - <<PYEOF
import json
from pathlib import Path
from ultralytics import YOLO

DEVICE = "$DEVICE"
DATA_YAML = "$DATA_YAML"
OUT_JSON = Path("$OUT_JSON")
OUT_MD = Path("$OUT_MD")
DATASET_NAME = "Extended test (V5+ 1.500 + VN Aerial 635 = 2.135)"

MODELS = [
    ("YOLOv8s-COCO (zero-shot)", "yolov8s.pt", 960),
    ("V5 honest (640)", "runs/detect/train_v8s_v5_honest/weights/best.pt", 640),
    ("V5+ 960px", "runs/detect/train_v8s_v5plus_960/weights/best.pt", 960),
    ("p4_scale", "runs/detect/p4_scale_from_baseline/weights/best.pt", 960),
    ("p4_imgsz1280", "runs/detect/p4_imgsz1280_from_baseline/weights/best.pt", 1280),
    ("V6 pseudo", "runs/detect/train_v8s_v6_pseudo/weights/best.pt", 960),
    ("r_p2 (ablation)", "runs/detect/r_p2/weights/best.pt", 960),
    ("r_clspw (mô hình cuối)", "runs/detect/r_clspw/weights/best.pt", 960),
    ("r_progressive (ablation)", "runs/detect/r_progressive/weights/best.pt", 1280),
]
CLASS_NAMES = ["motorcycle", "car", "bus", "truck"]
results = {}

for name, weights, imgsz in MODELS:
    if not Path(weights).exists():
        results[name] = {"status": "missing", "weights": weights}
        continue
    print(f"\n[EVAL] {name} imgsz={imgsz}")
    m = YOLO(weights).val(
        data=DATA_YAML,
        split="test",
        imgsz=imgsz,
        batch=16,
        device=DEVICE,
        verbose=False,
        plots=False,
    )
    ap50 = {c: float(m.box.ap50[i]) if i < len(m.box.ap50) else None for i, c in enumerate(CLASS_NAMES)}
    results[name] = {
        "status": "ok",
        "weights": weights,
        "dataset": DATASET_NAME,
        "imgsz": imgsz,
        "mAP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "AP50": ap50,
    }
    print(f"[RESULT] mAP50={m.box.map50:.6f} mAP50-95={m.box.map:.6f} P={m.box.mp:.6f} R={m.box.mr:.6f}")

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")

def fmt(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else "—"

lines = [
    "# Đánh giá mở rộng trên 2.135 ảnh",
    "",
    "Tập đánh giá gồm 1.500 ảnh V5+ test và 635 ảnh VN Aerial; không dùng để huấn luyện.",
    "",
    "| Mô hình | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | motorcycle | car | bus | truck |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for name, row in results.items():
    if row.get("status") != "ok":
        lines.append(f"| {name} | — | — | — | — | — | — | — | — |")
        continue
    ap = row["AP50"]
    lines.append(
        f"| {name} | {fmt(row['mAP50'])} | {fmt(row['mAP50_95'])} | {fmt(row['precision'])} | "
        f"{fmt(row['recall'])} | {fmt(ap['motorcycle'])} | {fmt(ap['car'])} | {fmt(ap['bus'])} | {fmt(ap['truck'])} |"
    )
OUT_MD.write_text("\n".join(lines) + "\n")
print(f"\n[SAVED] {OUT_JSON}\n[SAVED] {OUT_MD}")
PYEOF
