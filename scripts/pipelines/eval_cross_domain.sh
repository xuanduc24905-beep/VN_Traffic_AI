#!/usr/bin/env bash
# Eval cả 2 bộ test: V5+ (CCTV ngang, 1500 ảnh) + Vietnam Vehicles aerial (635 ảnh)
#
# Mục đích: báo cáo cross-view generalization — model tốt trên CCTV có generalize
# sang aerial view không?
#
# Chạy sau khi ablation (r_p2, r_clspw, r_progressive) xong.

set -euo pipefail

PROJECT_ROOT="/home/xuand/vn_traffic_ai"
cd "$PROJECT_ROOT"

DEVICE="${DEVICE:-0}"
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_crossdomain_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

OUT_JSON="docs/report/assets/cross_domain_results.json"
OUT_MD="docs/report/assets/cross_domain_table.md"
mkdir -p "$(dirname "$OUT_JSON")"

echo "=== Eval cross-domain bắt đầu $(date) ==="
echo "Device: $DEVICE"

python3 - <<PYEOF
import json
from pathlib import Path
from ultralytics import YOLO

DEVICE = "$DEVICE"
OUT_JSON = "$OUT_JSON"
OUT_MD = "$OUT_MD"

MODELS = [
    # (hiển thị, path best.pt, imgsz train)
    ("YOLOv8s-COCO (zero-shot)", "yolov8s.pt",                                             960),
    ("V5 honest (640)",          "runs/detect/train_v8s_v5_honest/weights/best.pt",         640),
    ("V5+ 960px",                "runs/detect/train_v8s_v5plus_960/weights/best.pt",        960),
    ("p4_scale (cuối)",          "runs/detect/p4_scale_from_baseline/weights/best.pt",      960),
    ("p4_imgsz1280",             "runs/detect/p4_imgsz1280_from_baseline/weights/best.pt", 1280),
    ("V6 pseudo",                "runs/detect/train_v8s_v6_pseudo/weights/best.pt",         960),
    ("r_p2 (ablation)",          "runs/detect/r_p2/weights/best.pt",                        960),
    ("r_clspw (ablation)",       "runs/detect/r_clspw/weights/best.pt",                     960),
    ("r_progressive (ablation)", "runs/detect/r_progressive/weights/best.pt",              1280),
]

DATASETS = [
    ("V5+ test (CCTV, 1.500)",     "data/data_v5plus.yaml",   "test"),
    ("VN Aerial test (149)",       "data/data_vn_aerial.yaml", "test"),
    ("VN Aerial val (486)",        "data/data_vn_aerial.yaml", "val"),
]

CLASS_NAMES = ["motorcycle", "car", "bus", "truck"]
results = {}

for mname, weights, imgsz in MODELS:
    if not Path(weights).exists():
        print(f"[SKIP] {mname}: không có {weights}")
        results[mname] = {"status": "missing", "weights": weights}
        continue
    results[mname] = {"status": "ok", "weights": weights, "per_dataset": {}}
    print(f"\n════════ {mname} (imgsz={imgsz}) ════════")
    for dname, data_yaml, split in DATASETS:
        try:
            m = YOLO(weights).val(
                data=data_yaml, split=split, imgsz=imgsz,
                batch=16, device=DEVICE, verbose=False, plots=False,
            )
            ap50 = {c: float(m.box.ap50[i]) if i < len(m.box.ap50) else None
                    for i, c in enumerate(CLASS_NAMES)}
            r = {
                "mAP50":     float(m.box.map50),
                "mAP50_95":  float(m.box.map),
                "precision": float(m.box.mp),
                "recall":    float(m.box.mr),
                "AP50":      ap50,
            }
            results[mname]["per_dataset"][dname] = r
            print(f"  {dname:<30}  mAP50={r['mAP50']:.4f}  mAP50-95={r['mAP50_95']:.4f}")
        except Exception as e:
            results[mname]["per_dataset"][dname] = {"error": str(e)}
            print(f"  {dname}: ERROR {e}")

Path(OUT_JSON).write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(f"\n[SAVED] {OUT_JSON}")

# Bảng markdown tổng hợp
lines = [
    "# Cross-domain evaluation — V5+ CCTV (ngang) vs Vietnam Vehicles aerial (trên cao)",
    "",
    "## Bảng chính: mAP@0.5 trên các bộ test",
    "",
    "| Mô hình | V5+ test (CCTV) | Aerial test | Aerial val | Gap CCTV→Aerial |",
    "|---|---|---|---|---|",
]
def fmt(x): return f"{x:.4f}" if isinstance(x, (int, float)) else "—"

for mname, r in results.items():
    if r.get("status") != "ok":
        lines.append(f"| {mname} | — | — | — | — |")
        continue
    pd = r.get("per_dataset", {})
    v5p = pd.get("V5+ test (CCTV, 1.500)", {}).get("mAP50")
    aerial_t = pd.get("VN Aerial test (149)", {}).get("mAP50")
    aerial_v = pd.get("VN Aerial val (486)", {}).get("mAP50")
    gap = (v5p - aerial_v) if (v5p and aerial_v) else None
    gap_str = f"{gap:+.4f}" if gap is not None else "—"
    lines.append(f"| {mname} | {fmt(v5p)} | {fmt(aerial_t)} | {fmt(aerial_v)} | {gap_str} |")

lines.append("")
lines.append("## Bảng phụ: mAP@0.5:0.95 (strict metric)")
lines.append("")
lines.append("| Mô hình | V5+ test | Aerial test | Aerial val |")
lines.append("|---|---|---|---|")
for mname, r in results.items():
    if r.get("status") != "ok":
        lines.append(f"| {mname} | — | — | — |")
        continue
    pd = r.get("per_dataset", {})
    lines.append(f"| {mname} | "
                 f"{fmt(pd.get('V5+ test (CCTV, 1.500)', {}).get('mAP50_95'))} | "
                 f"{fmt(pd.get('VN Aerial test (149)', {}).get('mAP50_95'))} | "
                 f"{fmt(pd.get('VN Aerial val (486)', {}).get('mAP50_95'))} |")

Path(OUT_MD).write_text("\n".join(lines) + "\n")
print(f"[SAVED] {OUT_MD}")
print("\n=== KẾT QUẢ ===")
print("\n".join(lines))
PYEOF

echo ""
echo "=== Xong $(date) ==="
echo "Log: $LOG_FILE"
echo "JSON: $OUT_JSON"
echo "Markdown: $OUT_MD"
