#!/usr/bin/env bash
# Eval TẤT CẢ recipe trên TEST set (data_v5plus.yaml split=test).
#
# Mục đích: lấy số test mAP cho tiểu luận (val chỉ để chọn mô hình, test mới là số honest).
#
# Mô hình được eval:
#   v5plus_baseline           -> train_v8s_v5plus_960 (nền V5+ 960, test mAP ~0.782)
#   r_p2                      -> yolov8s-p2
#   r_clspw                   -> cls_pw=0.5
#   r_progressive             -> 960->1280 curriculum
#   p4_scale_from_baseline    -> đã train sẵn (scale=0.7, translate=0.15)
#   p4_imgsz1280_from_baseline-> đã train sẵn (imgsz=1280 từ baseline)
#
# Output:
#   docs/report/assets/ablation_test_results.json  (dữ liệu thô cho tiểu luận)
#   docs/report/assets/ablation_test_table.md      (bảng markdown copy vào tiểu luận)
#   runs/detect/<name>/eval_test/                  (confusion matrix + predictions per model)
#
# Dùng:   bash scripts/pipelines/eval_test_all.sh
# Device: DEVICE=0 bash scripts/pipelines/eval_test_all.sh

set -euo pipefail

PROJECT_ROOT="/home/xuand/vn_traffic_ai"
cd "$PROJECT_ROOT"

DATA_YAML="data/data_v5plus.yaml"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-16}"

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_test_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo "[$(date +'%H:%M:%S')] $*"; }
section() { echo ""; echo "═══════════════════════════════════════════════════"; echo "  $*"; echo "═══════════════════════════════════════════════════"; }

START_TIME=$(date +%s)
log "Eval test set bắt đầu. Log: $LOG_FILE"

if [ ! -f "$DATA_YAML" ]; then
    echo "ERROR: thiếu $DATA_YAML" >&2; exit 1
fi

OUT_JSON="docs/report/assets/ablation_test_results.json"
OUT_MD="docs/report/assets/ablation_test_table.md"
mkdir -p "$(dirname "$OUT_JSON")"

section "Eval tất cả recipe trên TEST set"

python3 - <<PYEOF
import json
from pathlib import Path
from ultralytics import YOLO

DATA_YAML = "$DATA_YAML"
DEVICE    = "$DEVICE"
BATCH     = int("$BATCH")
OUT_JSON  = "$OUT_JSON"
OUT_MD    = "$OUT_MD"

# (tên hiển thị, đường dẫn best.pt, imgsz dùng khi eval)
# imgsz eval = imgsz train (progressive stage2 = 1280, imgsz1280 = 1280, còn lại 960)
MODELS = [
    ("v5plus_baseline",            "runs/detect/train_v8s_v5plus_960/weights/best.pt",         960),
    ("r_p2",                       "runs/detect/r_p2/weights/best.pt",                          960),
    ("r_clspw",                    "runs/detect/r_clspw/weights/best.pt",                       960),
    ("r_progressive",              "runs/detect/r_progressive/weights/best.pt",                1280),
    ("p4_scale_from_baseline",     "runs/detect/p4_scale_from_baseline/weights/best.pt",        960),
    ("p4_imgsz1280_from_baseline", "runs/detect/p4_imgsz1280_from_baseline/weights/best.pt",   1280),
]

CLASS_NAMES = ["motorcycle", "car", "bus", "truck"]
results = {}

for name, weights, imgsz in MODELS:
    print(f"\n──────── {name} ────────")
    if not Path(weights).exists():
        print(f"[SKIP] không có {weights}")
        results[name] = {"status": "missing", "weights": weights}
        continue

    try:
        model = YOLO(weights)
        m = model.val(
            data=DATA_YAML,
            split="test",
            imgsz=imgsz,
            batch=BATCH,
            device=DEVICE,
            project=f"runs/detect/{name}",
            name="eval_test",
            exist_ok=True,
            save_json=True,
            plots=True,     # sinh confusion matrix + PR curve
            verbose=False,
        )
        ap50 = {}
        for i, cls in enumerate(CLASS_NAMES):
            ap50[cls] = float(m.box.ap50[i]) if i < len(m.box.ap50) else None

        results[name] = {
            "status":    "ok",
            "weights":   weights,
            "imgsz":     imgsz,
            "mAP50":     float(m.box.map50),
            "mAP50_95":  float(m.box.map),
            "precision": float(m.box.mp),
            "recall":    float(m.box.mr),
            "AP50":      ap50,
        }
        print(f"  mAP50:    {m.box.map50:.4f}")
        print(f"  mAP50-95: {m.box.map:.4f}")
        for cls, ap in ap50.items():
            print(f"  AP50 {cls:<12}: {ap:.4f}" if ap is not None else f"  AP50 {cls:<12}: n/a")
    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        results[name] = {"status": "error", "weights": weights, "error": str(e)}

# ─── Save JSON ───
Path(OUT_JSON).write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(f"\n[SAVED] {OUT_JSON}")

# ─── Bảng markdown cho tiểu luận ───
lines = [
    "# Kết quả ablation trên TEST set (data_v5plus)",
    "",
    "| Mô hình | mAP@50 | mAP@50-95 | P | R | AP motorcycle | AP car | AP bus | AP truck |",
    "|---|---|---|---|---|---|---|---|---|",
]
def fmt(x): return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"

for name, r in results.items():
    if r.get("status") != "ok":
        lines.append(f"| {name} | — | — | — | — | — | — | — | — | *({r.get('status')})* |")
        continue
    ap = r["AP50"]
    lines.append(
        f"| {name} | {fmt(r['mAP50'])} | {fmt(r['mAP50_95'])} | {fmt(r['precision'])} | {fmt(r['recall'])} "
        f"| {fmt(ap['motorcycle'])} | {fmt(ap['car'])} | {fmt(ap['bus'])} | {fmt(ap['truck'])} |"
    )

# ─── So sánh delta vs baseline ───
base = results.get("v5plus_baseline")
if base and base.get("status") == "ok":
    lines += ["", "## Delta so với v5plus_baseline", "",
              "| Mô hình | Δ mAP@50 | Δ motorcycle | Δ car | Δ bus | Δ truck |",
              "|---|---|---|---|---|---|"]
    for name, r in results.items():
        if name == "v5plus_baseline" or r.get("status") != "ok":
            continue
        dmap = r["mAP50"] - base["mAP50"]
        d_cls = {c: r["AP50"][c] - base["AP50"][c] for c in CLASS_NAMES
                 if r["AP50"][c] is not None and base["AP50"][c] is not None}
        def sgn(x): return f"+{x:.4f}" if x >= 0 else f"{x:.4f}"
        lines.append(
            f"| {name} | {sgn(dmap)} | {sgn(d_cls.get('motorcycle', 0))} "
            f"| {sgn(d_cls.get('car', 0))} | {sgn(d_cls.get('bus', 0))} | {sgn(d_cls.get('truck', 0))} |"
        )

Path(OUT_MD).write_text("\n".join(lines) + "\n")
print(f"[SAVED] {OUT_MD}")
PYEOF

section "Xong"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
log "Thời gian eval: $((ELAPSED / 60)) phút"
log "Log: $LOG_FILE"

echo ""
echo "Kết quả:"
echo "  1) $OUT_JSON           (dữ liệu thô JSON)"
echo "  2) $OUT_MD             (bảng markdown copy vào tiểu luận)"
echo "  3) runs/detect/<name>/eval_test/  (confusion matrix + PR curve per model)"
echo ""
echo "Bảng markdown:"
cat "$OUT_MD"
