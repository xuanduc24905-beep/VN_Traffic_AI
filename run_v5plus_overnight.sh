#!/usr/bin/env bash
# Full V5+ pipeline: build dataset → train warm-start → evaluate → compare với V5.
# Chạy qua đêm an toàn: set -e để dừng khi lỗi, log đầy đủ với timestamp.
#
# Dùng:   bash run_v5plus_overnight.sh
# Resume: chạy lại chính script này — mọi step đều idempotent (skip/resume).

set -euo pipefail

PROJECT_ROOT="/home/xuand/vn_traffic_ai"
cd "$PROJECT_ROOT"

# ─────────── Logging ───────────
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/v5plus_pipeline_${TS}.log"

# Redirect mọi output vào cả stdout + logfile
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo "[$(date +'%H:%M:%S')] $*"; }
section() { echo ""; echo "═══════════════════════════════════════════════════"; echo "  $*"; echo "═══════════════════════════════════════════════════"; }

START_TIME=$(date +%s)
log "V5+ pipeline bắt đầu. Log: $LOG_FILE"

# ─────────── Preflight checks ───────────
section "[0/5] Preflight checks"

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 không tìm thấy" >&2; exit 1
fi

log "Python: $(python3 --version)"
log "Ultralytics: $(python3 -c 'import ultralytics; print(ultralytics.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
log "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || log "  (no GPU, sẽ chạy CPU rất chậm)"

for f in \
    "data/raw_vietnam_vehicles_v10/data.yaml" \
    "data/merged_v5/images/train" \
    "yolov8s.pt" \
    "runs/detect/train_v8s_v5_honest/weights/best.pt" \
    ; do
    if [ ! -e "$f" ]; then
        echo "ERROR: thiếu file/thư mục bắt buộc: $f" >&2
        exit 1
    fi
done
log "Mọi file bắt buộc tồn tại ✓"

# ─────────── Step 1: Build V5+ (idempotent) ───────────
section "[1/5] Build V5+ dataset"
log "Chạy scripts/data_prep/build_v5plus_merged.py..."
python3 scripts/data_prep/build_v5plus_merged.py

TRAIN_N=$(ls data/merged_v5plus/images/train 2>/dev/null | wc -l)
VAL_N=$(ls data/merged_v5plus/images/val 2>/dev/null | wc -l)
TEST_N=$(ls data/merged_v5plus/images/test 2>/dev/null | wc -l)
log "V5+ ảnh: train=$TRAIN_N, val=$VAL_N, test=$TEST_N"

if [ "$TRAIN_N" -lt 5000 ]; then
    echo "ERROR: train quá ít ($TRAIN_N), có gì đó sai" >&2
    exit 1
fi

# ─────────── Step 2: Train warm-start ───────────
section "[2/5] Train V5+ warm-start (YOLOv8s, 40 epochs, imgsz=960)"
log "Scripts: scripts/training/v5plus_warmstart.py"
log "Ước tính ~1-2 giờ trên RTX 3500 Ada Laptop"
python3 scripts/training/v5plus_warmstart.py

BEST_PT="runs/detect/train_v8s_v5plus_warm/weights/best.pt"
if [ ! -f "$BEST_PT" ]; then
    echo "ERROR: $BEST_PT không tồn tại sau training" >&2
    exit 1
fi
log "Best weights: $BEST_PT ($(du -h $BEST_PT | cut -f1))"

# ─────────── Step 3: Evaluate trên V5 test (fixed) ───────────
section "[3/5] Evaluate V5+ model trên test set V5 (honest)"
EVAL_OUT="runs/detect/train_v8s_v5plus_warm/eval_v5_test"
rm -rf "$EVAL_OUT"

python3 - <<'PYEOF'
from pathlib import Path
from ultralytics import YOLO

model = YOLO("runs/detect/train_v8s_v5plus_warm/weights/best.pt")
metrics = model.val(
    data="data/data_v5plus.yaml",
    split="test",
    imgsz=960,
    batch=16,
    project="runs/detect/train_v8s_v5plus_warm",
    name="eval_v5_test",
    exist_ok=True,
    save_json=True,
    verbose=True,
)
print(f"\n[V5+ metrics on V5 test]")
print(f"  mAP50:    {metrics.box.map50:.4f}")
print(f"  mAP50-95: {metrics.box.map:.4f}")
for i, name in enumerate(["motorcycle", "car", "bus", "truck"]):
    ap = metrics.box.ap50[i] if i < len(metrics.box.ap50) else float("nan")
    print(f"  AP50 {name:<12}: {ap:.4f}")

# Lưu summary json
import json
summary = {
    "model": "V5+ warm-start",
    "mAP50": float(metrics.box.map50),
    "mAP50_95": float(metrics.box.map),
    "AP50_per_class": {
        name: float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else None
        for i, name in enumerate(["motorcycle", "car", "bus", "truck"])
    },
}
Path("runs/detect/train_v8s_v5plus_warm/eval_v5_test/summary.json").write_text(
    json.dumps(summary, indent=2)
)
PYEOF

# ─────────── Step 4: So sánh V5 vs V5+ ───────────
section "[4/5] So sánh V5 baseline vs V5+ trên cùng test set"

python3 - <<'PYEOF'
import json
from pathlib import Path
from ultralytics import YOLO

results = {}

for name, weights in [
    ("V5 baseline",   "runs/detect/train_v8s_v5_honest/weights/best.pt"),
    ("V5+ warm-start", "runs/detect/train_v8s_v5plus_warm/weights/best.pt"),
]:
    if not Path(weights).exists():
        print(f"[SKIP] {name}: {weights} không tồn tại")
        continue
    print(f"\n--- Eval {name} on V5+ test ---")
    model = YOLO(weights)
    m = model.val(
        data="data/data_v5plus.yaml",
        split="test",
        imgsz=960,
        batch=16,
        verbose=False,
    )
    results[name] = {
        "mAP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
        "AP50": {
            cls: float(m.box.ap50[i]) if i < len(m.box.ap50) else None
            for i, cls in enumerate(["motorcycle", "car", "bus", "truck"])
        },
    }

# In bảng so sánh
print("\n" + "=" * 70)
print(f"{'Model':<20} {'mAP50':>10} {'mAP50-95':>10} {'moto':>8} {'car':>8} {'bus':>8} {'truck':>8}")
print("-" * 70)
for name, r in results.items():
    ap = r["AP50"]
    def fmt(x): return f"{x:.3f}" if x is not None else "n/a"
    print(f"{name:<20} {r['mAP50']:>10.4f} {r['mAP50_95']:>10.4f} "
          f"{fmt(ap['motorcycle']):>8} {fmt(ap['car']):>8} "
          f"{fmt(ap['bus']):>8} {fmt(ap['truck']):>8}")
print("=" * 70)

Path("docs/report/assets/v5_vs_v5plus.json").parent.mkdir(parents=True, exist_ok=True)
Path("docs/report/assets/v5_vs_v5plus.json").write_text(json.dumps(results, indent=2))
print("\nĐã lưu: docs/report/assets/v5_vs_v5plus.json")
PYEOF

# ─────────── Step 5: Final summary ───────────
section "[5/5] Pipeline hoàn tất"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINS=$(((ELAPSED % 3600) / 60))

log "Tổng thời gian: ${HOURS}h ${MINS}m"
log "Log file: $LOG_FILE"
log "Best weights: $BEST_PT"
log "Eval results: runs/detect/train_v8s_v5plus_warm/eval_v5_test/"
log "Comparison: docs/report/assets/v5_vs_v5plus.json"

echo ""
echo "Sáng dậy mở 2 file này:"
echo "  1) $LOG_FILE  (toàn bộ log)"
echo "  2) docs/report/assets/v5_vs_v5plus.json  (bảng so sánh)"
