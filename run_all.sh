#!/bin/bash
# VN Traffic AI — All-in-one pipeline script
#
# Tự động chạy đầy đủ:
#   0. Pre-flight check (env, GPU, data, weights)
#   1. Merge dataset v4 (nếu chưa có)
#   2. Train/fine-tune V4 (nếu best.pt chưa có)
#   3. Execute notebook final_report.ipynb
#   4. Report artifacts
#
# Chạy:
#   bash run_all.sh                   # Mặc định: fine-tune V3→V4 (nhanh, 2.5-3h)
#   bash run_all.sh --mode=fresh      # Train V4 fresh từ COCO (6-7h, có thể tốt hơn)
#   bash run_all.sh --mode=notebook   # Chỉ notebook (bỏ training, dùng V3 hiện có)
#   bash run_all.sh --help
#
# Chạy nền (recommend khi training lâu):
#   nohup bash run_all.sh > logs/run_all.log 2>&1 &
#   disown
#   echo "PID: $!"

set -e
cd "$(dirname "$0")"

# -------- Config --------
PY=/home/xuand/miniconda3/envs/vehicle_env/bin/python
JUP=/home/xuand/miniconda3/envs/vehicle_env/bin/jupyter

MODE="finetune"   # finetune | fresh | notebook

# Parse args
for arg in "$@"; do
    case $arg in
        --mode=*) MODE="${arg#*=}" ;;
        --help|-h)
            cat <<EOF
Usage: bash run_all.sh [--mode=finetune|fresh|notebook]

Modes:
  finetune (default)  Fine-tune V3 → V4 với vn_cctv data. ~2.5-3h.
                       Warm từ V3 best.pt, lr=0.001, 15 epochs, patience=8.
                       Low risk, expected mAP50 +1-2%.

  fresh                Train V4 from scratch (COCO init). ~6.5-7.5h.
                       lr=0.01, 40 epochs, patience=15.
                       Higher potential nhưng risky.

  notebook             Skip training, chỉ chạy notebook với weights hiện có.
                       ~20-30 phút. Dùng khi đã train xong trước đó.

Prerequisites:
  - weights/v3_vn_best.pt (V3 current best)
  - data/merged_v4/ hoặc data/raw_vn_cctv/ (sẽ auto merge nếu thiếu)

Artifacts output:
  - weights/trained/v4_vn_cctv.pt (model V4)
  - notebooks/final_report.ipynb (executed)
  - results/final_report/{charts,tables,videos}/
EOF
            exit 0
            ;;
    esac
done

# -------- Helpers --------
log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

banner() {
    echo ""
    echo "============================================================"
    echo "  $*"
    echo "============================================================"
}

check_file() {
    if [ ! -f "$1" ]; then
        log "ERROR: Missing file $1"
        return 1
    fi
    return 0
}

# -------- 0. Pre-flight --------
START_TS=$(date +%s)

banner "VN TRAFFIC AI — ALL-IN-ONE PIPELINE"
log "Mode: $MODE"
log "Working dir: $(pwd)"
log "Python: $PY"
log ""

banner "Step 0: Pre-flight check"

# Python env
if [ ! -x "$PY" ]; then
    log "ERROR: Python not found at $PY"
    exit 1
fi
log "✓ Python env: $PY"

# GPU
if ! command -v nvidia-smi &> /dev/null; then
    log "WARN: nvidia-smi not found, GPU training sẽ fail"
else
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    log "✓ GPU: ${FREE_MEM}MB free, ${TEMP}°C"
    if [ "$FREE_MEM" -lt 8000 ]; then
        log "WARN: GPU chỉ còn ${FREE_MEM}MB free (<8GB). Training có thể OOM."
    fi
fi

# Scripts/weights
if [ "$MODE" != "notebook" ]; then
    check_file "weights/v3_vn_best.pt" || log "WARN: Cần V3 best.pt cho fine-tune. Fallback fresh mode."
    check_file "yolov8s.pt" || { log "ERROR: Cần yolov8s.pt (COCO pretrained)"; exit 1; }
fi

check_file "notebooks/final_report.ipynb" || { log "ERROR: Cần notebook"; exit 1; }

log ""

# -------- 1. Merge data (if needed) --------
banner "Step 1: Dataset check"

if [ ! -f "data/data_v4.yaml" ] || [ ! -d "data/merged_v4/images/train" ]; then
    if [ ! -d "data/raw_vn_cctv/train/images" ]; then
        log "ERROR: Chưa có data/raw_vn_cctv/. Giải nén zip trước:"
        log "       cd data && unzip -q vn_cctv_traffic.v1i.yolov8.zip -d raw_vn_cctv/"
        exit 1
    fi
    log "merged_v4 chưa có → chạy merge_v4.py..."
    MERGE_START=$(date +%s)
    $PY scripts/data_prep/merge_v4.py
    log "Merge done in $(( ($(date +%s) - MERGE_START) / 60 )) phút"
else
    N_TRAIN=$(ls data/merged_v4/images/train | wc -l)
    log "✓ merged_v4 OK: $N_TRAIN ảnh train"
fi
log ""

# -------- 2. Train (if needed) --------
if [ "$MODE" = "notebook" ]; then
    log "Skip training (mode=notebook)"
elif [ -f "weights/trained/v4_vn_cctv.pt" ] && [ "${FORCE_TRAIN:-0}" != "1" ]; then
    log "✓ V4 weights đã tồn tại: weights/trained/v4_vn_cctv.pt"
    log "  → Skip training (set FORCE_TRAIN=1 để train lại)"
else
    STEP_NAME="Fine-tune V3→V4 (15 ep, lr=0.001, ~2.5h)"
    SCRIPT="scripts/training/v4_finetune_from_v3.py"
    RUN_DIR="runs/detect/train_v8s_v4_finetune"
    LOG_FILE="logs/train_v4_ft.log"

    if [ "$MODE" = "fresh" ]; then
        STEP_NAME="Train V4 fresh (40 ep, COCO init, ~7h)"
        SCRIPT="scripts/training/v4_vn_only.py"
        RUN_DIR="runs/detect/train_v8s_v4_vn_cctv"
        LOG_FILE="logs/train_v4.log"
    fi

    banner "Step 2: $STEP_NAME"
    log "Script: $SCRIPT"
    log "Log file: $LOG_FILE"
    log "Progress: wc -l $RUN_DIR/results.csv (trừ 1 = epoch done)"
    log ""

    TRAIN_START=$(date +%s)
    mkdir -p logs
    $PY -u "$SCRIPT" 2>&1 | tee "$LOG_FILE"
    TRAIN_END=$(date +%s)
    TRAIN_MIN=$(( (TRAIN_END - TRAIN_START) / 60 ))
    log ""
    log "Training done in ${TRAIN_MIN} phút ($(( TRAIN_MIN / 60 ))h $((TRAIN_MIN % 60))m)"

    BEST="$RUN_DIR/weights/best.pt"
    if [ ! -f "$BEST" ]; then
        log "ERROR: $BEST không tồn tại. Training fail."
        exit 1
    fi

    mkdir -p weights/trained
    cp "$BEST" weights/trained/v4_vn_cctv.pt
    log "✓ Copied: weights/trained/v4_vn_cctv.pt ($(du -h weights/trained/v4_vn_cctv.pt | cut -f1))"

    # Show best metrics
    log ""
    log "=== V4 best metrics ==="
    $PY -c "
import pandas as pd
df = pd.read_csv('$RUN_DIR/results.csv')
df.columns = [c.strip() for c in df.columns]
best = df.loc[df['metrics/mAP50(B)'].idxmax()]
print(f'  Best epoch {int(best[\"epoch\"])}: mAP50={best[\"metrics/mAP50(B)\"]:.4f}, mAP50-95={best[\"metrics/mAP50-95(B)\"]:.4f}')
print(f'  P={best[\"metrics/precision(B)\"]:.4f}  R={best[\"metrics/recall(B)\"]:.4f}')
print(f'  Total train time: {df[\"time\"].iloc[-1]/3600:.2f}h')
" 2>/dev/null || true
fi
log ""

# -------- 3. Notebook --------
banner "Step 3: Execute notebook final_report.ipynb"
log "Expected: 20-30 phút (eval 6 model + pipeline demo + benchmark)"
log ""

NB_START=$(date +%s)
$JUP nbconvert --to notebook --execute notebooks/final_report.ipynb \
    --output notebooks/final_report.ipynb --inplace \
    --ExecutePreprocessor.timeout=1800 \
    --ExecutePreprocessor.kernel_name=vehicle_env
NB_END=$(date +%s)
NB_MIN=$(( (NB_END - NB_START) / 60 ))
log "Notebook done in ${NB_MIN} phút"
log ""

# -------- 4. Summary --------
END_TS=$(date +%s)
TOTAL_SEC=$(( END_TS - START_TS ))
TOTAL_MIN=$(( TOTAL_SEC / 60 ))
TOTAL_H=$(( TOTAL_MIN / 60 ))
TOTAL_LEFTOVER=$(( TOTAL_MIN % 60 ))

banner "ALL DONE — Tổng ${TOTAL_H}h ${TOTAL_LEFTOVER}m"
log "Artifacts:"
if [ -f "weights/trained/v4_vn_cctv.pt" ]; then
    log "  ✓ Model V4: weights/trained/v4_vn_cctv.pt ($(du -h weights/trained/v4_vn_cctv.pt | cut -f1))"
fi
log "  ✓ Notebook: notebooks/final_report.ipynb (executed with cell outputs)"
if [ -d "results/final_report/charts" ]; then
    log "  ✓ Charts: $(ls results/final_report/charts/ 2>/dev/null | wc -l) files"
fi
if [ -d "results/final_report/tables" ]; then
    log "  ✓ Tables: $(ls results/final_report/tables/ 2>/dev/null | wc -l) files"
fi
if [ -d "results/final_report/videos" ]; then
    log "  ✓ Videos: $(ls results/final_report/videos/ 2>/dev/null | wc -l) files"
fi
if [ -f "results/final_report/summary.json" ]; then
    log "  ✓ Summary: results/final_report/summary.json"
    log ""
    log "=== Final summary ==="
    cat results/final_report/summary.json | $PY -m json.tool 2>/dev/null | head -40 || cat results/final_report/summary.json
fi

log ""
log "Mở notebook để xem outputs:"
log "  jupyter notebook notebooks/final_report.ipynb"
