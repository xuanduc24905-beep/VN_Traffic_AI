#!/bin/bash
# Wrapper: train V4 → copy weights → run notebook final_report.ipynb
# Có logging timing, status check, auto-resume, safe exit on error.
#
# Chạy foreground (xem log realtime):
#     bash scripts/utils/train_v4_then_notebook.sh
#
# Chạy background (cắm máy đi làm việc khác):
#     nohup bash scripts/utils/train_v4_then_notebook.sh > logs/v4_pipeline.log 2>&1 &
#     disown
#
# Check progress:
#     tail -f logs/v4_pipeline.log       # overall
#     tail -f logs/train_v4.log          # training detail
#     wc -l runs/detect/train_v8s_v4_vn_cctv/results.csv  # trừ 1 = epoch done

set -e  # exit nếu bất cứ step nào fail
cd /home/xuand/vn_traffic_ai

PY=/home/xuand/miniconda3/envs/vehicle_env/bin/python
JUP=/home/xuand/miniconda3/envs/vehicle_env/bin/jupyter

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

START_TS=$(date +%s)
log "===== PIPELINE START ====="
log "Working dir: $(pwd)"
log "Python: $PY"
log ""

# -------- Step 1: Train V4 --------
STEP1_START=$(date +%s)
log "===== Step 1/3: Train V4 (yolov8s, 49.7k ảnh, 40 epochs) ====="
log "Expected: 6-7h fresh | ít hơn nếu resume"
log "Log file: logs/train_v4.log"
log ""

$PY scripts/training/v4_vn_only.py 2>&1 | tee logs/train_v4.log

STEP1_END=$(date +%s)
STEP1_MIN=$(( (STEP1_END - STEP1_START) / 60 ))
log ""
log "Step 1 done in ${STEP1_MIN} phút ($(( STEP1_MIN / 60 ))h $((STEP1_MIN % 60))m)"

# Guard: nếu best.pt chưa có → fail hard
BEST=runs/detect/train_v8s_v4_vn_cctv/weights/best.pt
if [ ! -f "$BEST" ]; then
    log "ERROR: $BEST không tồn tại. Training có thể fail."
    log "Check logs/train_v4.log để debug."
    exit 1
fi

# -------- Step 2: Copy weights --------
STEP2_START=$(date +%s)
log ""
log "===== Step 2/3: Copy best.pt → weights/trained/ ====="
cp "$BEST" weights/trained/v4_vn_cctv.pt
log "Copied: weights/trained/v4_vn_cctv.pt ($(du -h weights/trained/v4_vn_cctv.pt | cut -f1))"

# Quick peek metrics từ results.csv
log ""
log "=== V4 best metrics (từ results.csv) ==="
$PY -c "
import pandas as pd
df = pd.read_csv('runs/detect/train_v8s_v4_vn_cctv/results.csv')
df.columns = [c.strip() for c in df.columns]
best = df.loc[df['metrics/mAP50(B)'].idxmax()]
print(f'  Best epoch {int(best[\"epoch\"])}: mAP50={best[\"metrics/mAP50(B)\"]:.4f}, mAP50-95={best[\"metrics/mAP50-95(B)\"]:.4f}')
print(f'  P={best[\"metrics/precision(B)\"]:.4f}  R={best[\"metrics/recall(B)\"]:.4f}')
print(f'  Total time: {df[\"time\"].iloc[-1]/3600:.2f}h')
" 2>/dev/null || log "(không đọc được metrics)"

STEP2_END=$(date +%s)
log "Step 2 done in $(( STEP2_END - STEP2_START )) giây"

# -------- Step 3: Execute notebook --------
STEP3_START=$(date +%s)
log ""
log "===== Step 3/3: Execute notebook final_report.ipynb ====="
log "Expected: 20-30 phút (eval 6 model + demo pipeline + benchmark)"

$JUP nbconvert --to notebook --execute notebooks/final_report.ipynb \
    --output notebooks/final_report.ipynb --inplace \
    --ExecutePreprocessor.timeout=1800 \
    --ExecutePreprocessor.kernel_name=vehicle_env

STEP3_END=$(date +%s)
STEP3_MIN=$(( (STEP3_END - STEP3_START) / 60 ))
log "Step 3 done in ${STEP3_MIN} phút"

# -------- Summary --------
TOTAL_SEC=$(( STEP3_END - START_TS ))
TOTAL_MIN=$(( TOTAL_SEC / 60 ))
TOTAL_H=$(( TOTAL_MIN / 60 ))
TOTAL_LEFTOVER=$(( TOTAL_MIN % 60 ))

log ""
log "===== PIPELINE DONE ====="
log "Tổng thời gian: ${TOTAL_H}h ${TOTAL_LEFTOVER}m (${TOTAL_SEC} giây)"
log ""
log "Artifacts:"
log "  Model:  weights/trained/v4_vn_cctv.pt"
log "  Notebook: notebooks/final_report.ipynb (executed with outputs)"
log "  Charts: results/final_report/charts/ ($(ls results/final_report/charts/ 2>/dev/null | wc -l) files)"
log "  Tables: results/final_report/tables/ ($(ls results/final_report/tables/ 2>/dev/null | wc -l) files)"
log "  Videos: results/final_report/videos/ ($(ls results/final_report/videos/ 2>/dev/null | wc -l) files)"
log "  Summary: results/final_report/summary.json"
log ""
log "Done. Open notebook để xem output đầy đủ."
