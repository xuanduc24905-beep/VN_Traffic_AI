#!/bin/bash
# Wrapper: fine-tune V4 (warm từ V3) → copy weights → run notebook
# Nhanh hơn train_v4_then_notebook.sh vì chỉ 15 epoch fine-tune (vs 40 fresh).
#
# Chạy foreground:
#     bash scripts/utils/finetune_v4_then_notebook.sh
#
# Chạy background:
#     nohup bash scripts/utils/finetune_v4_then_notebook.sh > logs/v4_ft_pipeline.log 2>&1 &
#     disown

set -e
cd /home/xuand/vn_traffic_ai

PY=/home/xuand/miniconda3/envs/vehicle_env/bin/python
JUP=/home/xuand/miniconda3/envs/vehicle_env/bin/jupyter

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

START_TS=$(date +%s)
log "===== PIPELINE START (V4 fine-tune) ====="
log "Working dir: $(pwd)"
log ""

# -------- Step 1: Fine-tune V4 --------
STEP1_START=$(date +%s)
log "===== Step 1/3: Fine-tune V4 (warm từ V3, 15 epoch, lr=0.001) ====="
log "Expected: 2-2.5h (vs 6-7h nếu fresh train)"
log "Log file: logs/train_v4_ft.log"
log ""

$PY scripts/training/v4_finetune_from_v3.py 2>&1 | tee logs/train_v4_ft.log

STEP1_END=$(date +%s)
STEP1_MIN=$(( (STEP1_END - STEP1_START) / 60 ))
log ""
log "Step 1 done in ${STEP1_MIN} phút ($(( STEP1_MIN / 60 ))h $((STEP1_MIN % 60))m)"

BEST=runs/detect/train_v8s_v4_finetune/weights/best.pt
if [ ! -f "$BEST" ]; then
    log "ERROR: $BEST không tồn tại."
    log "Check logs/train_v4_ft.log để debug."
    exit 1
fi

# -------- Step 2: Copy + metrics --------
STEP2_START=$(date +%s)
log ""
log "===== Step 2/3: Copy best.pt + so sánh V3 vs V4-ft ====="
cp "$BEST" weights/trained/v4_vn_cctv.pt
log "Copied: weights/trained/v4_vn_cctv.pt ($(du -h weights/trained/v4_vn_cctv.pt | cut -f1))"

log ""
log "=== Comparison V3 (train) vs V4-ft (train) ==="
$PY -c "
import pandas as pd
for name, csv in [('V3', 'runs/detect/train_v8s_v3_baseline/results.csv'),
                   ('V4-ft', 'runs/detect/train_v8s_v4_finetune/results.csv')]:
    try:
        df = pd.read_csv(csv)
        df.columns = [c.strip() for c in df.columns]
        best = df.loc[df['metrics/mAP50(B)'].idxmax()]
        print(f'  {name:<8}  epoch {int(best[\"epoch\"]):>2}/{len(df):>2}: '
              f'mAP50={best[\"metrics/mAP50(B)\"]:.4f}  '
              f'mAP50-95={best[\"metrics/mAP50-95(B)\"]:.4f}  '
              f'P={best[\"metrics/precision(B)\"]:.4f}  '
              f'R={best[\"metrics/recall(B)\"]:.4f}')
    except Exception as e:
        print(f'  {name}: error reading {csv} ({e})')
" 2>/dev/null || log "(không đọc được)"

STEP2_END=$(date +%s)
log "Step 2 done in $(( STEP2_END - STEP2_START )) giây"

# -------- Step 3: Execute notebook --------
STEP3_START=$(date +%s)
log ""
log "===== Step 3/3: Execute notebook final_report.ipynb ====="
log "Expected: 20-30 phút (eval 6 model + demo + benchmark)"

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
log "  Notebook: notebooks/final_report.ipynb (executed)"
log "  Charts: results/final_report/charts/ ($(ls results/final_report/charts/ 2>/dev/null | wc -l) files)"
log "  Tables: results/final_report/tables/ ($(ls results/final_report/tables/ 2>/dev/null | wc -l) files)"
log "  Videos: results/final_report/videos/ ($(ls results/final_report/videos/ 2>/dev/null | wc -l) files)"
log "  Summary: results/final_report/summary.json"
