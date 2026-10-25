#!/bin/bash
# Pipeline tuần tự: V5+ (imgsz 960) → pseudo-label vn_cctv → V6 (self-training)
# → full cross-eval all models.
#
# Chạy foreground (xem realtime):
#     bash scripts/utils/run_v5plus_v6_pipeline.sh
#
# Chạy background (cắm máy đi):
#     nohup bash scripts/utils/run_v5plus_v6_pipeline.sh > logs/pipeline_v6.log 2>&1 &
#     disown
#
# Track progress:
#     tail -f logs/pipeline_v6.log           # overall
#     tail -f logs/v5plus_960.log            # training V5+
#     tail -f logs/pseudo_label.log          # pseudo-labeling
#     tail -f logs/v6_pseudo.log             # training V6
#     tail -f logs/eval_all.log              # final eval
#
# Smart resume: nếu bị interrupt, chạy lại cùng lệnh. Mỗi script tự detect
# đã done chưa thông qua results.csv + best.pt.
#
# Expected total time: ~2.5-3h (V5+ 50m + pseudo 15m + V6 55m + eval 30m)

set -e
cd /home/xuand/vn_traffic_ai

PY=/home/xuand/miniconda3/envs/vehicle_env/bin/python
mkdir -p logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

START_TS=$(date +%s)
log "===== PIPELINE START ====="
log "Working dir: $(pwd)"
log ""

# ========== Phase 1: Train V5+ @ imgsz=960 ==========
PHASE1_START=$(date +%s)
log "===== Phase 1/4: Train V5+ (yolov8s @ imgsz=960, batch 16, 40ep) ====="
log "Expected: ~50 phút. Log: logs/v5plus_960.log"
log ""

$PY scripts/training/v5_plus_960.py 2>&1 | tee logs/v5plus_960.log

V5PLUS_BEST=runs/detect/train_v8s_v5plus_960/weights/best.pt
if [ ! -f "$V5PLUS_BEST" ]; then
    log "ERROR: $V5PLUS_BEST không tồn tại. V5+ training fail."
    exit 1
fi

PHASE1_END=$(date +%s)
PHASE1_MIN=$(( (PHASE1_END - PHASE1_START) / 60 ))
log ""
log "Phase 1 done in ${PHASE1_MIN} phút"
log ""

# Quick metric peek
$PY -c "
import pandas as pd
df = pd.read_csv('runs/detect/train_v8s_v5plus_960/results.csv')
df.columns = [c.strip() for c in df.columns]
best = df.loc[df['metrics/mAP50(B)'].idxmax()]
print(f'  V5+ best epoch {int(best[\"epoch\"])}: mAP50={best[\"metrics/mAP50(B)\"]:.4f}, mAP50-95={best[\"metrics/mAP50-95(B)\"]:.4f}')
" 2>/dev/null || log "(không đọc được metrics V5+)"

# ========== Phase 2: Pseudo-label vn_cctv ==========
PHASE2_START=$(date +%s)
log ""
log "===== Phase 2/4: Pseudo-label vn_cctv (V5+ predict + filter conf 0.75) ====="
log "Expected: ~15 phút. Log: logs/pseudo_label.log"
log ""

$PY scripts/data_prep/pseudo_label_vn_cctv.py 2>&1 | tee logs/pseudo_label.log

if [ ! -d "data/pseudo_vn_cctv/labels" ] || [ -z "$(ls data/pseudo_vn_cctv/labels 2>/dev/null)" ]; then
    log "ERROR: Pseudo-labels không được sinh. Check logs/pseudo_label.log"
    exit 1
fi

PHASE2_END=$(date +%s)
PHASE2_MIN=$(( (PHASE2_END - PHASE2_START) / 60 ))
log ""
log "Phase 2 done in ${PHASE2_MIN} phút"
log ""

# ========== Phase 3: Build V6 + Train ==========
PHASE3_START=$(date +%s)
log ""
log "===== Phase 3/4: Build V6 merged + Train V6 ====="
log ""

log "  3a. Build merged_v6..."
$PY scripts/data_prep/build_v6_merged.py 2>&1 | tee logs/build_v6.log

if [ ! -f "data/data_v6.yaml" ]; then
    log "ERROR: data_v6.yaml không được sinh."
    exit 1
fi

log ""
log "  3b. Train V6 (yolov8s @ 960, V5 + pseudo vn_cctv)"
log "  Expected: ~55 phút. Log: logs/v6_pseudo.log"
log ""

$PY scripts/training/v6_pseudo_lbl.py 2>&1 | tee logs/v6_pseudo.log

V6_BEST=runs/detect/train_v8s_v6_pseudo/weights/best.pt
if [ ! -f "$V6_BEST" ]; then
    log "ERROR: $V6_BEST không tồn tại. V6 training fail."
    exit 1
fi

PHASE3_END=$(date +%s)
PHASE3_MIN=$(( (PHASE3_END - PHASE3_START) / 60 ))
log ""
log "Phase 3 done in ${PHASE3_MIN} phút"

$PY -c "
import pandas as pd
df = pd.read_csv('runs/detect/train_v8s_v6_pseudo/results.csv')
df.columns = [c.strip() for c in df.columns]
best = df.loc[df['metrics/mAP50(B)'].idxmax()]
print(f'  V6 best epoch {int(best[\"epoch\"])}: mAP50={best[\"metrics/mAP50(B)\"]:.4f}, mAP50-95={best[\"metrics/mAP50-95(B)\"]:.4f}')
" 2>/dev/null || log "(không đọc được metrics V6)"

# ========== Phase 4: Full cross-eval ==========
PHASE4_START=$(date +%s)
log ""
log "===== Phase 4/4: Full cross-eval all models × all test sets ====="
log "Expected: ~15-30 phút. Log: logs/eval_all.log"
log ""

$PY scripts/eval/eval_all_models_full.py 2>&1 | tee logs/eval_all.log

PHASE4_END=$(date +%s)
PHASE4_MIN=$(( (PHASE4_END - PHASE4_START) / 60 ))
log ""
log "Phase 4 done in ${PHASE4_MIN} phút"

# ========== Summary ==========
TOTAL_SEC=$(( PHASE4_END - START_TS ))
TOTAL_MIN=$(( TOTAL_SEC / 60 ))
TOTAL_H=$(( TOTAL_MIN / 60 ))
TOTAL_LEFTOVER=$(( TOTAL_MIN % 60 ))

log ""
log "===== PIPELINE DONE ====="
log "Tổng thời gian: ${TOTAL_H}h ${TOTAL_LEFTOVER}m (${TOTAL_SEC} giây)"
log ""
log "Breakdown:"
log "  Phase 1 (V5+ 960):     ${PHASE1_MIN} phút"
log "  Phase 2 (pseudo-lbl):  ${PHASE2_MIN} phút"
log "  Phase 3 (V6 train):    ${PHASE3_MIN} phút"
log "  Phase 4 (full eval):   ${PHASE4_MIN} phút"
log ""
log "Artifacts:"
log "  V5+ weights: runs/detect/train_v8s_v5plus_960/weights/best.pt"
log "  V6 weights:  runs/detect/train_v8s_v6_pseudo/weights/best.pt"
log "  Pseudo data: data/pseudo_vn_cctv/ (manifest.json cho reproducibility)"
log "  V6 dataset:  data/merged_v6/"
log "  Final eval:  results/v5_cross_eval/FINAL_COMPARISON.md"
log "               results/v5_cross_eval/FINAL_COMPARISON.json"
log ""
log "Mở FINAL_COMPARISON.md để xem bảng tổng kết."
