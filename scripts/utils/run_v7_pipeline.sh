#!/bin/bash
# V7 pipeline: import DETRAC bus/truck subset → build V7 → train 1280 warm từ V5+
# → full cross-eval all models (bao gồm V7).
#
# Yêu cầu trước khi chạy:
#   - V5+ đã train xong (runs/detect/train_v8s_v5plus_960/weights/best.pt)
#   - UA-DETRAC data có ở /home/xuand/yolov8_finetune/data/detrac/
#
# Chạy foreground (xem realtime):
#     bash scripts/utils/run_v7_pipeline.sh
#
# Chạy background (recommended):
#     nohup bash scripts/utils/run_v7_pipeline.sh > logs/pipeline_v7.log 2>&1 &
#     disown
#
# Track:
#     tail -f logs/pipeline_v7.log           # overall
#     tail -f logs/import_detrac.log         # Phase 1
#     tail -f logs/v7_1280.log               # Phase 2 (training, LONGEST)
#     tail -f logs/eval_v7.log               # Phase 3
#
# Smart resume: nếu interrupt, chạy lại cùng lệnh. Mỗi script tự check state.
#
# Expected total time: ~2.5-3.5h (import 5m + build 1m + train 1280 ~2-3h + eval 20m)

set -e
cd /home/xuand/vn_traffic_ai

PY=/home/xuand/miniconda3/envs/vehicle_env/bin/python
mkdir -p logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

START_TS=$(date +%s)
log "===== V7 PIPELINE START ====="
log "Working dir: $(pwd)"
log ""

# ========== Phase 1: Import DETRAC bus/truck subset ==========
PHASE1_START=$(date +%s)
log "===== Phase 1/4: Import DETRAC bus/truck subset ====="
log "Expected: ~5 phút. Log: logs/import_detrac.log"
log ""

$PY scripts/data_prep/import_detrac_bustruck.py 2>&1 | tee logs/import_detrac.log

if [ ! -d "data/detrac_bustruck_subset/labels" ] || \
   [ -z "$(ls data/detrac_bustruck_subset/labels 2>/dev/null)" ]; then
    log "ERROR: DETRAC subset không được sinh. Check logs/import_detrac.log"
    exit 1
fi

PHASE1_END=$(date +%s)
log ""
log "Phase 1 done in $(( (PHASE1_END - PHASE1_START) / 60 )) phút"

# ========== Phase 2: Build V7 merged ==========
PHASE2_START=$(date +%s)
log ""
log "===== Phase 2/4: Build V7 merged (V5 train + DETRAC subset) ====="
log ""

$PY scripts/data_prep/build_v7_merged.py 2>&1 | tee logs/build_v7.log

if [ ! -f "data/data_v7.yaml" ]; then
    log "ERROR: data_v7.yaml không sinh được."
    exit 1
fi

PHASE2_END=$(date +%s)
log "Phase 2 done in $(( (PHASE2_END - PHASE2_START) )) giây"

# ========== Phase 3: Train V7 @ 1280 warm từ V5+ ==========
PHASE3_START=$(date +%s)
log ""
log "===== Phase 3/4: Train V7 (yolov8s @ 1280, batch 8, warm từ V5+) ====="
log "Expected: ~2-3h (batch 8 @ 1280 chậm hơn 960). Log: logs/v7_1280.log"
log ""

$PY scripts/training/v7_train_1280.py 2>&1 | tee logs/v7_1280.log

V7_BEST=runs/detect/train_v8s_v7_1280/weights/best.pt
if [ ! -f "$V7_BEST" ]; then
    log "ERROR: $V7_BEST không tồn tại. V7 training fail."
    exit 1
fi

PHASE3_END=$(date +%s)
PHASE3_MIN=$(( (PHASE3_END - PHASE3_START) / 60 ))
log ""
log "Phase 3 done in ${PHASE3_MIN} phút"

$PY -c "
import pandas as pd
df = pd.read_csv('runs/detect/train_v8s_v7_1280/results.csv')
df.columns = [c.strip() for c in df.columns]
best = df.loc[df['metrics/mAP50(B)'].idxmax()]
print(f'  V7 best epoch {int(best[\"epoch\"])}: mAP50={best[\"metrics/mAP50(B)\"]:.4f}, mAP50-95={best[\"metrics/mAP50-95(B)\"]:.4f}')
" 2>/dev/null || log "(không đọc được metrics V7)"

# ========== Phase 4: Full cross-eval (include V7) ==========
PHASE4_START=$(date +%s)
log ""
log "===== Phase 4/4: Full cross-eval all models (V7 included) ====="
log "Expected: ~20-30 phút. Log: logs/eval_v7.log"
log ""

$PY scripts/eval/eval_all_models_full.py 2>&1 | tee logs/eval_v7.log

PHASE4_END=$(date +%s)
PHASE4_MIN=$(( (PHASE4_END - PHASE4_START) / 60 ))
log "Phase 4 done in ${PHASE4_MIN} phút"

# ========== Summary ==========
TOTAL_SEC=$(( PHASE4_END - START_TS ))
TOTAL_MIN=$(( TOTAL_SEC / 60 ))
TOTAL_H=$(( TOTAL_MIN / 60 ))

log ""
log "===== V7 PIPELINE DONE ====="
log "Tổng thời gian: ${TOTAL_H}h $(( TOTAL_MIN % 60 ))m"
log ""
log "Breakdown:"
log "  Phase 1 (import DETRAC): $(( (PHASE1_END - PHASE1_START) / 60 )) phút"
log "  Phase 2 (build V7):      $(( (PHASE2_END - PHASE2_START) )) giây"
log "  Phase 3 (train V7):      ${PHASE3_MIN} phút"
log "  Phase 4 (full eval):     ${PHASE4_MIN} phút"
log ""
log "Artifacts:"
log "  V7 weights: runs/detect/train_v8s_v7_1280/weights/best.pt"
log "  DETRAC subset: data/detrac_bustruck_subset/ (manifest.json)"
log "  V7 dataset: data/merged_v7/"
log "  Final eval: results/v5_cross_eval/FINAL_COMPARISON.md (updated with V7)"
log ""
log "Mở FINAL_COMPARISON.md để xem V7 vs V5+ vs V6 vs baseline."
