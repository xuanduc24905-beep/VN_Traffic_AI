#!/bin/bash
# Phase 4 — V7 pipeline (DETRAC bus/truck boost + yolov8s @ imgsz=1280).
#
# Pipeline này SELF-CONTAINED — tất cả script V7 nằm trong scripts/phase4_v7/.
# Chỉ reuse script eval tổng (scripts/eval/eval_all_models_full.py) vì shared
# giữa các pipeline (V5+/V6/V7 cùng dùng).
#
# Yêu cầu trước khi chạy:
#   - V5+ đã train xong: runs/detect/train_v8s_v5plus_960/weights/best.pt
#   - UA-DETRAC data có ở /home/xuand/yolov8_finetune/data/detrac/
#
# Chạy foreground (xem realtime):
#     bash scripts/phase4_v7/run_pipeline.sh
#
# Chạy background (recommended — pipeline ~2.5-3.5h):
#     nohup bash scripts/phase4_v7/run_pipeline.sh > logs/phase4_v7.log 2>&1 &
#     disown
#
# Track:
#     tail -f logs/phase4_v7.log               # overall
#     tail -f logs/phase4_01_import_detrac.log # Phase 4a: DETRAC import
#     tail -f logs/phase4_02_build_v7.log     # Phase 4b: V7 build
#     tail -f logs/phase4_03_train_v7.log     # Phase 4c: train (LONGEST)
#     tail -f logs/phase4_04_eval.log         # Phase 4d: full cross-eval
#     wc -l runs/detect/train_v8s_v7_1280/results.csv  # minus 1 = epoch done
#
# Smart resume: chạy lại cùng lệnh sẽ tự continue từ chỗ dừng.
# Expected total: ~2.5-3.5h.

set -e
cd /home/xuand/vn_traffic_ai

PY=/home/xuand/miniconda3/envs/vehicle_env/bin/python
HERE=scripts/phase4_v7
EVAL=scripts/eval/eval_all_models_full.py   # shared eval script
mkdir -p logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

START_TS=$(date +%s)
log "===== PHASE 4 (V7) PIPELINE START ====="
log "Working dir: $(pwd)"
log "Phase 4 scripts: $HERE/"
log ""

# Sanity check prerequisites
V5PLUS_BEST=runs/detect/train_v8s_v5plus_960/weights/best.pt
if [ ! -f "$V5PLUS_BEST" ]; then
    log "ERROR: Thiếu V5+ weights ($V5PLUS_BEST). Chạy phase 3 (V5+/V6 pipeline) trước."
    exit 1
fi
if [ ! -d "/home/xuand/yolov8_finetune/data/detrac/labels/train" ]; then
    log "ERROR: Thiếu UA-DETRAC data. Expected: /home/xuand/yolov8_finetune/data/detrac/"
    exit 1
fi
log "[OK] Prerequisites verified"
log ""

# ========== Phase 4a: Import DETRAC bus/truck subset ==========
P1_START=$(date +%s)
log "===== Phase 4a: Import DETRAC bus/truck subset ====="
log "Expected: ~5 phút. Log: logs/phase4_01_import_detrac.log"
log ""

$PY $HERE/import_detrac_bustruck.py 2>&1 | tee logs/phase4_01_import_detrac.log

if [ ! -d "data/detrac_bustruck_subset/labels" ] || \
   [ -z "$(ls data/detrac_bustruck_subset/labels 2>/dev/null)" ]; then
    log "ERROR: DETRAC subset không sinh được."
    exit 1
fi

P1_END=$(date +%s)
log ""
log "Phase 4a done in $(( (P1_END - P1_START) / 60 )) phút"

# ========== Phase 4b: Build V7 merged ==========
P2_START=$(date +%s)
log ""
log "===== Phase 4b: Build V7 merged (V5 train + DETRAC subset) ====="
log ""

$PY $HERE/build_v7_merged.py 2>&1 | tee logs/phase4_02_build_v7.log

if [ ! -f "data/data_v7.yaml" ]; then
    log "ERROR: data_v7.yaml không sinh được."
    exit 1
fi

P2_END=$(date +%s)
log "Phase 4b done in $(( P2_END - P2_START )) giây"

# ========== Phase 4b.5: Audit leak merged_v7 (SAFETY NET) ==========
P25_START=$(date +%s)
log ""
log "===== Phase 4b.5: Audit leak merged_v7 (fail hard nếu có leak) ====="
log ""

$PY scripts/data_prep/audit_leak.py \
    --datasets data/merged_v7 \
    --out docs/v3_report/leak_audit_v7.json \
    2>&1 | tee logs/phase4_025_audit_v7.log

# Parse leak % — fail hard nếu val hoặc test có bất kỳ leak nào
LEAK_PCT=$($PY -c "
import json
with open('docs/v3_report/leak_audit_v7.json') as f:
    d = json.load(f)[0]
val_pct = d['leak_metrics']['val']['pct_leaked']
test_pct = d['leak_metrics']['test']['pct_leaked']
print(max(val_pct, test_pct))
" 2>&1)

if [ -z "$LEAK_PCT" ] || [ "$(echo "$LEAK_PCT != 0.0" | bc 2>/dev/null)" = "1" ] || \
   [ "$LEAK_PCT" != "0.0" ]; then
    log "ERROR: V7 có leak! val/test max pct=$LEAK_PCT"
    log "       → DỪNG, không train phí thời gian."
    log "       Check docs/v3_report/leak_audit_v7.json để debug."
    exit 1
fi

log "[OK] V7 leak audit PASS (val=0%, test=0%)"
P25_END=$(date +%s)
log "Phase 4b.5 done in $(( P25_END - P25_START )) giây"

# ========== Phase 4c: Train V7 @ 1280 warm từ V5+ ==========
P3_START=$(date +%s)
log ""
log "===== Phase 4c: Train V7 (yolov8s @ 1280, batch 8, warm từ V5+) ====="
log "Expected: ~2-3h (batch 8 @ 1280 lâu nhất). Log: logs/phase4_03_train_v7.log"
log ""

$PY $HERE/v7_train_1280.py 2>&1 | tee logs/phase4_03_train_v7.log

V7_BEST=runs/detect/train_v8s_v7_1280/weights/best.pt
if [ ! -f "$V7_BEST" ]; then
    log "ERROR: $V7_BEST không tồn tại. V7 training fail."
    exit 1
fi

P3_END=$(date +%s)
P3_MIN=$(( (P3_END - P3_START) / 60 ))
log ""
log "Phase 4c done in ${P3_MIN} phút"

$PY -c "
import pandas as pd
df = pd.read_csv('runs/detect/train_v8s_v7_1280/results.csv')
df.columns = [c.strip() for c in df.columns]
best = df.loc[df['metrics/mAP50(B)'].idxmax()]
print(f'  V7 best epoch {int(best[\"epoch\"])}: '
      f'mAP50={best[\"metrics/mAP50(B)\"]:.4f}, '
      f'mAP50-95={best[\"metrics/mAP50-95(B)\"]:.4f}')
" 2>/dev/null || log "(không đọc được metrics V7)"

# ========== Phase 4d: Full cross-eval (shared script, V7 included) ==========
P4_START=$(date +%s)
log ""
log "===== Phase 4d: Full cross-eval all models (V7 included) ====="
log "Expected: ~20-30 phút. Log: logs/phase4_04_eval.log"
log "Shared script: $EVAL"
log ""

$PY $EVAL 2>&1 | tee logs/phase4_04_eval.log

P4_END=$(date +%s)
P4_MIN=$(( (P4_END - P4_START) / 60 ))
log "Phase 4d done in ${P4_MIN} phút"

# ========== Summary ==========
TOTAL_SEC=$(( P4_END - START_TS ))
TOTAL_MIN=$(( TOTAL_SEC / 60 ))
TOTAL_H=$(( TOTAL_MIN / 60 ))

log ""
log "===== PHASE 4 (V7) PIPELINE DONE ====="
log "Tổng thời gian: ${TOTAL_H}h $(( TOTAL_MIN % 60 ))m"
log ""
log "Breakdown:"
log "  4a   (import DETRAC): $(( (P1_END - P1_START) / 60 )) phút"
log "  4b   (build V7):      $(( P2_END - P2_START )) giây"
log "  4b.5 (leak audit):    $(( P25_END - P25_START )) giây"
log "  4c   (train V7):      ${P3_MIN} phút"
log "  4d   (full eval):     ${P4_MIN} phút"
log ""
log "Artifacts:"
log "  V7 weights:   runs/detect/train_v8s_v7_1280/weights/best.pt"
log "  DETRAC subset: data/detrac_bustruck_subset/ (có manifest.json)"
log "  V7 dataset:   data/merged_v7/"
log "  Final eval:   results/v5_cross_eval/FINAL_COMPARISON.md (updated V7)"
log ""
log "Mở FINAL_COMPARISON.md để xem V7 vs V5+ vs V6 vs baseline."
