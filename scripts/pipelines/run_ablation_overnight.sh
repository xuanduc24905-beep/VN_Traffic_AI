#!/usr/bin/env bash
# Chạy tuần tự 3 recipe ablation trên nền V5+ (data_v5plus.yaml).
#
#   r_p2          -> yolov8s-p2 (thêm đầu stride 4, tốt cho vật nhỏ)
#   r_clspw       -> cls_pw=0.5 (tăng trọng số lớp hiếm bus/truck)
#   r_progressive -> 2 giai đoạn 960 -> 1280 (curriculum)
#
# So sánh với nền V5+ đã có: train_v8s_v5plus_960 (test mAP50 ~0.782).
# Kết quả tự ghi vào runs/detect/pipelines_summary.csv (do train.py append).
#
# Dùng:   bash scripts/pipelines/run_ablation_overnight.sh
# Resume: chạy lại -> mỗi recipe check best.pt, nếu có thì SKIP;
#         nếu có last.pt mà chưa xong -> truyền --resume tự động.

set -euo pipefail

PROJECT_ROOT="/home/xuand/vn_traffic_ai"
cd "$PROJECT_ROOT"

DATA_YAML="data/data_v5plus.yaml"
DEVICE="${DEVICE:-0}"
RUNS_DIR="runs/detect"

# ─────────── Logging ───────────
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/ablation_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

log()     { echo "[$(date +'%H:%M:%S')] $*"; }
section() { echo ""; echo "═══════════════════════════════════════════════════"; echo "  $*"; echo "═══════════════════════════════════════════════════"; }

START_TIME=$(date +%s)
log "Ablation V5+ bắt đầu. Log: $LOG_FILE"
log "Device: $DEVICE | Data: $DATA_YAML"

# ─────────── Preflight ───────────
section "[0/4] Preflight"

if [ ! -f "$DATA_YAML" ]; then
    echo "ERROR: thiếu $DATA_YAML" >&2; exit 1
fi
if [ ! -f "yolov8s.pt" ]; then
    echo "ERROR: thiếu yolov8s.pt (trọng số COCO)" >&2; exit 1
fi

log "Python: $(python3 --version)"
log "Ultralytics: $(python3 -c 'import ultralytics; print(ultralytics.__version__)' 2>/dev/null || echo 'MISSING')"
log "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || log "  (không có GPU, sẽ rất chậm)"

TRAIN_N=$(ls data/merged_v5plus/images/train 2>/dev/null | wc -l)
log "V5+ train ảnh: $TRAIN_N"

V5P_BEST="runs/detect/train_v8s_v5plus_960/weights/best.pt"

# ─────────── Helper: chạy 1 recipe (SKIP nếu xong, RESUME nếu dở) ───────────
# $1 = tên run (vd r_p2), $2 = recipe (vd p2), $3 = epochs, $@4+ = flag thêm (--init, --set ...)
run_recipe() {
    local NAME="$1"
    local RECIPE="$2"
    local EPOCHS="$3"
    shift 3
    local EXTRA=("$@")
    local RUN_DIR="$RUNS_DIR/$NAME"
    local BEST="$RUN_DIR/weights/best.pt"
    local LAST="$RUN_DIR/weights/last.pt"
    local RESULTS="$RUN_DIR/results.csv"

    section "[$NAME] recipe=$RECIPE epochs=$EPOCHS"

    local DONE=0
    if [ -f "$RESULTS" ]; then
        DONE=$(( $(wc -l < "$RESULTS") - 1 ))
        [ "$DONE" -lt 0 ] && DONE=0
    fi
    log "[STATE] done=$DONE/$EPOCHS best=$([ -f "$BEST" ] && echo yes || echo no) last=$([ -f "$LAST" ] && echo yes || echo no)"

    if [ -f "$BEST" ] && [ "$DONE" -ge "$EPOCHS" ]; then
        log "[SKIP] $NAME đã xong. Best: $BEST"
        return 0
    fi

    if [ -f "$LAST" ] && [ "$DONE" -gt 0 ]; then
        log "[RESUME] từ epoch $DONE"
        python3 scripts/pipelines/train.py --data "$DATA_YAML" --name "$NAME" --recipe "$RECIPE" --device "$DEVICE" --resume
    else
        log "[FRESH] bắt đầu từ đầu"
        python3 scripts/pipelines/train.py --data "$DATA_YAML" --name "$NAME" --recipe "$RECIPE" \
            --epochs "$EPOCHS" --device "$DEVICE" "${EXTRA[@]}"
    fi

    if [ ! -f "$BEST" ]; then
        echo "ERROR: $BEST không tồn tại sau khi train" >&2
        exit 1
    fi
    log "[XONG] $NAME"
}

# ─────────── Chạy 3 recipe (cấu hình rút gọn, tổng ~3.5-4h) ───────────
#
#   r_p2         : kiến trúc khác (thêm P2 head stride 4) -> KHÔNG finetune được từ V5+,
#                  phải train từ COCO. 25 epoch + patience=8 để sớm dừng khi hội tụ.
#   r_clspw     : cùng kiến trúc V5+ -> FINETUNE từ V5+ best.pt, 15 epoch, LR thấp.
#                  Đây là pattern giống p4_scale/p4_imgsz1280 đã làm (val mAP ~0.855).
#   r_progressive: 2 giai đoạn 960->1280 từ COCO. Rút: stage1 20 epoch + stage2 10 epoch.
#
# Muốn FULL 40 epoch như gốc -> set EPOCHS_P2=40 EPOCHS_CLS=40 EPOCHS_PROG=40 ...

EPOCHS_P2="${EPOCHS_P2:-25}"
EPOCHS_CLS="${EPOCHS_CLS:-15}"
EPOCHS_PROG_S1="${EPOCHS_PROG_S1:-20}"
EPOCHS_PROG_S2="${EPOCHS_PROG_S2:-10}"

# p2: từ COCO, patience=8 để early-stop nếu plateau
run_recipe "r_p2" "p2" "$EPOCHS_P2" --set "patience=8"

# clspw: finetune từ V5+ best.pt (nhanh), LR thấp + close_mosaic sớm
if [ -f "$V5P_BEST" ]; then
    run_recipe "r_clspw" "clspw" "$EPOCHS_CLS" \
        --init "$V5P_BEST" \
        --set "lr0=0.001" "lrf=0.1" "close_mosaic=5" "patience=8"
else
    log "[WARN] không có $V5P_BEST, chạy from scratch 25 epoch cho clspw"
    run_recipe "r_clspw" "clspw" 25 --set "patience=8"
fi

# progressive: 2 stage từ COCO. train.py tự xử lý stage 2 dùng --epochs2
python3 scripts/pipelines/train.py --data "$DATA_YAML" --name "r_progressive" --recipe "progressive" \
    --epochs "$EPOCHS_PROG_S1" --epochs2 "$EPOCHS_PROG_S2" --device "$DEVICE" \
    --set "patience=8" || {
        log "[WARN] r_progressive gặp lỗi, kiểm tra log"
    }

# ─────────── Summary ───────────
section "[4/4] Tổng kết"

SUMMARY_CSV="$RUNS_DIR/pipelines_summary.csv"
if [ -f "$SUMMARY_CSV" ]; then
    log "Bảng kết quả (val) — $SUMMARY_CSV:"
    column -s, -t < "$SUMMARY_CSV" || cat "$SUMMARY_CSV"
else
    log "Không thấy $SUMMARY_CSV (train.py chưa ghi)"
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINS=$(((ELAPSED % 3600) / 60))

log "Tổng thời gian: ${HOURS}h ${MINS}m"
log "Log: $LOG_FILE"

echo ""
echo "Sáng dậy mở:"
echo "  1) $LOG_FILE                     (toàn bộ log)"
echo "  2) $SUMMARY_CSV  (bảng val mAP mọi recipe)"
echo "  3) runs/detect/r_p2|r_clspw|r_progressive/  (weights + results per run)"
echo ""
echo "Bước tiếp: viết script eval test set cho từng recipe để có số TEST (không chỉ val)."
