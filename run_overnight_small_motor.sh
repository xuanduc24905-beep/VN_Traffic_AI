#!/usr/bin/env bash
# ==============================================================================
# Script cắm đêm: Huấn luyện P2 Head & Huấn luyện theo vùng (Tiles) cho xe máy nhỏ
# Tự động chạy tuần tự, ghi log, hỗ trợ resume khi đứt gánh.
# ==============================================================================

set -uo pipefail

PROJECT_ROOT="/home/xuand/vn_traffic_ai"
cd "$PROJECT_ROOT"

# Đường dẫn Python của môi trường conda yolov8_ft
PY_BIN="/home/xuand/miniconda3/envs/yolov8_ft/bin/python"

# Nếu PY_BIN không tìm thấy, fallback sang python hiện tại
if [ ! -x "$PY_BIN" ]; then
    echo "[WARN] Không tìm thấy $PY_BIN, thử dùng python từ PATH..."
    PY_BIN=$(which python3 || which python)
fi

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="logs/overnight_motor_${TS}.log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | tee -a "$MAIN_LOG"
}

banner() {
    log "========================================================================"
    log "  $*"
    log "========================================================================"
}

# ----------------- 0. KIỂM TRA PRE-FLIGHT -----------------
banner "BẮT ĐẦU PIPELINE QUA ĐÊM: SMALL MOTORCYCLE (P2 + TILES)"
log "Project Root : $PROJECT_ROOT"
log "Python       : $PY_BIN"
log "Log tổng     : $MAIN_LOG"

if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader | head -1)
    GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | head -1)
    log "GPU          : $GPU_NAME (${GPU_MEM} free, ${GPU_TEMP}°C)"
else
    log "[WARN] Không tìm thấy nvidia-smi, chạy trên CPU có thể rất chậm!"
fi

# ----------------- 1. THÍ NGHIỆM 1: P2 HEAD (imgsz 960) -----------------
banner "PHASE 1/3: Train YOLOv8s-P2 Head (imgsz 960, 40 epochs)"
P2_BEST="runs/detect/motor_p2_head/weights/best.pt"
P2_LAST="runs/detect/motor_p2_head/weights/last.pt"

if [ -f "$P2_BEST" ]; then
    log "[SKIP] motor_p2_head đã có $P2_BEST. Chuyển sang phase tiếp theo."
else
    if [ -f "$P2_LAST" ]; then
        log "[RESUME] Tìm thấy $P2_LAST, tiếp tục train motor_p2_head..."
        $PY_BIN -u scripts/small_motor/train_p2_motor.py --exp p2_head --resume --device 0 2>&1 | tee -a "$MAIN_LOG"
    else
        log "[START] Bắt đầu train mới motor_p2_head..."
        $PY_BIN -u scripts/small_motor/train_p2_motor.py --exp p2_head --device 0 2>&1 | tee -a "$MAIN_LOG"
    fi
fi

# ----------------- 2. THÍ NGHIỆM 2: TILE TRAINING MIXED -----------------
banner "PHASE 2/3: Huấn luyện theo vùng (Tiles 640 + Ảnh gốc, YOLOv8s chuẩn)"
TILE_BEST="runs/detect/motor_tile_mixed/weights/best.pt"

if [ -f "$TILE_BEST" ]; then
    log "[SKIP] motor_tile_mixed đã có $TILE_BEST. Chuyển sang phase tiếp theo."
else
    log "[START] Cắt ảnh và train motor_tile_mixed..."
    $PY_BIN -u scripts/small_motor/tile_motor_train.py --mode mixed --device 0 2>&1 | tee -a "$MAIN_LOG"
fi

# ----------------- 3. THÍ NGHIỆM 3: COMBO P2 HEAD + TILES MIXED -----------------
banner "PHASE 3/3: Combo P2 Head + Tile Mixed (Cấu hình tối ưu nhất)"
COMBO_BEST="runs/detect/motor_tile_mixed_p2/weights/best.pt"

if [ -f "$COMBO_BEST" ]; then
    log "[SKIP] motor_tile_mixed_p2 đã có $COMBO_BEST."
else
    log "[START] Train combo P2 Head + Tile Mixed..."
    $PY_BIN -u scripts/small_motor/tile_motor_train.py --mode mixed --p2 --device 0 2>&1 | tee -a "$MAIN_LOG"
fi

# ----------------- 4. TỔNG HỢP SO SÁNH CUỐI CÙNG -----------------
banner "HOÀN TẤT: SO SÁNH VỚI BASELINE"
log "Chạy đánh giá tổng hợp trên val set..."
$PY_BIN -u scripts/small_motor/eval_compare.py --device 0 2>&1 | tee -a "$MAIN_LOG"

log "Xong toàn bộ! Xem chi tiết tại:"
log "  - Bảng CSV : runs/detect/motor_p2_summary.csv và runs/detect/motor_tile_summary.csv"
log "  - Biểu đồ  : analysis/model_compare/compare_val.png"
log "  - Log đầy đủ : $MAIN_LOG"
