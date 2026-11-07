#!/bin/bash
# Fine-tune V6 từ V5 weights + dataset gold+pseudo
#
# Chạy:
#   ./scripts/pseudo/09_train_v6.sh
#
# Hoặc override param:
#   EPOCHS=20 BATCH=8 ./scripts/pseudo/09_train_v6.sh
#
# Theo dõi trong lúc train:
#   tail -f runs/detect/train_v8s_v6_pseudo_ft/results.csv
#   # Hoặc mở W&B dashboard nếu đã login

set -e

cd /home/xuand/vn_traffic_ai

PY=/home/xuand/miniconda3/envs/yolov8_ft/bin/python3
V5_WEIGHTS=runs/detect/train_v8s_v5_honest/weights/best.pt
DATA_V6=configs/data_v6.yaml

# ─── Params (override bằng env var) ───
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-16}
IMGSZ=${IMGSZ:-640}
LR0=${LR0:-0.001}        # fine-tune LR thấp (1/10 default)
PATIENCE=${PATIENCE:-10} # early stop
NAME=${NAME:-train_v8s_v6_pseudo_ft}

# ─── Verify ───
[ -f "$V5_WEIGHTS" ] || { echo "✗ Không thấy V5 weights: $V5_WEIGHTS"; exit 1; }
[ -f "$DATA_V6" ]    || { echo "✗ Không thấy config: $DATA_V6 — chạy 08_build trước"; exit 1; }

# ─── GPU check ───
echo "=== GPU status ==="
nvidia-smi --query-gpu=memory.free,memory.total --format=csv

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Fine-tune V6 từ V5"
echo "═══════════════════════════════════════════════════"
echo "  Base weights : $V5_WEIGHTS"
echo "  Data config  : $DATA_V6"
echo "  Epochs       : $EPOCHS"
echo "  Batch        : $BATCH"
echo "  Imgsz        : $IMGSZ"
echo "  LR0          : $LR0  (fine-tune LR)"
echo "  Patience     : $PATIENCE"
echo "  Output       : runs/detect/$NAME"
echo "═══════════════════════════════════════════════════"
echo ""

# ─── Launch train ───
$PY -c "
from ultralytics import YOLO
model = YOLO('$V5_WEIGHTS')
model.train(
    data='$DATA_V6',
    epochs=$EPOCHS,
    batch=$BATCH,
    imgsz=$IMGSZ,
    lr0=$LR0,
    patience=$PATIENCE,
    project='runs/detect',
    name='$NAME',
    exist_ok=True,
    workers=4,
    device=0,
)
"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Train xong!"
echo "═══════════════════════════════════════════════════"
echo "  Best weights : runs/detect/$NAME/weights/best.pt"
echo "  Results CSV  : runs/detect/$NAME/results.csv"
echo "  Confusion    : runs/detect/$NAME/confusion_matrix.png"
echo ""
echo "  So sánh với V5:"
echo "    yolo val model=$V5_WEIGHTS data=$DATA_V6 split=test"
echo "    yolo val model=runs/detect/$NAME/weights/best.pt data=$DATA_V6 split=test"
