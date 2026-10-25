#!/bin/bash
# Train YOLOv8s cho VN Traffic — tự auto-resume nếu đã có checkpoint.
#
# Chạy:
# ./scripts/train.sh # auto: resume nếu có last.pt, ngược lại train mới
# ./scripts/train.sh fresh # ép train từ đầu (ghi đè run cũ)
# ./scripts/train.sh resume # ép resume, lỗi nếu không có last.pt
#
# Sau khi in PID, có thể chạy kèm watcher eval:
# nohup ./scripts/eval_after_training.sh <PID> > logs/eval_watcher.log 2>&1 &
# tail -f logs/ft_vnv3.log

set -e
cd "$(dirname "$0")/.."

NAME="train_v8s_4cls"
LOG="logs/${NAME}.log"
MODE="${1:-auto}"

# 2 vị trí Ultralytics có thể save (do settings.runs_dir prepend)
CKPT1="runs/detect/${NAME}/weights/last.pt"
CKPT2="runs/runs/detect/${NAME}/weights/last.pt"
CKPT3="runs/detect/runs/detect/${NAME}/weights/last.pt"

HAS_CKPT="no"
for c in "$CKPT1" "$CKPT2" "$CKPT3"; do
    if [ -f "$c" ]; then HAS_CKPT="yes"; break; fi
done

# Quyết định mode
case "$MODE" in
    fresh)
        DO_RESUME="no"
        ;;
    resume)
        if [ "$HAS_CKPT" = "no" ]; then
            echo "[] Không có checkpoint để resume. Xoá 'resume' hoặc chạy 'fresh'."
            exit 1
        fi
        DO_RESUME="yes"
        ;;
    auto)
        DO_RESUME="$HAS_CKPT"
        ;;
    *)
        echo "Usage: $0 [auto|fresh|resume]"
        exit 1
        ;;
esac

mkdir -p logs

# Load conda
source /home/xuand/miniconda3/etc/profile.d/conda.sh
conda activate yolov8_ft

if [ "$DO_RESUME" = "yes" ]; then
    echo "[↻] Resume training run: $NAME"
    nohup python -m src.detection.train \
        --resume --name "$NAME" \
        >> "$LOG" 2>&1 &
else
    echo "[▶] Fresh training run: $NAME"
    nohup python -m src.detection.train \
        --data data/data.yaml \
        --pretrained yolov8s.pt \
        --epochs 50 \
        --batch 32 \
        --patience 15 \
        --imgsz 512 \
        --workers 12 \
        --cache ram \
        --device 0 \
        --name "$NAME" \
        > "$LOG" 2>&1 &
fi

PID=$!
echo "PID: $PID"
echo "Log: $LOG"
echo ""
echo "Kill: kill $PID"
echo "Auto-eval: nohup ./scripts/eval_after_training.sh $PID > logs/eval_watcher.log 2>&1 &"
echo ""
echo "─── Live epoch progress (Ctrl+C để thoát view, training vẫn chạy nền) ───"
echo ""

# Chờ log có nội dung
while [ ! -s "$LOG" ]; do sleep 0.5; done

# Tail raw — để terminal tự render \r (progress bar update tại chỗ trong khi
# các dòng thường như epoch header, val summary "all", "Results saved"... hiện trên dòng mới.
tail -n +1 -f --pid=$PID "$LOG"
