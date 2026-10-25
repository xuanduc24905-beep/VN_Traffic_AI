#!/bin/bash
# Chờ training PID xong rồi tự động chạy eval so sánh models.
#
# Chạy:
# ./scripts/eval_after_training.sh <TRAINING_PID>
# ./scripts/eval_after_training.sh 51013
#
# Log ra: logs/eval_after_training.log

set -e
cd "$(dirname "$0")/.."

PID="${1:?Truyền TRAINING_PID (vd: 51013)}"
LOG="logs/eval_after_training.log"

echo "[$(date '+%H:%M:%S')] Chờ training PID=$PID xong..." | tee "$LOG"

while kill -0 "$PID" 2>/dev/null; do
    sleep 60
done

echo "[$(date '+%H:%M:%S')] Training xong. Bắt đầu eval." | tee -a "$LOG"

source /home/xuand/miniconda3/etc/profile.d/conda.sh
conda activate yolov8_ft

python -m src.evaluation.eval_full 2>&1 | tee -a "$LOG"

echo "[$(date '+%H:%M:%S')] Done. Xem results/tables/eval_full.json" | tee -a "$LOG"
