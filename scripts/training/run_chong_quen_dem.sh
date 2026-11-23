#!/usr/bin/env bash
# Chuỗi chạy qua đêm: Fisher -> LR thấp -> L2-SP (mid, lo, hi) -> EWC; mỗi run xong thì chấm điểm ngay.
# Chạy lại được sau khi bị ngắt: run nào đủ 30 epoch thì bỏ qua, run dở thì resume từ last.pt.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${TRAIN_PYTHON:-/home/xuand/miniconda3/envs/yolov8_ft/bin/python}"
cd "$ROOT"
mkdir -p logs
exec 9>logs/chong_quen_dem.lock
flock -n 9 || { echo "Chuoi chong quen dang chay o process khac"; exit 1; }
LOG="logs/chong_quen_dem_$(date +%Y%m%d_%H%M%S).log"
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

done_epochs() { [[ -s "$1" ]] && awk -F, 'NR>1&&NF>1{c++} END{print c+0}' "$1" || echo 0; }

if [[ ! -s weights/chong_quen_fisher.pt ]]; then
    log "Tinh Fisher cho EWC"
    "$PY" scripts/training/chong_quen_fisher.py >> "$LOG" 2>&1 || log "LOI: Fisher that bai (EWC se bi bo qua)"
fi

# key:ten_run:ten_model_trong_eval
for item in lowlr:cq_lowlr_s2:lowlr_S2 l2sp_mid:cq_l2sp_mid:l2sp_lam_mid l2sp_lo:cq_l2sp_lo:l2sp_lam_lo \
            l2sp_hi:cq_l2sp_hi:l2sp_lam_hi ewc:cq_ewc:ewc; do
    IFS=: read -r key run model <<< "$item"
    if [[ "$key" == ewc && ! -s weights/chong_quen_fisher.pt ]]; then log "Bo qua EWC: chua co Fisher"; continue; fi
    n=$(done_epochs "runs/detect/$run/results.csv")
    if (( n < 30 )); then
        log "Train $key ($run), da co $n/30 epoch"
        "$PY" scripts/training/chong_quen_train.py --run "$key" > "outputs/cq_${key}.log" 2>&1
        n=$(done_epochs "runs/detect/$run/results.csv")
        (( n < 30 )) && { log "LOI: $key dung o $n/30 epoch, sang run tiep theo"; continue; }
    fi
    log "Cham diem $model"
    "$PY" scripts/evaluation/eval_chong_quen.py --models "$model" >> "outputs/cq_eval_${key}.log" 2>&1
    log "Xong $key"
done
log "Hoan tat chuoi chong quen"
