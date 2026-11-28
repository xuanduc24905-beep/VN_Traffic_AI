#!/usr/bin/env bash
# Chuỗi train + chấm điểm cho project cuối kì (resume được sau khi mất điện).
#   bash scripts/training/run_cuoi_ki.sh <buoc>
# Bước:
#   seeds   : T2 với seed 0 và 1 (2 run x 30 epoch) rồi chấm điểm
#   pseudo  : T2 + nhãn giả yolov8x (khoá p6) rồi chấm điểm
#   <key>   : run bất kì trong chong_quen_train.py; model eval cùng tên key
# Run nào đủ số epoch thì bỏ qua, run dở thì chong_quen_train.py tự resume từ last.pt.
# Toàn bộ thân script nằm trong hàm main → sửa file khi đang chạy không làm hỏng tiến trình đang chạy.
main() {
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${TRAIN_PYTHON:-/home/xuand/miniconda3/envs/yolov8_ft/bin/python}"
cd "$ROOT"
mkdir -p logs outputs
exec 9>logs/cuoi_ki.lock
flock -n 9 || { echo "Chuoi cuoi ki dang chay o process khac"; exit 1; }
LOG="logs/cuoi_ki_$(date +%Y%m%d_%H%M%S).log"
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
done_epochs() { [[ -s "$1" ]] && awk -F, 'NR>1&&NF>1{c++} END{print c+0}' "$1" || echo 0; }

# train_one <key> <seed> <ten_run> <so_epoch>
train_one() {
    local key=$1 seed=$2 run=$3 ep=$4 n tries=0
    n=$(done_epochs "runs/detect/$run/results.csv")
    while (( n < ep && tries < 3 )); do
        log "Train $key seed $seed ($run), da co $n/$ep epoch"
        "$PY" scripts/training/chong_quen_train.py --run "$key" --seed "$seed" >> "outputs/ck_${run}.log" 2>&1
        n=$(done_epochs "runs/detect/$run/results.csv"); tries=$((tries + 1))
    done
    (( n < ep )) && { log "LOI: $run dung o $n/$ep epoch"; return 1; }
    log "Xong train $run ($n epoch)"
}
evalm() { log "Cham diem $*"; "$PY" scripts/evaluation/eval_chong_quen.py --models "$@" >> "outputs/ck_eval.log" 2>&1; log "Xong cham diem $*"; }

case "${1:-}" in
    seeds)
        train_one t2_truckx2 0 cq_l2sp_coco5k_truckx2_s0 30
        train_one t2_truckx2 1 cq_l2sp_coco5k_truckx2_s1 30
        evalm t2_truckx2_s0 t2_truckx2_s1 ;;
    *)
        [[ -n "${1:-}" ]] || { echo "Can ten buoc"; exit 2; }
        key=$1; seed=${2:-42}
        run=$("$PY" -c "import sys; sys.path.insert(0,'scripts/training'); import chong_quen_train as c; \
cfg=c.RUNS_CFG['$key']; print(cfg['name'] if $seed==c.SEED else cfg['name']+'_s$seed', cfg.get('epochs', c.EPOCHS))")
        read -r rname rep <<< "$run"
        train_one "$key" "$seed" "$rname" "$rep" && evalm "${3:-$key}" ;;
esac
log "Hoan tat buoc ${1:-}"
}
main "$@"; exit
