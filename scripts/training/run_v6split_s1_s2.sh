#!/usr/bin/env bash
# Resume/finish S1, then run S2. Safe to run again after an interruption.

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_SCRIPT="$PROJECT_ROOT/scripts/training/v6final_train.py"
TRAIN_PYTHON="${TRAIN_PYTHON:-/home/xuand/miniconda3/envs/yolov8_ft/bin/python}"
TARGET_EPOCHS="${TARGET_EPOCHS:-30}"
LOG_DIR="$PROJECT_ROOT/logs"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOG_DIR/v6split_s1_s2_${RUN_STAMP}.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

# Avoid starting the same two-stage pipeline twice by accident.
exec 9>"$LOG_DIR/v6split_s1_s2.lock"
if ! flock -n 9; then
    echo "ERROR: pipeline S1 -> S2 dang chay trong mot process khac." >&2
    exit 1
fi

exec > >(tee -a "$MASTER_LOG") 2>&1

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

completed_epochs() {
    local results_file="$1"
    if [[ ! -s "$results_file" ]]; then
        printf '0\n'
        return
    fi
    awk -F, 'NR > 1 && NF > 1 { count++ } END { print count + 0 }' "$results_file"
}

run_stage() {
    local key="$1"
    local run_name="$2"
    local run_dir="$PROJECT_ROOT/runs/detect/$run_name"
    local results_file="$run_dir/results.csv"
    local last_weights="$run_dir/weights/last.pt"
    local best_weights="$run_dir/weights/best.pt"
    local done_epochs

    done_epochs="$(completed_epochs "$results_file")"

    if (( done_epochs >= TARGET_EPOCHS )) && [[ -s "$last_weights" && -s "$best_weights" ]]; then
        log "$key ($run_name) da xong: $done_epochs/$TARGET_EPOCHS epoch -> bo qua."
        return
    fi

    if [[ -s "$last_weights" ]]; then
        log "$key ($run_name): resume tu last.pt ($done_epochs/$TARGET_EPOCHS epoch da xong)."
    else
        log "$key ($run_name): chua co last.pt, bat dau lai tu dau epoch hien tai."
    fi

    "$TRAIN_PYTHON" "$TRAIN_SCRIPT" --run "$key"

    done_epochs="$(completed_epochs "$results_file")"
    if (( done_epochs < TARGET_EPOCHS )) || [[ ! -s "$last_weights" || ! -s "$best_weights" ]]; then
        log "ERROR: $key dung khi moi co $done_epochs/$TARGET_EPOCHS epoch; khong chay stage tiep theo."
        return 1
    fi

    log "$key hoan tat: $done_epochs/$TARGET_EPOCHS epoch."
}

if [[ ! -x "$TRAIN_PYTHON" ]]; then
    log "ERROR: khong tim thay Python executable: $TRAIN_PYTHON"
    log "Co the doi bang: TRAIN_PYTHON=/duong/dan/python bash $0"
    exit 1
fi

if [[ ! -f "$TRAIN_SCRIPT" || ! -f "$PROJECT_ROOT/data/data_v6split.yaml" ]]; then
    log "ERROR: thieu train script hoac data/data_v6split.yaml."
    exit 1
fi

log "Bat dau pipeline S1 -> S2"
log "Python: $TRAIN_PYTHON"
log "Log: $MASTER_LOG"

run_stage s1 v6split_f1_freeze10
run_stage s2 v6split_f2_nofreeze

log "Da hoan tat ca S1 va S2."
