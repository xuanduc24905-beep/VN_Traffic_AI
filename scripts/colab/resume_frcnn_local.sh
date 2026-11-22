#!/usr/bin/env bash
# Train tiep Faster R-CNN tren may local tu last.pt da train tren Colab.
# 1) Tai ca thu muc MyDrive/vn_traffic_frcnn/frcnn_v5plus/ ve weights/trained/frcnn_v5plus/
#    (can it nhat: last.pt, best.pt, train_log.csv, env.json, config.json)
# 2) bash scripts/colab/resume_frcnn_local.sh      (log: logs/frcnn_v5plus_resume.log)
set -euo pipefail
ROOT=/home/xuand/vn_traffic_ai
cd "$ROOT"
export FRCNN_LOCAL=1 FRCNN_LOCAL_DATA="$ROOT/data/merged_v5plus" FRCNN_LOCAL_OUT="$ROOT/weights/trained/frcnn_v5plus"
cd /tmp  # tranh import nham module tu thu muc du lieu
~/miniconda3/envs/vehicle_env/bin/python "$ROOT/scripts/colab/run_notebook_cells.py" "$ROOT/notebooks/colab_frcnn_v5plus.ipynb"
