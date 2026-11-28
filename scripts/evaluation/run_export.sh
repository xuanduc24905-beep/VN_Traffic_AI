#!/usr/bin/env bash
# Gọi export_trien_khai.py bằng venv có TensorRT; thêm thư viện CUDA/cuDNN của env yolov8_ft cho onnxruntime-gpu.
#   bash scripts/evaluation/run_export.sh --weights <best.pt> --name <ten>
SP=/home/xuand/miniconda3/envs/yolov8_ft/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$(ls -d $SP/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec "${EXPORT_PYTHON:-$HOME/venvs/yolov8_export/bin/python}" scripts/evaluation/export_trien_khai.py "$@"
