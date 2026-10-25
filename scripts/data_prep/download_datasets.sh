#!/usr/bin/env bash
# Hướng dẫn download dataset. Chưa auto-download (cần API key/tài khoản).
# Chạy sau khi đã điền các link.
set -e

RAW_DIR="$(dirname "$0")/../data/raw"
mkdir -p "$RAW_DIR"

echo "=========================================="
echo "1. UA-DETRAC (có sẵn ở project cũ nếu đã tải)"
echo "=========================================="
echo " copy hoặc symlink từ /home/xuand/yolov8_finetune/data/processed"
echo " cần map class: 0=car, 1=bus, 2=truck, 3=null(others)"
echo ""

echo "=========================================="
echo "2. Roboflow Vietnamese Traffic Vehicles"
echo "=========================================="
echo " https://universe.roboflow.com/ (search 'vietnam vehicle')"
echo " download YOLOv8 format, giải nén vào: $RAW_DIR/roboflow_vn/"
echo ""

echo "=========================================="
echo "3. VisDrone-VID (video + tracking ID)"
echo "=========================================="
echo " https://github.com/VisDrone/VisDrone-Dataset"
echo " tải VisDrone2019-VID train + val, giải nén vào: $RAW_DIR/visdrone/"
echo ""

echo "Sau khi có đủ 3 dataset, chạy:"
echo " python scripts/data_prep/merge_datasets.py"
