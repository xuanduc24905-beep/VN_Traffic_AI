#!/usr/bin/env bash
# One-shot setup dataset cho VN Traffic AI.
# - Import UA-DETRAC từ project cũ (symlink JPG + remap label).
# - Import Roboflow VN nếu đã có ở data/raw/roboflow_vn/.
# - Import VisDrone nếu đã có ở data/raw/visdrone/.
# - In stats cuối.
#
# Chạy: bash scripts/data_prep/setup_datasets.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "======================================================"
echo " VN Traffic AI — Dataset Setup"
echo "======================================================"
echo ""

# ---------- 1. UA-DETRAC ----------
echo "[1/3] Import UA-DETRAC từ /home/xuand/yolov8_finetune ..."
if [[ ! -d /home/xuand/yolov8_finetune/data/detrac/labels ]]; then
    echo " [!] Không tìm thấy project cũ. Bỏ qua UA-DETRAC."
else
    python scripts/data_prep/import_ua_detrac.py --out data/merged
fi
echo ""

# ---------- 2. Roboflow VN ----------
echo "[2/3] Import Roboflow Vietnamese Traffic Vehicles ..."
if [[ ! -d data/raw/roboflow_vn ]]; then
    cat <<EOF
  [!] Chưa có data/raw/roboflow_vn/. Bỏ qua.

  Để có dataset này:
    1. Vào https://universe.roboflow.com/ tìm 'vietnam vehicle'
    2. Download YOLOv8 format, giải nén vào: data/raw/roboflow_vn/
    3. Cấu trúc cần: data/raw/roboflow_vn/{images,labels}/
    4. Kiểm tra thứ tự class trong dataset gốc, chỉnh
       configs/class_mapping.json > roboflow_vn nếu cần
    5. Chạy lại: bash scripts/data_prep/setup_datasets.sh

EOF
else
    python scripts/data_prep/merge_datasets.py --sources roboflow_vn --out data/merged
fi
echo ""

# ---------- 3. VisDrone ----------
echo "[3/3] Import VisDrone ..."
if [[ ! -d data/raw/visdrone ]]; then
    cat <<EOF
  [!] Chưa có data/raw/visdrone/. Bỏ qua.

  Để có dataset này (không bắt buộc cho detection, hữu ích cho tracking test):
    1. https://github.com/VisDrone/VisDrone-Dataset
    2. Tải VisDrone2019-VID-train + val, giải nén vào: data/raw/visdrone/
    3. Chạy lại: bash scripts/data_prep/setup_datasets.sh

EOF
else
    python scripts/data_prep/merge_datasets.py --sources visdrone --out data/merged
fi
echo ""

# ---------- Stats cuối ----------
echo "======================================================"
echo " DATASET STATS"
echo "======================================================"
for split in train val test; do
    n_img=$(find data/merged/images/$split -type l -o -type f 2>/dev/null | wc -l || echo 0)
    n_lbl=$(find data/merged/labels/$split -type f 2>/dev/null | wc -l || echo 0)
    printf " %-6s images=%7d labels=%7d\n" "$split" "$n_img" "$n_lbl"
done
echo ""

n_total=$(find data/merged/images -type l -o -type f 2>/dev/null | wc -l)
if [[ "$n_total" -eq 0 ]]; then
    echo "[X] Không có ảnh nào được import. Kiểm tra path project cũ."
    exit 1
fi

echo "[] Dataset đã sẵn ở data/merged/"
echo "[] File config: data/data.yaml"
echo ""
echo "Bước tiếp theo:"
echo " conda activate yolov8_ft"
echo " python -m src.detection.train --data data/data.yaml --epochs 50 --batch 16"
