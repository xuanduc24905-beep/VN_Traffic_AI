# Hướng dẫn chuẩn bị dữ liệu

Ba nguồn dataset + 1 phần video tự quay.

## 1. UA-DETRAC — benchmark quốc tế (car/bus/truck)

Nếu đã tải sẵn ở project cũ [/home/xuand/yolov8_finetune/data/processed](../../yolov8_finetune/data/processed):

```bash
mkdir -p data/raw/ua_detrac
ln -s /home/xuand/yolov8_finetune/data/processed/images data/raw/ua_detrac/images
ln -s /home/xuand/yolov8_finetune/data/processed/labels data/raw/ua_detrac/labels
```

**Class mapping** (cấu hình trong `configs/class_mapping.json`):
- 0 (DETRAC) car
- 1 bus
- 2 truck
- 3 (others) null (bỏ)

## 2. Roboflow — Vietnamese Traffic Vehicles

**Cách 1 — thủ công:**
1. Vào https://universe.roboflow.com/
2. Search: "vietnam vehicle" hoặc "vietnam traffic"
3. Chọn dataset có motorcycle + bicycle (nên chọn 1000+ ảnh)
4. Download YOLOv8 format
5. Giải nén vào `data/raw/roboflow_vn/`

Cấu trúc mong đợi:
```
data/raw/roboflow_vn/
├── images/{train,valid,test}/
├── labels/{train,valid,test}/
└── data.yaml
```

**Kiểm tra thứ tự class** trong `data.yaml` của dataset, chỉnh
`configs/class_mapping.json > roboflow_vn` cho khớp.

## 3. VisDrone-VID — video có tracking ID

Dùng cho Người 2 để test tracker (ByteTrack cần input có GT ID).

```bash
mkdir -p data/raw/visdrone
cd data/raw/visdrone
# Download VisDrone2019-VID-train.zip từ https://github.com/VisDrone/VisDrone-Dataset
# Giải nén ra thư mục hiện tại
```

**Class mapping** (VisDrone gốc 10 class):
- 0 (ignored), 1 (pedestrian) null
- 2 bicycle, 3 car, 4 vancar, 5 truck
- 6 (tricycle), 7 (awning-tricycle) null
- 8 bus, 9 motorcycle

## 4. Video test VN (Người 4 tự quay)

- Điện thoại đặt cố định tại ngã tư/đường đông trong 5-10 phút mỗi video.
- Ưu tiên góc chéo cao (nhìn được cả 2 chiều).
- Tối thiểu 2-3 video ở khung giờ và địa điểm khác nhau.
- Lưu vào `data/test_videos/raw/`.

Rồi:
```bash
python scripts/data_prep/prepare_test_video.py \
    --video data/test_videos/raw/nga_tu_01.mp4 \
    --out data/test_videos/ground_truth/nga_tu_01.json
```

Mở JSON template, điền tay số lượt cho từng class.

## Merge tất cả về data/merged

```bash
python scripts/data_prep/merge_datasets.py
```

Kiểm tra:
```bash
find data/merged/images -type f | wc -l # tổng số ảnh
ls data/merged/images/train | head # sample
cat data/data.yaml # config 5 class đã có sẵn
```
