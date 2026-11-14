# Pipeline: Phát hiện Motorcycle Nhỏ (P2 Head + Tile Training)

Mục tiêu: cải thiện AP50 của class **motorcycle** trên camera CCTV Việt Nam bằng
2 kỹ thuật có kiểm soát: **P2 feature head** và **huấn luyện theo vùng ảnh (tile)**.

## Các script

| Script | Vai trò |
|---|---|
| `00_audit_size.py` | Thống kê phân phối kích thước bbox motorcycle → biết bao nhiêu % là "small" |
| `train_p2_motor.py` | **P2 head**: so sánh 4 cấu hình có kiểm soát |
| `tile_motor_train.py` | **Tile training**: cắt ảnh theo vùng ưu tiên motor nhỏ + train |
| `eval_compare.py` | So sánh tất cả model đã train, xuất CSV + biểu đồ |

---

## Thứ tự chạy

### Bước 0 — Kiểm tra dữ liệu
```bash
python3 scripts/small_motor/00_audit_size.py \
    --data data/data_v5plus.yaml \
    --model runs/detect/motor_baseline/weights/best.pt   # tuỳ chọn
```
→ `analysis/size_audit/size_histogram.png` + `size_stats.json`

---

### Bước 1 — Train baseline (bắt buộc, làm đối chứng)
```bash
python3 scripts/small_motor/train_p2_motor.py \
    --exp baseline --data data/data_v5plus.yaml --device 0
```

### Bước 2a — P2 Head (chỉ đổi kiến trúc)
```bash
python3 scripts/small_motor/train_p2_motor.py \
    --exp p2_head --data data/data_v5plus.yaml --device 0
```

### Bước 2b — P2 + Augmentation nhỏ
```bash
python3 scripts/small_motor/train_p2_motor.py \
    --exp p2_aug --data data/data_v5plus.yaml --device 0
```

### Bước 2c — P2 + imgsz 1280
```bash
python3 scripts/small_motor/train_p2_motor.py \
    --exp p2_1280 --data data/data_v5plus.yaml --device 0
```

**Hoặc chạy tuần tự cả 4:**
```bash
python3 scripts/small_motor/train_p2_motor.py \
    --all --data data/data_v5plus.yaml --device 0
```

---

### Bước 3a — Tile training (3 mode so sánh)
```bash
# Cắt ảnh + train, so sánh full / tile / mixed
python3 scripts/small_motor/tile_motor_train.py \
    --compare --src data/merged_v5plus --device 0
```

**Hoặc chạy từng mode:**
```bash
# Mode: mixed (ảnh đầy đủ + ô — thường tốt nhất)
python3 scripts/small_motor/tile_motor_train.py \
    --mode mixed --src data/merged_v5plus \
    --out data/merged_v5plus_tiles_mixed --device 0

# Kết hợp P2 head + tile (cấu hình tốt nhất dự kiến)
python3 scripts/small_motor/tile_motor_train.py \
    --mode mixed --p2 --src data/merged_v5plus --device 0
```

---

### Bước 4 — So sánh tất cả model
```bash
# Val (dùng để chọn model)
python3 scripts/small_motor/eval_compare.py \
    --data data/data_v5plus.yaml --device 0

# Test (chỉ 1 lần sau khi đã chốt)
python3 scripts/small_motor/eval_compare.py \
    --data data/data_v5plus.yaml --test --device 0
```
→ `analysis/model_compare/compare_val.csv` + `compare_val.png`

---

## Thí nghiệm kiểm soát (P2 Head)

| Run | Arch | imgsz | Aug | Mục đích |
|---|---|---|---|---|
| `motor_baseline` | yolov8s | 960 | chuẩn | Đối chứng |
| `motor_p2_head` | yolov8s-p2 | 960 | chuẩn | Chỉ đổi kiến trúc |
| `motor_p2_aug` | yolov8s-p2 | 960 | aug_small | Kiến trúc + aug |
| `motor_p2_1280` | yolov8s-p2 | 1280 | aug_small | Kiến trúc + phân giải cao |

## Thí nghiệm kiểm soát (Tile Training)

| Run | Mode | Dữ liệu train | Mục đích |
|---|---|---|---|
| `motor_tile_full` | full | Ảnh đầy đủ gốc | Đối chứng tile |
| `motor_tile_tile` | tile | Chỉ ô cắt | Extreme tile |
| `motor_tile_mixed` | mixed | Ảnh đầy đủ + ô | Thực tế nhất |
| `motor_tile_mixed_p2` | mixed + P2 | Ảnh đầy đủ + ô | Kết hợp tốt nhất |

---

## Tham số quan trọng (tile_motor_train.py)

| Tham số | Mặc định | Giải thích |
|---|---|---|
| `--tile` | 640 | Kích thước ô (px) |
| `--overlap` | 0.25 | Chồng lấn 25% → bắt xe ở rìa |
| `--min-visible` | 0.5 | Bbox phải hiện ≥50% trong ô mới giữ |
| `--keep-no-motor` | 0.2 | 20% ô không có motor được giữ (tránh background bias) |

---

## Chạy thử nhanh (3 epoch, 10% data)
```bash
python3 scripts/small_motor/train_p2_motor.py \
    --exp p2_head --data data/data_v5plus.yaml \
    --epochs 3 --fraction 0.1 --device cpu

python3 scripts/small_motor/tile_motor_train.py \
    --mode mixed --src data/merged_v5plus \
    --epochs 3 --fraction 0.1 --device cpu
```
