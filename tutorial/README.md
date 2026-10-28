# Tutorial — Train YOLOv8 từ đầu tới cuối

Đây là **tutorial đi từng bước** để train 1 model YOLOv8 phát hiện phương tiện giao thông VN. 7 script số thứ tự, chạy lần lượt sẽ ra được:
- 1 model `best.pt` đã train
- Report đánh giá mAP/precision/recall
- Video demo có bbox overlay
- Biểu đồ loss + confusion matrix

**Dành cho ai:** biết Python cơ bản (vòng lặp, hàm, import), **chưa biết** ML/DL/YOLO. Mỗi file có comment giải thích khái niệm bên trong code.

---

## Những khái niệm cần biết trước (đọc 5 phút)

### 1. Object detection là gì?
Cho ảnh → model vẽ **bounding box** (hình chữ nhật) quanh mỗi vật thể + nói nó là gì (class).

Ví dụ: ảnh đường phố → 1 box quanh xe máy (nhãn: motorcycle), 1 box quanh ô tô (nhãn: car).

### 2. YOLO là gì?
"You Only Look Once" — họ model object detection nhanh. YOLOv8 là phiên bản 8. YOLOv8**s** là variant small (nhẹ, cân bằng tốc độ/độ chính xác).

### 3. Train/Val/Test là gì?

| Set | Dùng để | Model có "thấy" không? |
|---|---|---|
| **Train** | Dạy model học | Có — model update weights dựa vào set này |
| **Val** | Theo dõi trong lúc train, chỉnh hyperparameter | Có — model không update weights nhưng ta nhìn số để chọn config |
| **Test** | Đánh giá cuối cùng | **Không** — chỉ chạm sau khi đã chốt model |

### 4. Epoch / Batch / Learning rate

- **1 epoch** = model đi qua **tất cả** ảnh train 1 lần. 40 epoch = đi qua 40 lần.
- **Batch size** = số ảnh model nhìn cùng lúc trước khi update weights. GPU 12GB thường batch 16 với imgsz=960.
- **Learning rate (lr0)** = tốc độ model "sửa mình" khi sai. Cao quá → nhảy loạn, không hội tụ. Thấp quá → học chậm, kẹt ở local optimum.

### 5. mAP là gì?

**mAP** (mean Average Precision) = điểm số đo "model đoán đúng được bao nhiêu %". Càng cao càng tốt (0 → 1).

- **mAP@0.5** = coi là đoán đúng nếu box dự đoán trùng ≥50% với box thật (IoU ≥ 0.5)
- **mAP@0.5:0.95** = trung bình mAP với 10 ngưỡng IoU từ 0.5 → 0.95 (khó hơn)

### 6. Confidence / IoU threshold

- **Confidence** = model "tự tin" bao nhiêu % về prediction. Mặc định filter box có conf < 0.25.
- **IoU NMS** = khi 2 box chồng lên nhau, giữ box tin tưởng cao hơn (nếu IoU vượt ngưỡng). Mặc định 0.7.

### 7. YOLO label format

Mỗi ảnh `abc.jpg` có 1 file `abc.txt` cùng tên:

```
0 0.512 0.634 0.08 0.12
1 0.730 0.550 0.14 0.09
```

Mỗi dòng = 1 bbox: `class_id center_x center_y width height` — tất cả toạ độ **normalize [0,1]** theo kích thước ảnh.

---

## Quy trình 7 bước

```
┌──────────────────────────────────────────────────────────────┐
│  1_check_env.py          Kiểm tra Python, torch, CUDA, GPU   │
│          ↓                                                    │
│  2_prepare_data.py       Verify data layout + count ảnh/bbox │
│          ↓                                                    │
│  3_data.yaml             File config dataset                 │
│          ↓                                                    │
│  4_train.py              TRAIN model (lâu nhất, ~2-6h)       │
│          ↓                                                    │
│  5_evaluate.py           Chấm điểm mAP trên val/test         │
│          ↓                                                    │
│  6_inference.py          Chạy trên ảnh/video mới             │
│          ↓                                                    │
│  7_plot_results.py       Vẽ biểu đồ loss, mAP, confusion     │
└──────────────────────────────────────────────────────────────┘
```

Thời gian tổng:
- Bước 1-3: 5 phút
- Bước 4 (train): 2-6 giờ tuỳ GPU
- Bước 5-7: 15 phút

---

## Chuẩn bị môi trường (chạy 1 lần)

```bash
# Trong terminal, đứng ở thư mục gốc project
cd /home/xuand/vn_traffic_ai

# Tạo conda env (nếu chưa có)
conda create -n yolov8_ft python=3.11 -y
conda activate yolov8_ft

# Cài dependencies
pip install -r requirements.txt

# Kiểm tra GPU
python tutorial/1_check_env.py
```

---

## Chạy từng bước

```bash
# Bước 1 — Check môi trường
python tutorial/1_check_env.py

# Bước 2 — Verify data (data đã có sẵn ở data/merged_v5/)
python tutorial/2_prepare_data.py

# Bước 3 — Xem file config (đọc comment)
cat tutorial/3_data.yaml

# Bước 4 — TRAIN (chạy lâu, nên để qua đêm hoặc dùng tmux/screen)
python tutorial/4_train.py

# Bước 5 — Eval trên val + test
python tutorial/5_evaluate.py

# Bước 6 — Inference trên 1 video thử
python tutorial/6_inference.py --video path/to/video.mp4

# Bước 7 — Vẽ biểu đồ
python tutorial/7_plot_results.py
```

---

## Khi gặp lỗi

| Lỗi | Fix |
|---|---|
| `CUDA out of memory` | Giảm `batch` trong `4_train.py` (16 → 8) hoặc `imgsz` (960 → 640) |
| `No module named ultralytics` | Chưa `pip install -r requirements.txt` |
| `FileNotFoundError: data/merged_v5/...` | Chưa có data — xem [docs/DATA_GUIDE.md](../docs/DATA_GUIDE.md) để setup |
| `KeyError: 'class_id'` | File label (.txt) sai format — xem bước 2 để verify |

---

## Sau khi làm xong tutorial

Bạn sẽ hiểu:
- Flow data: raw images + labels → split → train → model.pt
- Hyperparameters quan trọng của YOLOv8 (và chọn thế nào)
- Đọc output training (loss, mAP curves)
- Chạy model đã train trên data mới

Bước tiếp theo:
- Thử đổi `imgsz`, `batch`, `lr0` để xem mAP thay đổi thế nào
- Thử model variant khác (yolov8n = nano, yolov8m = medium, yolov8l = large)
- Áp dụng vào dataset của bạn (chỉ cần tạo `data.yaml` mới + đổi `path`)
