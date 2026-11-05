# Workflow V6 — Pseudo-labeling pipeline (chi tiết từng bước)

> Tài liệu này dùng để chèn vào tiểu luận (Chương 4 — Cải tiến mô hình)
> và đồng thời là checklist thực hiện cho bản thân.

---

## 1. Tổng quan pipeline

```
[Video CCTV mới]
      │
      ▼
[B1] Thu thập video nguồn (yt-dlp)
      │
      ▼
[B2] Trích frame (ffmpeg, 1 fps/2s)
      │
      ▼
[B3] Inference V5 → pseudo-label (YOLO)
      │
      ▼
[B4] Lọc tự động (confidence, size, NMS)
      │
      ▼
[B5] Spot-check tay 500 frame (FiftyOne)
      │  (đo precision pseudo-label)
      ▼
[B6] Merge pseudo + gold → train V6
      │
      ▼
[B7] Evaluate V5 vs V6 trên test_v5
      │
      ▼
[Báo cáo kết quả]
```

---

## 2. Chuẩn bị môi trường (một lần)

### 2.1. Cài công cụ
```bash
# Trong conda env của project
pip install yt-dlp ultralytics fiftyone wandb

# ffmpeg (hệ thống)
sudo apt install ffmpeg          # Ubuntu/WSL
# hoặc: brew install ffmpeg      # macOS
```

### 2.2. Cấu trúc thư mục V6
```
data/
├── v5_gold/                  # dataset V5 đã có (train/val/test)
│   ├── train/
│   ├── val/
│   └── test/
├── raw/
│   ├── pseudo_sources.txt    # danh sách URL video
│   ├── videos/               # video tải về
│   └── frames/               # frame trích xuất
└── pseudo_v6/
    ├── round1/               # output YOLO inference
    ├── accepted/             # frame qua filter tự động
    ├── rejected/             # frame bị loại
    ├── review_tags.csv       # kết quả spot-check
    └── final/                # merge gold + pseudo cho train
```

---

## 3. Chi tiết từng bước

### Bước 1 — Thu thập video nguồn

**Mục tiêu**: 20–40 video CCTV VN, KHÁC video_id với V5

**Cách làm**:
1. Tìm trên YouTube với từ khoá:
   - `"CCTV Hà Nội giao thông"`, `"camera giao thông HCM live"`,
     `"vn_cctv highway"`, `"traffic camera Vietnam"`
2. Chọn video đa dạng:
   - Giờ cao điểm (sáng 7–9h, chiều 17–19h)
   - Giờ đêm (sau 22h) — test low-light
   - Thời tiết: nắng, mưa, sương (nếu có)
   - Góc camera: trên cao (bird-eye), ngang đường
3. Lưu URL vào `data/raw/pseudo_sources.txt`:
   ```
   https://youtube.com/watch?v=xxx   # HN - Nguyen Trai - 7h sang
   https://youtube.com/watch?v=yyy   # HCM - Vo Van Kiet - mua
   ...
   ```

**Checklist**:
- [ ] ≥20 URL
- [ ] Không trùng video đã dùng cho V5
- [ ] Ghi chú địa điểm + điều kiện mỗi URL

---

### Bước 2 — Download & trích frame

**Script**: `scripts/pseudo/01_download_extract.sh`

```bash
#!/bin/bash
# Đọc từng URL, download và trích frame 0.5 fps

while IFS= read -r line; do
    # Bỏ qua comment và dòng trống
    [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
    url=$(echo "$line" | awk '{print $1}')

    # Download video
    yt-dlp -f 'best[height<=720]' \
           -o "data/raw/videos/%(id)s.%(ext)s" \
           "$url"

    # Lấy video_id
    vid=$(yt-dlp --get-id "$url")
    mkdir -p "data/raw/frames/$vid"

    # Trích frame 0.5 fps (1 frame / 2 giây)
    ffmpeg -i "data/raw/videos/$vid."* \
           -vf "fps=0.5" \
           -q:v 2 \
           "data/raw/frames/$vid/frame_%06d.jpg"

done < data/raw/pseudo_sources.txt
```

**Chạy**:
```bash
chmod +x scripts/pseudo/01_download_extract.sh
./scripts/pseudo/01_download_extract.sh
```

**Kết quả mong đợi**:
- Mỗi video 10 phút → ~300 frame
- 30 video → ~9000 frame ứng viên
- Dung lượng: video ~5 GB, frame ~2 GB

**Checklist**:
- [ ] Mỗi folder `frames/<video_id>/` có frame
- [ ] Kiểm tra ngẫu nhiên 5–10 ảnh xem rõ, không bị chèn logo che

---

### Bước 3 — Inference pseudo-label

**Script**: `scripts/pseudo/02_infer.py`

```python
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "runs/v5/train/weights/best.pt"
FRAMES_DIR = "data/raw/frames"
OUTPUT_DIR = "data/pseudo_v6/round1"

model = YOLO(MODEL_PATH)

# Chạy inference toàn bộ frame
results = model.predict(
    source=FRAMES_DIR,
    conf=0.50,           # lấy rộng, lọc chặt ở B4
    iou=0.5,
    save=False,          # không cần ảnh visualize
    save_txt=True,       # xuất .txt YOLO format
    save_conf=True,      # giữ confidence
    project=OUTPUT_DIR,
    name="predict",
    exist_ok=True,
    stream=True          # tránh OOM với nhiều ảnh
)

# Consume generator
for _ in results:
    pass

print("Done. Labels tại:", f"{OUTPUT_DIR}/predict/labels/")
```

**Chạy**:
```bash
python scripts/pseudo/02_infer.py
```

**Format output** `.txt` YOLO (mỗi dòng 1 box):
```
<class_id> <x_center> <y_center> <w> <h> <confidence>
```

---

### Bước 4 — Lọc tự động

**Script**: `scripts/pseudo/03_filter.py`

Logic lọc:
| Tiêu chí | Ngưỡng | Lý do |
|---|---|---|
| Confidence per-class | car ≥0.75, motorbike ≥0.85, truck ≥0.80, bus ≥0.80 | Class khó đặt ngưỡng cao hơn |
| Box area | ≥ 400 px² (20×20) | Model V5 yếu ở small object |
| Số box / frame | ≥ 1 | Frame trống vô dụng cho train |
| Tỉ lệ box low-conf | <30% | Frame "khó" → chuyển pool review |

Output:
- `data/pseudo_v6/accepted/images/` + `labels/`
- `data/pseudo_v6/rejected/` (để sampling review)
- `data/pseudo_v6/stats.csv` — số box giữ/bỏ mỗi class

**Chạy**:
```bash
python scripts/pseudo/03_filter.py \
  --input data/pseudo_v6/round1/predict \
  --frames data/raw/frames \
  --output data/pseudo_v6
```

**Kết quả mong đợi**:
- Giữ 40–60% frame ban đầu
- Báo cáo `stats.csv` → **đưa vào tiểu luận** (bảng phân bố pseudo-label theo class)

---

### Bước 5 — Spot-check 500 frame với FiftyOne

**Mục tiêu**: Đo **precision pseudo-label** để báo cáo

**Script**: `scripts/pseudo/04_review.py`

```python
import fiftyone as fo
import random

ACCEPTED_DIR = "data/pseudo_v6/accepted"

# Load dataset YOLO format
dataset = fo.Dataset.from_dir(
    dataset_dir=ACCEPTED_DIR,
    dataset_type=fo.types.YOLOv5Dataset,
    name="pseudo_v6_review",
    overwrite=True
)

# Chọn ngẫu nhiên 500 frame để review (seed cố định)
random.seed(42)
sample_ids = random.sample(dataset.values("id"), min(500, len(dataset)))
review_view = dataset.select(sample_ids)

# Tag sẵn các label để review
review_view.tag_samples("to_review")

# Launch UI
session = fo.launch_app(review_view)
session.wait()
```

**Chạy**:
```bash
python scripts/pseudo/04_review.py
# Mở browser: http://localhost:5151
```

**Thao tác trong UI**:
1. Click từng ảnh → xem bbox
2. Với mỗi ảnh, đánh giá tổng thể:
   - **Tag `correct`** (phím `c`): tất cả bbox đúng class + đúng vị trí
   - **Tag `partial`** (phím `p`): có 1–2 bbox sai, còn lại đúng
   - **Tag `wrong`** (phím `w`): đa số sai
3. Keyboard shortcut: `→` next, `←` prev
4. Thời gian ước tính: **500 frame × 20s = ~2.5 giờ** (chia 2–3 buổi)

**Sau khi review xong**, export kết quả:

```python
# Trong Python/Jupyter
import fiftyone as fo

ds = fo.load_dataset("pseudo_v6_review")

correct   = len(ds.match_tags("correct"))
partial   = len(ds.match_tags("partial"))
wrong     = len(ds.match_tags("wrong"))
total     = correct + partial + wrong

precision_strict = correct / total
precision_loose  = (correct + partial) / total

print(f"Pseudo-label precision (strict): {precision_strict:.2%}")
print(f"Pseudo-label precision (loose) : {precision_loose:.2%}")

# Export CSV
ds.export(
    export_dir="data/pseudo_v6/review_result",
    dataset_type=fo.types.CSVDataset,
    fields=["filepath", "tags"]
)
```

**Quyết định**:
| Precision strict | Hành động |
|---|---|
| ≥ 85% | OK, dùng toàn bộ pseudo accepted cho train V6 |
| 70–85% | Tăng conf threshold lên, chạy lại B3+B4 |
| < 70% | Pseudo-label noisy quá, xem lại từ đầu |

**⚠️ Quan trọng**: Con số precision này **PHẢI viết vào tiểu luận** — là bằng chứng pseudo-label có kiểm soát chất lượng.

---

### Bước 6 — Merge dataset & train V6

**Script**: `scripts/pseudo/05_build_dataset.py`

Logic:
```python
# train_v6 = train_v5_gold + pseudo_accepted (tỉ lệ 1:1, cap để cân bằng)
# val_v6   = val_v5    (KHÔNG đổi — honest)
# test_v6  = test_v5   (KHÔNG đổi — honest)
```

- **Giữ trace**: mỗi pseudo frame gắn `source_video_id` trong filename →
  sau này audit được leak
- **Không để pseudo frame có video_id trùng val/test**:
  đoạn này cần script check:

```python
val_test_vids = set(get_video_ids("data/v5_gold/val") +
                    get_video_ids("data/v5_gold/test"))
pseudo_vids   = set(get_video_ids("data/pseudo_v6/accepted"))
leak = val_test_vids & pseudo_vids
assert len(leak) == 0, f"LEAK DETECTED: {leak}"
```

**Config YOLO**: `configs/v6.yaml`
```yaml
path: ../data/pseudo_v6/final
train: train
val: ../v5_gold/val
test: ../v5_gold/test
nc: 6
names: [motorbike, car, truck, bus, bicycle, person]
```

**Train**:
```bash
yolo train \
  data=configs/v6.yaml \
  model=yolov8s.pt \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  project=runs/v6 \
  name=round1_pseudo \
  patience=20
```

**Track với W&B** (tuỳ chọn, 1 lần):
```bash
wandb login              # paste API key từ wandb.ai
# Ultralytics tự log vào W&B khi đã login
```

---

### Bước 7 — So sánh V5 vs V6 (honest)

**Script**: `scripts/pseudo/06_compare.py`

```python
from ultralytics import YOLO

TEST_DATA = "configs/v5.yaml"   # test set GỐC, không đổi

for name, weight in [
    ("V5_gold", "runs/v5/train/weights/best.pt"),
    ("V6_pseudo", "runs/v6/round1_pseudo/weights/best.pt"),
]:
    model = YOLO(weight)
    metrics = model.val(data=TEST_DATA, split="test")
    print(f"\n=== {name} ===")
    print(f"mAP@50   : {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Per-class AP50: {dict(zip(metrics.names.values(), metrics.box.ap50))}")
```

**Bảng kết quả cho tiểu luận**:

| Model | Train size | mAP@50 | mAP@50-95 | AP motorbike | AP truck |
|---|---|---|---|---|---|
| V5 (gold, baseline) | N ảnh | 0.70 | 0.45 | 0.65 | 0.72 |
| V6 (gold + pseudo)  | 2N ảnh | ?.?? | ?.?? | ?.?? | ?.?? |
| **Δ** | — | +?.?? | +?.?? | +?.?? | +?.?? |

**Phân tích cho báo cáo**:
- Nếu Δ mAP ≈ +1–4% → **hợp lý**, pseudo-label hiệu quả thật
- Nếu Δ > +10% → nghi ngờ leak, kiểm tra lại script B6
- Nếu Δ < 0 → pseudo-label noisy, cần tăng threshold hoặc weighting

---

## 4. Checklist tổng cho tiểu luận

Khi viết Chương "Cải tiến V6", đảm bảo đề cập:

- [ ] Nguồn video pseudo-label (số video, địa điểm, điều kiện)
- [ ] Ngưỡng confidence per-class (bảng cụ thể)
- [ ] Số frame pseudo ban đầu → sau filter → sau review
- [ ] **Precision pseudo-label strict/loose** (từ B5)
- [ ] Kiểm tra leak giữa pseudo và val/test (B6 — có script assert)
- [ ] Bảng so sánh V5 vs V6 trên **cùng test set**
- [ ] Phân tích: cải thiện ở class nào, không cải thiện ở class nào, tại sao
- [ ] Hạn chế: confirmation bias, pseudo chỉ tốt khi V5 đã khá ổn

---

## 5. Thời gian ước lượng

| Bước | Thời gian | Người làm |
|---|---|---|
| B1. List video | 1–2 giờ | Bạn |
| B2. Download + extract | 2–4 giờ (I/O) | Bạn (chạy script) |
| B3. Inference | 30p–1h (GPU) | Bạn (chạy script) |
| B4. Filter | 10 phút | Bạn (chạy script) |
| B5. Spot-check 500 frame | 2–3 giờ | **Bạn — không uỷ quyền** |
| B6. Train V6 | 4–12 giờ (GPU) | Bạn (chạy script) |
| B7. Evaluate + viết bảng | 1 giờ | Bạn + Claude |

**Tổng: 2–3 ngày tập trung** cho 1 chương cải tiến chất lượng.

---

## 6. Rủi ro & cách xử lý

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| Confirmation bias | V6 cải thiện 1 class V5 đã mạnh, class yếu vẫn yếu | Report thẳng, đừng che |
| Video leak | V6 tăng mAP >10% bất thường | Chạy assert ở B6, xem lại |
| Pseudo-label quá noisy | Precision strict <70% | Tăng conf threshold B4 |
| Dataset imbalance | 1 class chiếm >60% pseudo | Cap số sample / class khi merge |
| Overfit domain mới | V6 tăng ở pseudo-val nhưng giảm ở test gold | Giảm tỉ lệ pseudo trong train |

---

*Tài liệu cập nhật: 2026-11-05*
