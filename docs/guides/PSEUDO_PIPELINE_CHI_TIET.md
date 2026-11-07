# Pipeline Pseudo-Labeling V6 — Tài liệu chi tiết dành cho người mới

> **Đối tượng**: người chưa rành tech, cần hiểu toàn bộ quy trình.
> **Tài liệu này giải thích**: từng dòng code, từng URL, từng thư mục, YOLO xử lý thế nào, dữ liệu đi đâu.

---

## 📖 Mục lục

1. [Tổng quan — pseudo-labeling là gì?](#1-tổng-quan)
2. [Sơ đồ pipeline toàn cảnh](#2-sơ-đồ-pipeline)
3. [Cấu trúc thư mục scripts/pseudo/](#3-cấu-trúc-thư-mục)
4. [Chi tiết từng bước](#4-chi-tiết-từng-bước)
   - [B1 — Scrape URL camera](#b1-scrape-url-camera)
   - [B2 — Collect snapshot](#b2-collect-snapshot)
   - [B3 — Dedup ảnh trùng](#b3-dedup-ảnh-trùng)
   - [B4 — Inference V5](#b4-inference-v5)
   - [B5 — Filter toán học](#b5-filter-toán-học)
   - [B6 — Visualize review](#b6-visualize-review)
   - [B7 — Remove bad samples](#b7-remove-bad)
5. [Cấu trúc thư mục dữ liệu](#5-cấu-trúc-thư-mục-dữ-liệu)
6. [Config files quan trọng](#6-config-files)
7. [Troubleshooting thường gặp](#7-troubleshooting)

---

## 1. Tổng quan

### Pseudo-labeling là gì?

Thông thường để train 1 model AI nhận diện xe cộ, ta cần:
- Thu thập 10,000 ảnh
- Thuê người **gán nhãn bằng tay** mỗi ảnh (vẽ khung quanh xe, ghi "xe máy" / "ô tô" / ...)
- Tốn **thời gian và tiền bạc** rất lớn

**Pseudo-labeling** là thủ thuật:
- Có sẵn 1 model đã train (V5) chạy ở độ chính xác ~80%
- Dùng V5 **tự gán nhãn** cho ảnh mới (không cần người vẽ tay)
- Chấp nhận nhãn có ~15% sai, nhưng được **miễn phí**
- Lọc tự động + người duyệt mẫu nhỏ để đo chất lượng
- Train model mới (V6) với dataset lớn hơn → V6 giỏi hơn V5

### Tại sao đáng làm

```
V5:     train với 5,000 ảnh gold → mAP 70%
V6:     train với 5,000 ảnh gold + 4,000 ảnh pseudo → mAP kỳ vọng 73-75%
Chi phí thêm: 0 đồng gán nhãn, chỉ tốn code + thời gian cào data
```

### Nguồn dữ liệu

Camera giao thông **công khai** của Sở GTVT TP.HCM:
- **782 camera** phủ 138 phường
- Ảnh snapshot JPEG 512×288, refresh mỗi **15 giây**
- Truy cập qua trang bên thứ 3: `cameragiaothong.net`
- Hợp pháp cho nghiên cứu học thuật

---

## 2. Sơ đồ pipeline

### Toàn cảnh 7 bước

```
                    ┌─────────────────────────────────────────┐
                    │  INTERNET — cameragiaothong.net         │
                    │  (wrapper của Sở GTVT HCM)              │
                    └────────────────┬────────────────────────┘
                                     │
                                     ▼  cào 1 lần duy nhất (~20 phút)
         ┌───────────────────────────────────────────┐
  B1 ──  │  01_scrape_camera_urls.py                 │
         │  Duyệt 800 ID → lấy 782 URL ảnh           │
         └───────────────┬───────────────────────────┘
                         │
                         ▼
              📁 configs/hcm_cameras.yaml
              (782 URL camera + tên + phường)
                         │
                         ▼  mỗi session 30 phút (chạy nhiều lần)
         ┌───────────────────────────────────────────┐
  B2 ──  │  02_collect_snapshots.py                  │
         │  Chọn 30-40 cam random → tải ảnh 20s/vòng │
         │  (bỏ qua blacklist + cam đã sample trước) │
         └───────────────┬───────────────────────────┘
                         │
                         ▼
              📁 data/raw/frames/hcm/<phường>/
              (ảnh .jpg, ~1500-2500 ảnh/session)
                         │
                         ▼  ~1 phút
         ┌───────────────────────────────────────────┐
  B3 ──  │  03_dedup.py                              │
         │  Loại ảnh gần giống bằng pHash            │
         └───────────────┬───────────────────────────┘
                         │
                         ▼  (xoá ~13% ảnh trùng)
                         │
                         ▼  ~2-3 phút GPU
         ┌───────────────────────────────────────────┐
  B4 ──  │  04_inference.py                          │
         │  YOLO V5 chạy trên mỗi ảnh                │
         │  → tự động vẽ bbox + gán class            │
         │  Chia chunks 500 + auto resume            │
         └───────────────┬───────────────────────────┘
                         │
                         ▼
              📁 data/pseudo_v6/round1/predict/labels/
              (file .txt YOLO format cho mỗi ảnh)
                         │
                         ▼  ~10 giây
         ┌───────────────────────────────────────────┐
  B5 ──  │  05_filter.py                             │
         │  Lọc box: confidence thấp, size nhỏ       │
         └───────────────┬───────────────────────────┘
                         │
                         ▼
              📁 data/pseudo_v6/accepted/
              (ảnh + label đã qua filter toán học)
                         │
                         ▼  ~1 phút
         ┌───────────────────────────────────────────┐
  B6 ──  │  06_visualize_review.py                   │
         │  Vẽ bbox màu lên 500 ảnh ngẫu nhiên       │
         └───────────────┬───────────────────────────┘
                         │
                         ▼
              📁 data/pseudo_v6/review/
              (500 ảnh có bbox vẽ sẵn)
                         │
                         ▼  2-3 giờ người duyệt
         ┌───────────────────────────────────────────┐
  B7 ──  │  07_remove_bad.py                         │
         │  Xoá ảnh WRONG do người đánh dấu +        │
         │  xoá toàn bộ camera trong blacklist       │
         └───────────────┬───────────────────────────┘
                         │
                         ▼
                   🎯 Dataset V6 sạch
                   (sẵn sàng train model V6)
```

---

## 3. Cấu trúc thư mục

### `scripts/pseudo/` — các script

```
scripts/pseudo/
├── 01_scrape_camera_urls.py          ← Thu URL camera (1 lần)
├── 01_scrape_camera_urls_opencctv_DEPRECATED.py  ← Phiên bản cũ (bỏ)
├── 02_collect_snapshots.py           ← Tải ảnh từ camera
├── 03_dedup.py                       ← Loại ảnh trùng
├── 04_inference.py                   ← YOLO infer → pseudo-label
├── 05_filter.py                      ← Filter toán học
├── 06_visualize_review.py            ← Vẽ bbox cho review
├── 07_remove_bad.py                  ← Xoá ảnh sai sau review
└── run.sh                            ← Menu wrapper gọn nhẹ
```

**Quy ước đánh số**: 01→07 là thứ tự chạy. Luôn chạy tuần tự, không nhảy cóc.

---

## 4. Chi tiết từng bước

### B1 — Scrape URL camera

**File**: `01_scrape_camera_urls.py`
**Mục đích**: Lấy 782 URL ảnh camera từ internet, lưu vào yaml.

#### 4.1.1. Nguồn dữ liệu

Có 3 lớp:

```
┌────────────────────────────────────────────────────┐
│  LỚP 1 (gốc):                                      │
│  giaothong.hochiminhcity.gov.vn                    │
│  (Sở GTVT HCM — khó scrape, map load bằng JS)      │
└─────────────────┬──────────────────────────────────┘
                  │  Lớp 2 lấy ảnh từ đây
                  ▼
┌────────────────────────────────────────────────────┐
│  LỚP 2 (wrapper):                                  │
│  cameragiaothong.net                               │
│  ID dễ enumerate: hochiminh-city-000001..000800    │
└─────────────────┬──────────────────────────────────┘
                  │  Script của mình đi qua lớp 2 này
                  ▼
┌────────────────────────────────────────────────────┐
│  LỚP 3 (code của mình):                            │
│  01_scrape_camera_urls.py                          │
└────────────────────────────────────────────────────┘
```

**URL mẫu**:
```
Trang detail của 1 camera trên cameragiaothong.net:
  https://www.cameragiaothong.net/cctv?id=hochiminh-city-000005

Trong HTML của trang đó, có thẻ <img> trỏ đến ảnh thật:
  <img src="https://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id=662b86c41afb9c00172dd31c">

ID cuối (662b86c4...) là MongoDB ObjectID — ID thật của camera trong hệ thống Sở GTVT.
```

#### 4.1.2. Cơ chế scrape

```
cho i từ 1 đến 800:
    url = f"https://www.cameragiaothong.net/cctv?id=hochiminh-city-{i:06d}"
    html = requests.get(url).text

    # Dùng regex tìm URL ảnh trong HTML
    match = re.search(r"giaothong\.hochiminhcity\.gov\.vn/render/ImageHandler\.ashx\?id=([a-f0-9]+)", html)
    if match:
        image_url = "https://" + match.group(0)
        yaml['cameras'][f"{i:06d}"] = {
            'id': f"{i:06d}",
            'handler_id': match.group(1),
            'name': extract_name_from_title(html),
            'ward': extract_ward_from_breadcrumb(html),
            'image_url': image_url,
        }

    sleep(1.5)  # lịch sự với server, tránh bị ban
```

#### 4.1.3. Output — `configs/hcm_cameras.yaml`

```yaml
meta:
  source: cameragiaothong.net
  upstream: giaothong.hochiminhcity.gov.vn (Sở GTVT HCM)
  city: ho-chi-minh-city

cameras:
  '000001':
    id: '000001'
    handler_id: 662b86c41afb9c00172dd31c
    name: "Camera Trần Quang Khải - Trần Khắc Chân, Phường Tân Định"
    ward: phuong-tan-dinh
    detail_url: https://www.cameragiaothong.net/cctv?id=hochiminh-city-000001
    image_url: https://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id=662b86c41afb9c00172dd31c
    error: null

  '000002':
    ...  # 782 camera tổng cộng
```

#### 4.1.4. Khi nào cần chạy lại

**Hầu như KHÔNG BAO GIỜ**. Chạy 1 lần duy nhất, commit yaml vào git, mọi người dùng chung.

Chạy lại chỉ khi:
- Sở GTVT HCM đổi cấu trúc URL (hiếm)
- cameragiaothong.net đổi HTML (hiếm)

---

### B2 — Collect snapshot

**File**: `02_collect_snapshots.py`
**Mục đích**: Tải ảnh thật từ 30-40 camera trong 30 phút.

#### 4.2.1. Vấn đề "refresh" của camera

Server Sở GTVT serve ảnh snapshot (không phải video). Ảnh **update mỗi ~15 giây** trên server.

```
      Thời gian →
      00s      15s      30s      45s      60s
      │         │         │         │         │
      │  ảnh A  │  ảnh B  │  ảnh C  │  ảnh D  │  ảnh E
      │         │         │         │         │
Server update ảnh mới ──▼──▼──▼──▼──▼

Nếu script của mình fetch mỗi 20s:
  fetch 00s → ảnh A ✓
  fetch 20s → ảnh B ✓ (khác A)
  fetch 40s → ảnh C ✓ (khác B)
```

→ Mỗi camera cho ra ~**2-3 ảnh unique mỗi phút**.

#### 4.2.2. Cơ chế chọn camera

Script có 3 lớp lọc:

```
782 cam (tất cả)
     │
     ▼ trừ blacklist (7 cam xấu)
775 cam
     │
     ▼ trừ history (các cam đã sample session trước)
X cam còn lại
     │
     ▼ random 40 cam (seed cố định cho reproducible)
40 cam để session này
```

**File chứa blacklist**: `configs/hcm_cameras_blacklist.yaml`
**File chứa history**: `data/raw/sampled_cameras_history.txt`

Sau mỗi session, 40 ID vừa sample được ghi thêm vào history → session sau không trùng.

#### 4.2.3. Cơ chế download + crop overlay

```
┌─────────────────────────────────────────────┐
│  Ảnh gốc từ server (512×288 JPEG)           │
│  ┌─────────────────────────────────────┐    │
│  │ ⏱ 05.Nov 2026 11:35:07 — Cổng TDC2 │  ← overlay 50px trên
│  ├─────────────────────────────────────┤    │
│  │                                     │    │
│  │                                     │    │
│  │         [ảnh đường phố]             │    │
│  │                                     │    │
│  │                                     │    │
│  └─────────────────────────────────────┘    │
└──────────────────┬──────────────────────────┘
                   │  crop bỏ 50px trên (dùng PIL)
                   ▼
┌─────────────────────────────────────────────┐
│  Ảnh sau crop (512×238 JPEG)                │
│  ┌─────────────────────────────────────┐    │
│  │                                     │    │
│  │                                     │    │
│  │         [ảnh đường phố]             │    │
│  │                                     │    │
│  │                                     │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

**Tại sao bỏ overlay**: text timestamp lặp lại mỗi ảnh → model có thể học sai feature "chữ trắng ở góc = xe cộ".

#### 4.2.4. Đặt tên file

```
hcm_<ward>_<camera_id>_<timestamp>.jpg

Ví dụ:
hcm_phuong-tan-dinh_000001_20261105_113507.jpg
└┬┘ └──────┬──────┘ └┬───┘ └────┬────┘
 │         │         │          │
 │         │         │          └─ thời gian chụp
 │         │         └──────────── ID camera
 │         └────────────────────── phường
 └──────────────────────────────── city
```

→ Từ tên file đoán được cam nào, giờ nào, khu nào.

#### 4.2.5. Output

```
data/raw/frames/hcm/
├── phuong-tan-dinh/
│   ├── hcm_phuong-tan-dinh_000001_20261105_113507.jpg
│   ├── hcm_phuong-tan-dinh_000001_20261105_113529.jpg
│   ├── ...
├── phuong-cat-lai/
│   ├── hcm_phuong-cat-lai_000026_20261105_113522.jpg
│   └── ...
└── <83 ward folders khác>/
```

**Thống kê điển hình** (session 30 phút, 40 cam):
- Rounds: ~70 (= 30 min × 60s / 25s/round)
- Ảnh/round: 40
- Ảnh tổng: ~2800
- Dung lượng: ~120 MB

---

### B3 — Dedup ảnh trùng

**File**: `03_dedup.py`
**Mục đích**: Loại ảnh gần giống nhau để tránh overfit.

#### 4.3.1. Tại sao cần dedup

Dù fetch cách nhau 20s, vẫn có ảnh giống nhau vì:
- Server đôi khi chưa refresh kịp → trả cùng ảnh cũ
- Hoặc ảnh thật nhưng xe chưa di chuyển (đèn đỏ)

Nếu train với 100 ảnh giống nhau → model overfit scene đó.

#### 4.3.2. pHash — perceptual hash

**MD5 (hash byte)**: đổi 1 pixel → hash khác hẳn → không bắt được near-duplicate.

**pHash (perceptual hash)**: hash nội dung thị giác, không phải byte.

```
Ảnh gốc 512×288
     │
     ▼ resize 32×32
     │
     ▼ convert grayscale
     │
     ▼ DCT (Discrete Cosine Transform)
     │
     ▼ lấy 8×8 block tần số thấp
     │
     ▼ so với median
     │
     ▼
64-bit hash: 0b101101...01110101
```

**So sánh 2 hash**: Hamming distance = số bit khác nhau.
- 0 bit khác → ảnh identical
- 1-3 bit khác → gần như giống (near-duplicate)
- >10 bit khác → khác hẳn

#### 4.3.3. Cơ chế sliding window

```
Giả sử có 10 ảnh cùng camera, sắp xếp theo filename (= theo thứ tự thời gian):

Ảnh:  A  B  C  D  E  F  G  H  I  J
      │  │  │  │  │  │  │  │  │  │
      pHash lần lượt

Với mỗi ảnh, so sánh với 5 ảnh trước đó:
  A: không so (không có trước)
  B: so với A                     → nếu dup, đánh dấu B
  C: so với A, B                  → nếu dup với bất kỳ, đánh dấu C
  D: so với A, B, C               → ...
  E: so với A, B, C, D            → ...
  F: so với B, C, D, E (bỏ A)     → sliding window 5
  ...

Kết quả: ảnh được đánh dấu sẽ bị xoá.
```

#### 4.3.4. Output

```bash
# Dry-run — chỉ report
python 03_dedup.py

# Output mẫu:
# [phuong-tan-dinh] total=70 dup=4
# [phuong-cat-lai]  total=68 dup=3
# ...
# [SUMMARY] total=2800 dup=350 keep=2450 (87.5%)

# Apply — thực sự xoá
python 03_dedup.py --apply
```

**Thống kê điển hình**: giữ 85-90% ảnh unique.

---

### B4 — Inference V5

**File**: `04_inference.py`
**Mục đích**: Dùng model V5 đã train chạy trên ảnh mới → tự tạo nhãn bbox.

#### 4.4.1. YOLO inference là gì?

YOLO (You Only Look Once) là model detection:
- **Input**: 1 ảnh (bất kỳ kích thước, nhưng thường resize 640×640)
- **Output**: list các bounding box, mỗi box có:
  - Toạ độ (x, y, width, height)
  - Class (motorcycle / car / bus / truck)
  - Confidence (độ tin cậy 0.0 - 1.0)

#### 4.4.2. Sơ đồ inference 1 ảnh

```
┌──────────────────┐
│  Ảnh JPEG        │
│  512×238         │
└────────┬─────────┘
         │
         ▼  resize + normalize
┌──────────────────┐
│  Tensor PyTorch  │
│  (1, 3, 640, 640)│   [batch=1, 3 kênh RGB, 640×640]
└────────┬─────────┘
         │
         ▼  đi qua model V5 (CNN backbone + detection head)
┌──────────────────┐
│  Raw predictions │
│  (grid 20×20,    │   model dự đoán ở 3 scale (20×20, 40×40, 80×80)
│   40×40, 80×80)  │   mỗi cell dự đoán anchor box
└────────┬─────────┘
         │
         ▼  non-max suppression (bỏ box overlap)
┌──────────────────┐
│  Final boxes     │
│  [5 boxes]:      │
│  box 1: car 0.87 │
│  box 2: motor 0.76│
│  box 3: truck 0.65│
│  box 4: car 0.55 │
│  box 5: car 0.42 │   ← lọc tiếp bằng conf threshold
└──────────────────┘
```

#### 4.4.3. Format file label YOLO

```
# File: hcm_phuong-tan-dinh_000001_20261105_113507.txt
# Mỗi dòng = 1 bounding box
# Format: <class_id> <x_center> <y_center> <width> <height> <confidence>

0 0.350 0.540 0.120 0.220 0.830
1 0.620 0.410 0.180 0.280 0.720
3 0.100 0.600 0.090 0.150 0.650
```

**Giải thích**:
- `0` = motorcycle (class ID theo thứ tự train)
- `0.350 0.540` = tâm box ở 35% chiều ngang, 54% chiều dọc của ảnh
- `0.120 0.220` = width 12%, height 22% của ảnh
- `0.830` = model tin 83% đây là motorcycle

**Lưu ý**: toạ độ **normalized** (0.0 - 1.0), không phải pixel. Lợi: ảnh đổi kích thước, label vẫn dùng được.

#### 4.4.4. Vấn đề OOM — tại sao cần chunking?

```
Nếu truyền 3518 ảnh 1 lần vào model:
  [paths ──────────────▶ Ultralytics load dataset
                        │
                        ▼ giữ metadata, image tensor trong RAM
                        │
                        ▼ GPU: forward pass từng batch
                        │
                        ▼ accumulate output trong RAM
                        │
                        ▼ CUDA cache fragment theo thời gian
                        │
                        ▼ chunk 1000 ảnh → cache 2GB
                        │
                        ▼ chunk 2000 ảnh → cache 5GB
                        │
                        ▼ chunk 3000 ảnh → cache 10GB → 💥 OOM
```

**Giải pháp chunking**:
```
Chia 3518 ảnh thành 8 chunks × 500 ảnh.

for mỗi chunk:
    model.predict(chunk)         # 500 ảnh
    torch.cuda.empty_cache()     # giải phóng GPU cache
    gc.collect()                 # dọn RAM
    # chunk sau bắt đầu sạch sẽ
```

#### 4.4.5. Resume logic

Script đã có cơ chế resume: chạy lại = tiếp tục từ chỗ dừng.

```python
# Scan folder output
done_stems = set()
for txt in labels_dir.glob("*.txt"):
    done_stems.add(txt.stem)
# Ví dụ: 2852 stems đã có

# Lọc frames chưa có label
pending = [p for p in all_frames if p.stem not in done_stems]
# Ví dụ: còn 666 ảnh

# Chạy tiếp
run_inference(pending)
```

→ Crash OOM? Chạy lại 1 lệnh, auto tiếp tục.

#### 4.4.6. Output

```
data/pseudo_v6/round1/predict/
└── labels/
    ├── hcm_phuong-tan-dinh_000001_20261105_113507.txt
    ├── hcm_phuong-tan-dinh_000001_20261105_113529.txt
    └── ...  (1 file .txt cho mỗi ảnh có >=1 detection)
```

**Thống kê điển hình**: 97% ảnh có ≥1 detection. Ảnh không có detection = ảnh trống (đường vắng, lỗi camera, v.v.).

---

### B5 — Filter toán học

**File**: `05_filter.py`
**Mục đích**: Lọc bỏ box kém chất lượng TRƯỚC khi cho người duyệt.

#### 4.5.1. 3 tiêu chí lọc

**Tiêu chí 1: Per-class confidence threshold**

```
Class       Threshold   Lý do
─────────── ─────────── ──────────────────────────────────────
motorcycle  0.60        Class khó nhất của V5, cần strict
car         0.45        Class dễ, có thể loose hơn
bus         0.50        Class trung bình
truck       0.50        Class trung bình
```

Lý do khác nhau: V5 detect xe máy yếu hơn xe hơi → nhiều false positive → cần cao để chắc chắn.

**Tiêu chí 2: Size filter**

```
Box có area < 0.0025 (= 400/(512*288)) → loại.

Giải thích: 400 pixel² ≈ box 20×20.
Lý do: V5 yếu với object quá nhỏ → pseudo-label nhỏ không tin được.
```

**Tiêu chí 3: Min boxes per image**

```
Ảnh sau filter có 0 box → loại cả ảnh.
Lý do: ảnh không có xe thì không đóng góp gì cho training.
```

#### 4.5.2. Sơ đồ filter

```
Input: 1 label file .txt với 5 box
┌──────────────────────────────────────┐
│ Box 1: motorcycle, conf 0.72, size L │ ✓ pass (conf > 0.60)
│ Box 2: car,        conf 0.40, size M │ ✗ loại (conf < 0.45)
│ Box 3: truck,      conf 0.62, size L │ ✓ pass
│ Box 4: motorcycle, conf 0.55, size S │ ✗ loại (conf < 0.60)
│ Box 5: car,        conf 0.80, size S │ ✗ loại (size < 400 px²)
└──────────────────────────────────────┘
                 │
                 ▼  giữ lại 2/5 box
┌──────────────────────────────────────┐
│ Box 1: motorcycle                    │
│ Box 3: truck                         │
└──────────────────────────────────────┘
                 │
                 ▼ >= 1 box → giữ ảnh
┌──────────────────────────────────────┐
│ Copy ảnh + label vào accepted/       │
└──────────────────────────────────────┘
```

#### 4.5.3. Output

```
data/pseudo_v6/accepted/
├── images/     ← ảnh gốc được copy sang
│   ├── hcm_phuong-tan-dinh_000001_20261105_113507.jpg
│   └── ...
└── labels/     ← label đã lọc, KHÔNG có confidence nữa (YOLO train format)
    ├── hcm_phuong-tan-dinh_000001_20261105_113507.txt
    └── ...
```

**Format label sau filter** (không có conf):
```
0 0.350 0.540 0.120 0.220
3 0.100 0.600 0.090 0.150
```

**Thống kê điển hình**:
- Trước filter: 11,665 box
- Sau filter: 7,811 box (giữ 67%)
- Ảnh trước: 1,427 → Ảnh sau: 1,401 (giữ 98%)

---

### B6 — Visualize review

**File**: `06_visualize_review.py`
**Mục đích**: Tạo 500 ảnh có bbox vẽ sẵn để người duyệt xem.

#### 4.6.1. Tại sao cần

Label `.txt` là số, con người không đọc được trực tiếp. Phải vẽ lên ảnh để xem:
- Box đúng vị trí không?
- Class đúng không?
- Có miss xe không?

#### 4.6.2. Cơ chế vẽ

```
1. Load ảnh gốc từ accepted/images/
2. Đọc label .txt
3. Với mỗi box:
   - Convert toạ độ normalized → pixel
   - Vẽ rectangle outline màu theo class:
     * motorcycle → ĐỎ
     * car        → XANH LÁ
     * bus        → CAM
     * truck      → XANH DƯƠNG
   - Vẽ label text phía trên box
4. Save vào review/<thứ tự>__<original_name>.jpg
```

#### 4.6.3. Output

```
data/pseudo_v6/review/
├── 0001__hcm_xa-binh-my_000575_20261105_113905.jpg  (ảnh có bbox vẽ)
├── 0002__hcm_phuong-linh-xuan_000224_20261105_112618.jpg
├── ...
├── 0500__hcm_unknown_000028_20261105_111940.jpg
├── bad_samples.txt         ← template để user ghi ảnh SAI
└── RESULT_FORM.md          ← template điền kết quả
```

Số thứ tự đầu tên file (`0001__`) để duyệt tuần tự dễ.

---

### B7 — Remove bad

**File**: `07_remove_bad.py`
**Mục đích**: Xoá ảnh WRONG + camera xấu khỏi dataset.

#### 4.7.1. 2 nguồn "ảnh cần xoá"

**Nguồn 1: `bad_samples.txt` do người ghi**

```
# Người duyệt ghi tên ảnh SAI vào đây:
0042__hcm_phuong-cat-lai_000026_20261105_111241.jpg   # bắt nhầm cây
0133__hcm_unknown_000605_20261105_111832.jpg          # gộp cụm xe
...
```

Script đọc file này → extract tên gốc (bỏ prefix `0042__`) → xoá cặp `.jpg` + `.txt` trong `accepted/`.

**Nguồn 2: Camera blacklist**

```yaml
# configs/hcm_cameras_blacklist.yaml
blacklist:
  '000605':
    name: "Nút giao Mỹ Thủy..."
    reason: "góc cam cao, gộp cụm xe"
    problem_rate: 0.53
  '000430':
    ...
```

Script lấy tất cả ID trong blacklist, tìm ảnh có `_<id>_` trong tên → xoá hết.

#### 4.7.2. Cơ chế

```
Input:
  bad_samples.txt → 11 ảnh cụ thể
  blacklist      → 7 camera (ví dụ 000605 có 32 ảnh)

Combine:
  stems_to_remove = set()
  for name in bad_samples:
      stems_to_remove.add(strip_prefix(name))  # "0042__abc.jpg" → "abc"
  for cam_id in blacklist:
      for img in accepted/images/:
          if f"_{cam_id}_" in img.name:
              stems_to_remove.add(img.stem)

Apply:
  for stem in stems_to_remove:
      delete images/<stem>.jpg
      delete labels/<stem>.txt
```

#### 4.7.3. Output

Dataset `accepted/` sạch sẽ, sẵn sàng merge với V5 gold để train V6.

---

## 5. Cấu trúc thư mục dữ liệu

```
data/
├── raw/
│   ├── frames/
│   │   └── hcm/                                    ← ảnh thô sau collect
│   │       ├── phuong-tan-dinh/
│   │       │   ├── hcm_phuong-tan-dinh_000001_20261105_113507.jpg
│   │       │   └── ...
│   │       ├── phuong-cat-lai/
│   │       └── ...  (~83 ward folder)
│   ├── hcm_cameras.yaml                            ← scrape output (full)
│   └── sampled_cameras_history.txt                 ← nhớ cam đã sample
│
└── pseudo_v6/
    ├── round1/
    │   └── predict/
    │       └── labels/                             ← B4 output: .txt per ảnh
    │           ├── hcm_phuong-tan-dinh_000001_*.txt
    │           └── ...
    │
    ├── accepted/                                   ← B5 output: ảnh + label
    │   ├── images/
    │   └── labels/
    │
    └── review/                                     ← B6 output: ảnh có bbox
        ├── 0001__*.jpg
        ├── ...
        ├── 0500__*.jpg
        ├── bad_samples.txt
        └── RESULT_FORM.md
```

**Phân biệt thư mục quan trọng**:

| Thư mục | Chứa gì | Dùng cho gì |
|---|---|---|
| `data/raw/frames/hcm/` | Ảnh THÔ từ camera | Input cho B3, B4 |
| `data/pseudo_v6/round1/predict/labels/` | Label RAW từ YOLO | Input cho B5 |
| `data/pseudo_v6/accepted/` | Ảnh + label đã FILTER | Input cho B6; **TRAIN V6** |
| `data/pseudo_v6/review/` | Ảnh có BBOX vẽ | Người duyệt tay |

---

## 6. Config files

### `configs/hcm_cameras.yaml`
- 782 URL camera + metadata
- Commit vào git, mọi người dùng chung
- Sinh ra 1 lần bởi B1, hầu như không đổi

### `configs/hcm_cameras_blacklist.yaml`
- List camera xấu (V5 fail systematic)
- Cập nhật DẦN qua các vòng review
- Commit vào git để duy trì khi scrape mới

### `data/raw/sampled_cameras_history.txt`
- List ID camera đã dùng (1 ID/dòng)
- Auto cập nhật mỗi session
- KHÔNG commit (local state)

---

## 7. Troubleshooting

### Lỗi "No module named 'PIL'"
```bash
# Cần dùng conda env yolov8_ft
/home/xuand/miniconda3/envs/yolov8_ft/bin/python3 scripts/pseudo/...
# Hoặc activate env
conda activate yolov8_ft
```

### Lỗi GPU OOM (CUDA out of memory)
```bash
# Giảm chunk size
python scripts/pseudo/04_inference.py --chunk-size 200

# Hoặc dùng CPU (chậm nhưng chắc chắn work)
python scripts/pseudo/04_inference.py --device cpu
```

### Inference bị crash giữa chừng
```bash
# Chạy lại, resume tự động
python scripts/pseudo/04_inference.py
# → sẽ thấy "[resume] đã có X label → bỏ qua"
```

### Server camera không phản hồi (HTTP 403/503)
```
Bị rate limit do fetch quá nhanh. Tăng INTERVAL trong 02_collect_snapshots.py:
  INTERVAL = 20   # thử 30 hoặc 40
```

### Dedup xoá nhầm ảnh khác nhau
```bash
# Giảm threshold (default 3, tăng thành 5 chặt hơn)
python scripts/pseudo/03_dedup.py --threshold 5
```

---

## 📊 Thống kê thực tế (project hiện tại)

| Giai đoạn | Session 1 | Session 2 | Session 3 | Total |
|---|---|---|---|---|
| Cam mới dùng | 30 | 40 | 40 | 110 unique |
| Ảnh thô | 1,588 | 2,559 | 2,919 | 7,066 |
| Ảnh unique (dedup) | ~1,445 | ~2,073 | — | ~6,437 |
| Pseudo-label (infer) | 1,427 | — | — | ... |
| Accepted (filter) | 1,401 | — | — | ... |
| Precision (review 500) | 87.6% | — | — | — |

---

## 🎓 Tóm tắt 1 câu cho mỗi bước

| Bước | Tóm tắt 1 câu |
|---|---|
| B1 Scrape | "Lấy 782 URL ảnh camera từ internet, lưu yaml" |
| B2 Collect | "Tải ảnh từ camera mỗi 20 giây trong 30 phút, bỏ cam đã dùng + cam xấu" |
| B3 Dedup | "Loại ảnh gần giống bằng hash nội dung thị giác" |
| B4 Inference | "YOLO V5 tự vẽ bbox + gán class cho mỗi ảnh" |
| B5 Filter | "Giữ box có confidence cao và kích thước đủ lớn" |
| B6 Visualize | "Vẽ bbox màu lên 500 ảnh random cho người xem" |
| B7 Remove | "Xoá ảnh WRONG do người đánh dấu + ảnh của cam xấu" |

---

*Tài liệu cập nhật: 2026-11-07. Xem code thực tế trong `scripts/pseudo/`.*
