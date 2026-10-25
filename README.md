# VN Traffic AI — Phát hiện, theo vết và đếm phương tiện giao thông Việt Nam

Hệ thống end-to-end: **video giao thông VN → detection → tracking → counting → thống kê + đánh giá**.
Đề tài nhóm, chủ đề 10, dùng chung 1 conda env `yolov8_ft`.

---

## Điểm nổi bật (kết quả mới nhất)

- **2 pipeline so sánh trực tiếp:**
  - **Baseline** — `yolov8s.pt` COCO gốc (80 lớp, filter 4 phương tiện), không huấn luyện.
  - **Improved** — best.pt sau **2 vòng fine-tune** trên dataset Việt Nam.
- **Detection val (14,344 ảnh):** mAP@0.5 = **0.850**, mAP@0.5:0.95 = **0.638**.
  - motorcycle mAP@0.5 = **0.949** (cao nhất 4 class, nhờ boost vnv3).
  - bus 0.911, car 0.848, truck 0.695.
- **Tốc độ:** Improved 72.7 FPS vs Baseline 40.7 FPS (**nhanh hơn 1.79×**).
- **Streamlit dashboard** có 2 chế độ: chạy 1 pipeline hoặc so sánh 2 pipeline side-by-side + video preview H.264 inline.
- **Đánh giá 3 tầng:** Detection (mAP) + Tracking proxy (ID stability) + Counting (MAE/MAPE).

---

## Tech stack

- **Detection:** Ultralytics YOLOv8s (11.2M params, 28.6 GFLOPs, imgsz 512-640)
- **Tracking:** ByteTrack (default) / BoT-SORT — built-in Ultralytics
- **Counting:** line/zone crossing, direction (cross-product), ID-based deduplication
- **Stats & Viz:** pandas + matplotlib
- **UI:** Streamlit dashboard (dual pipeline compare mode)
- **Evaluation:** mAP, Accuracy, MAE, MAPE + tracking proxy (id_stability, overcount_rate)

---

## Cấu trúc thư mục

```
vn_traffic_ai/
├── baseline/
│   └── 10_traffic_counting.py         # code baseline (đã đổi sang v8s cho fair)
├── configs/
│   └── counting_zones.json            # line/zone đếm cho từng video test
├── data/
│   ├── data.yaml                      # schema 4 lớp {motorcycle, car, bus, truck}
│   ├── merged/                        # dataset gộp UA-DETRAC + CanTho + vnv3 (gitignored)
│   └── test_videos/                   # video test + ground truth (gitignored)
├── docs/
│   ├── comparison_report/             # BÁO CÁO SO SÁNH BASELINE vs IMPROVED
│   │   ├── comparison_report.md
│   │   ├── BAO_CAO_BASELINE_VS_IMPROVED.docx
│   │   └── charts/                    # 8 chart PNG dùng cho PPT
│   ├── TEAM_ASSIGNMENT.md
│   └── DATA_GUIDE.md
├── notebooks/
│   └── vn_traffic_ai_full_project.ipynb   # NOTEBOOK end-to-end (setup → train → eval → count)
├── scripts/
│   ├── data_prep/                     # import, merge, prepare dataset
│   ├── evaluation/                    # bench + eval counting/tracking
│   ├── utils/                         # strip_emojis, train.sh (legacy)
│   ├── week2/                         # experiments: freeze, class_weights, v8m
│   └── week3/                         # MLOps: mlflow, docker, onnx (sẽ thêm)
├── src/
│   ├── detection/                     # Người 1 — train.py, evaluate.py, infer.py
│   ├── tracking/                      # Người 2 — track.py, counter.py
│   ├── stats/                         # Người 3 — aggregator.py, visualize.py
│   ├── evaluation/                    # Người 4 — metrics.py (+ tracking proxy)
│   └── pipeline/                      # tích hợp end-to-end: run.py
├── ui/
│   └── streamlit_app.py               # dashboard 2 mode: single / compare
├── results/
│   ├── tables/                        # events.csv, eval JSON, benchmark
│   ├── figures/                       # biểu đồ auto-sinh
│   └── videos/                        # video output (gitignored, mp4v + H.264)
├── weights/
│   └── yolov8s.pt                     # pretrained COCO (baseline)
├── runs/                              # training outputs (gitignored)
│   └── detect/.../train_v8s_ft_vnv3/weights/best.pt  # ← model tốt nhất
└── logs/
```

---

## Cài đặt

### Yêu cầu hệ thống
- **OS:** Ubuntu 20.04+, Windows 10/11 (WSL2), hoặc macOS 12+
- **Python:** 3.10 hoặc 3.11 (khuyến nghị dùng conda)
- **GPU:** NVIDIA GPU với ≥6 GB VRAM (CUDA 11.8+). CPU cũng chạy được nhưng chậm ~30×.
- **Disk:** ~5 GB cho code + weights + dataset nhỏ (chưa tính raw data 137k ảnh)

### Bước 1 — Clone repo
```bash
git clone https://github.com/xuanduc24905-beep/BT_Traffic_Detect.git
cd BT_Traffic_Detect
```

### Bước 2 — Tạo môi trường Python
```bash
# Cách A: dùng conda (khuyến nghị)
conda create -n yolov8_ft python=3.11 -y
conda activate yolov8_ft

# Cách B: dùng venv
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows
```

### Bước 3 — Cài dependencies
```bash
pip install -r requirements.txt

# ffmpeg (bắt buộc cho Streamlit preview video H.264 inline)
sudo apt install ffmpeg -y                     # Ubuntu/WSL/Debian
# brew install ffmpeg                          # macOS
# choco install ffmpeg                         # Windows (dùng Chocolatey)
```

### Bước 4 — Kiểm tra CUDA (nếu có GPU)
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"
```

Kết quả mong đợi: `CUDA: True, GPU: NVIDIA RTX ...`

### Bước 5 — Tải trọng số fine-tune (nếu chưa có)

Repo **không commit weights** (dung lượng lớn). Tải riêng:
- **Baseline `yolov8s.pt`** — Ultralytics tự tải khi chạy lần đầu.
- **Improved `best.pt`** — download từ [Google Drive / release page] hoặc train lại theo hướng dẫn ở mục "Quy trình training" bên dưới.

Đặt vào:
```
weights/yolov8s.pt                                           ← Baseline
runs/detect/runs/detect/train_v8s_ft_vnv3/weights/best.pt    ← Improved
```

---

## Chạy nhanh (Quick Start)

### Cách 1 — Streamlit dashboard (khuyến nghị cho demo)

```bash
streamlit run ui/streamlit_app.py
```

Mở browser: **http://localhost:8501**

Thao tác trong UI:
1. Chọn **Pipeline** ở sidebar (Improved hoặc Baseline).
2. Chọn **Model weights**.
3. Upload video .mp4/.avi/.mov.
4. Chỉnh conf/iou/imgsz + vị trí line đếm.
5. Nhấn **Chạy pipeline** → xem video output + bảng thống kê inline.

Chế độ **⚖ So sánh 2 pipeline**: chạy đồng thời baseline + improved side-by-side.

### Cách 2 — Chạy pipeline từ terminal (batch mode)

```bash
python -m src.pipeline.run \
    --video data/test_videos/raw/demo_traffic.mp4 \
    --weights runs/detect/runs/detect/train_v8s_ft_vnv3/weights/best.pt \
    --tracker bytetrack.yaml \
    --counting-config configs/counting_zones.json \
    --video-key demo_traffic \
    --out-csv results/tables/demo_events.csv \
    --out-video results/videos/demo_out.mp4
```

Output:
- `results/videos/demo_out.mp4` — video có bbox + line + counter overlay
- `results/tables/demo_events.csv` — log từng lượt qua line

### Cách 3 — Chạy baseline gốc (code giáo viên)

```bash
python baseline/10_traffic_counting.py \
    --video data/test_videos/raw/demo_traffic.mp4 \
    --model yolov8s.pt \
    --line-y 474 \
    --out-video baseline/baseline_out.mp4 \
    --no-show
```

### Cách 4 — Sinh báo cáo eval

```bash
# Chạy eval counting 2 pipeline
python scripts/evaluation/eval_counting_2pipelines.py

# Tracking proxy metrics (ID stability)
PYTHONPATH=. python scripts/evaluation/eval_tracking_proxy.py
```

Kết quả nằm ở [docs/comparison_report/](docs/comparison_report/).

---

## Troubleshooting

**Lỗi:** `ModuleNotFoundError: No module named 'src'`
→ Chạy từ **root project**: `cd /path/to/vn_traffic_ai` rồi thêm `PYTHONPATH=.`:
```bash
PYTHONPATH=. python scripts/evaluation/eval_counting_2pipelines.py
```

**Lỗi:** `RuntimeError: CUDA out of memory`
→ Giảm `batch` (32 → 16) hoặc `imgsz` (640 → 512).

**Lỗi:** Streamlit video output trắng, không play
→ Cài `ffmpeg` (xem Bước 3). Streamlit cần H.264 transcode để play inline browser.

**Lỗi:** `FileNotFoundError: runs/detect/.../best.pt`
→ Chưa có weights improved. Chọn model khác trong dropdown, hoặc train theo mục "Quy trình training".

---

## Dataset (schema 4 lớp)

```
0: motorcycle    1: car    2: bus    3: truck
```

**3 nguồn hợp nhất → `data/merged/`:**

| Nguồn | Vai trò | Ảnh |
|---|---|---|
| **UA-DETRAC** (Trung Quốc) | car, bus, truck | ~68k |
| **Vehicle Vietnam-CanTho v19** (Roboflow) | motorcycle chính | 1,235 |
| **Vietnamese vehicle v3** (Roboflow) | boost motorcycle (+2,232 bbox) | 1,547 |
| **Tổng train** | | ~137k ảnh, ~503k bbox |

Val: 14,344 ảnh. Chi tiết download: [docs/DATA_GUIDE.md](docs/DATA_GUIDE.md).

**Class remapping vnv3:** vì Roboflow export với class id khác, script [scripts/data_prep/import_vnv3.py](scripts/data_prep/import_vnv3.py) tự động remap `{0:1, 1:0, 2:3, 3:2}` trước khi merge.

---

## Quy trình training (Improved model)

**Vòng 1 — `train_v8s_4cls`:**
```bash
yolo detect train model=yolov8s.pt data=data/data.yaml \
    epochs=50 batch=32 imgsz=512 lr0=0.01 \
    project=runs/detect name=train_v8s_4cls
```
- Init COCO pretrained, reset head 80→4 lớp.
- Thời gian ~5h 30m trên RTX 3500 Ada 12GB.
- Best mAP50 = 0.848 (epoch 6).

**Vòng 2 — `train_v8s_ft_vnv3` (continual):**
```bash
yolo detect train model=runs/.../train_v8s_4cls/weights/best.pt \
    data=data/data.yaml \
    epochs=25 batch=32 imgsz=512 \
    lr0=0.001 lrf=0.01 patience=8 \
    project=runs/detect name=train_v8s_ft_vnv3
```
- Init từ best.pt vòng 1, thêm 1,547 ảnh vnv3.
- **LR 0.001** (thấp 10×) → tránh catastrophic forgetting.
- Early stop ở epoch 18, tổng 2h 20m.
- Best mAP50 = **0.850** (epoch 10).

---

## Quy trình end-to-end

```bash
# 1. Chạy pipeline single model
python -m src.pipeline.run \
    --video data/test_videos/raw/demo_traffic.mp4 \
    --weights runs/detect/.../train_v8s_ft_vnv3/weights/best.pt \
    --tracker bytetrack.yaml \
    --counting-config configs/counting_zones.json \
    --video-key demo_traffic \
    --out-csv results/tables/demo_events.csv \
    --out-video results/videos/demo_out.mp4

# 2. Benchmark counting: baseline vs improved
python scripts/evaluation/eval_counting_2pipelines.py

# 3. Tracking proxy metrics (ID stability)
PYTHONPATH=. python scripts/evaluation/eval_tracking_proxy.py

# 4. Chạy Streamlit dashboard (2 mode: single / compare)
streamlit run ui/streamlit_app.py
```

---

## Đánh giá — 3 tầng

### Tầng 1: Detection (mAP)

Chạy `yolo val` trên val set 14,344 ảnh:

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| motorcycle | 0.858 | **0.917** | **0.949** ⭐ | 0.609 |
| car | 0.899 | 0.794 | 0.848 | 0.629 |
| bus | 0.917 | 0.819 | 0.911 | 0.772 |
| truck | 0.803 | 0.631 | 0.695 | 0.543 |
| **all** | **0.869** | **0.790** | **0.850** | **0.638** |

### Tầng 2: Counting (task-level)

So sánh trên `demo_traffic.mp4` (5.9s, GT=87 xe):

| Chỉ số | Baseline (COCO) | Improved (fine-tune) |
|---|---|---|
| Total pred | 71 | 104 |
| MAE | 4.0 | 5.25 |
| MAPE (%) | 49.9 | 46.8 |
| Overall accuracy | 81.6% | 80.5% |
| **FPS** | 40.7 | **72.7** (1.79× nhanh) |

### Tầng 3: Tracking proxy (không có GT tracking per-frame)

Metric mới trong [src/evaluation/metrics.py](src/evaluation/metrics.py):
- `track_count` — unique track_id qua line
- `overcount_rate = (pred - gt) / gt`
- `id_stability = 1 - |overcount_rate|` — proxy cho IDF1

| | Track count | Overcount rate | ID stability |
|---|---|---|---|
| Baseline | 71 | **-18.4%** (miss) | 0.816 |
| Improved | 104 | **+19.5%** (ID switch) | 0.805 |

Insight: 2 pipeline có stability tương đương nhưng bị lỗi ở **khía cạnh khác nhau** — baseline miss, improved đếm dư.

**Chưa có IDF1/MOTA/HOTA chuẩn** vì cần GT tracking per-frame (chi phí label quá cao cho scope coursework).

---

## Streamlit dashboard — Hướng dẫn chi tiết

### Khởi chạy

```bash
cd /home/xuand/vn_traffic_ai      # hoặc path tới repo của bạn
conda activate yolov8_ft          # nếu dùng conda
streamlit run ui/streamlit_app.py
```

Mặc định mở tại `http://localhost:8501`. Nếu port bị chiếm, thêm `--server.port 8502`.

> **Yêu cầu tiên quyết:**
> 1. Đã cài `requirements.txt` (đặc biệt `streamlit`, `ultralytics`, `opencv-python`, `pandas`).
> 2. Đã cài `ffmpeg` trong PATH (để transcode H.264 preview video inline browser). Nếu thiếu, video vẫn tải về được nhưng preview trong app có thể lỗi.
> 3. Có ít nhất 1 file `.pt` trong `weights/` hoặc `runs/detect/*/weights/best.pt`. Nếu không có, app sẽ báo lỗi và dừng.

### Sidebar — Bảng cấu hình chung (áp dụng cho mọi tab)

| Nút / Nhóm | Ý nghĩa | Khi nào chỉnh |
|---|---|---|
| **Pipeline** (radio) | Chọn logic pipeline: `Improved (nhóm)` hoặc `Baseline (giáo viên)`. **Chỉ dùng cho chế độ chạy 1 pipeline.** Ở chế độ so sánh, mỗi bên chọn riêng. | Muốn demo pipeline nào |
| **Model weights** (dropdown) | Chọn file `.pt` — quét cả `weights/` và `runs/detect/*/weights/best.pt`. Mặc định ưu tiên `weights/baseline_detrac4.pt` (model tốt nhất, acc 97.7% trên demo). | Đổi giữa COCO pretrained (`yolov8s.pt`) và fine-tune (`best.pt`) |
| **Tracker** (radio) | `bytetrack.yaml` hoặc `botsort.yaml`. **Chỉ dùng cho Improved pipeline** (Baseline luôn dùng ByteTrack mặc định của Ultralytics). | ByteTrack nhanh; BoT-SORT chính xác hơn ở cảnh nhiều xe che nhau |
| **Confidence threshold** (0.1–0.9, mặc định 0.3) | Ngưỡng tin cậy — box có `conf < threshold` sẽ bị lọc. Cao → ít false positive nhưng dễ miss. Thấp → bắt nhiều nhưng có thể có box rác. | Nếu miss xe máy → giảm còn 0.2. Nếu có box rác → tăng lên 0.4 |
| **IoU threshold (NMS)** (0.1–0.9, mặc định 0.5) | Ngưỡng IoU cho Non-Maximum Suppression — 2 box có IoU vượt ngưỡng thì loại box confidence thấp hơn. | Cảnh xe sát nhau → tăng lên 0.6-0.7 (giữ nhiều box). Ngược lại giảm |
| **Image size** (dropdown 320/416/512/640/768, mặc định 640) | Resize ảnh input trước khi đưa vào YOLO. Nhỏ → nhanh nhưng miss xe nhỏ. | Video 1080p có xe nhỏ → 640/768. Test nhanh → 416 |
| **FP16 inference** (checkbox, mặc định bật) | Bật inference dùng half-precision (nhanh 1.3-1.5× trên GPU Ada/Ampere/Turing). | Tắt nếu chạy CPU hoặc GPU cũ (GTX 10xx trở về trước) |
| **Vị trí line (%)** (slider 10–90, mặc định 50) | Vị trí counting line theo % chiều cao (nếu horizontal) hoặc chiều rộng (nếu vertical). | Đặt ngang giữa video (50%) hoặc điều chỉnh theo góc quay |
| **Hướng line** (radio) | `horizontal` (kẻ ngang) hoặc `vertical` (kẻ dọc). | Video quay từ trên xuống → horizontal. Camera bên đường → vertical |
| **Chiều đếm** (radio) | `both` (đếm cả 2 chiều, phân biệt trong output) / `ltr` (chỉ đếm khi cross-product > 0) / `rtl` (< 0). | Ngã tư chỉ muốn đếm 1 chiều → `ltr` hoặc `rtl` |

---

### Tab 1: `▶ Chạy pipeline`

**Bước 1 — Upload video:** kéo thả file `.mp4/.avi/.mov` vào ô upload. App hiển thị preview và thông tin `WxH @ FPS · N frames · duration`.

**Bước 2 — Chọn chế độ:**

#### Chế độ A: `▶ Chạy 1 pipeline` (mặc định)
Dùng cấu hình sidebar → bấm nút `Chạy pipeline`. Có thanh progress theo frame.

Output hiển thị:
- Video annotated (đã transcode H.264, play inline được)
- 2 nút tải: video mp4v gốc và video H.264
- Data tự động chuyển sang **Tab Thống kê** và **Tab Đánh giá**

#### Chế độ B: `⚖ So sánh 2 pipeline (Baseline vs Improved)`

App hiện 2 cột **Side A** và **Side B**, mỗi bên có:
- **Pipeline** — chọn giữa Baseline (1 line, không direction) và Improved (multi-line + direction + log event)
- **Weights** — chọn file `.pt` riêng

**Mặc định thông minh**: Side A gợi ý `yolov8s.pt` (COCO gốc), Side B gợi ý `best.pt` fine-tune (tìm `ft_vnv3` trước, không có thì `4cls`).

Bấm `⚖ Chạy so sánh 2 pipeline` → app chạy tuần tự 2 side, mỗi bên có progress bar riêng.

Kết quả side-by-side:
- 2 video output cạnh nhau
- Bảng đếm per-class từng side + metric: Tổng đếm / Runtime / FPS
- Bảng so sánh chỉ số chính (Tổng đếm, Runtime, FPS) + chênh lệch + speedup
- Bar chart per-class dạng grouped bar (2 màu Baseline vs Improved)
- Info box tự đọc kết quả bằng chữ

Tab Thống kê và Tab Đánh giá sẽ có thêm radio **"Xem thống kê của pipeline nào?"** để chọn nguồn dữ liệu.

---

### Tab 2: `Thống kê`

Chỉ hoạt động sau khi chạy pipeline ở Tab 1.

**Nút chọn nguồn (chỉ có trong compare mode):** radio `Improved (fine-tune)` / `Baseline (COCO)` — quyết định thống kê nào đang xem.

**Chia thời gian theo (dropdown):** `30 giây / 1 phút / 5 phút / 15 phút / 30 phút / 1 giờ` — quyết định bucket cho chart theo thời gian và peak period.

**Các mục hiển thị:**

| Mục | Nội dung |
|---|---|
| Metrics tổng | Tổng lượt, số loại xe, loại nhiều nhất, thời lượng video |
| Tốc độ lưu lượng | `Tổng xe/phút` và `Tổng xe/giờ` + bảng chi tiết per class |
| Khung giờ cao điểm | Bucket đông nhất + số lượt trong bucket đó (thay đổi theo lựa chọn "Chia thời gian") |
| Biểu đồ theo bucket | Line chart hoặc Bar chart chồng (toggle radio) |
| Heatmap | Bảng có gradient màu, hàng = bucket, cột = class |
| Đếm tích luỹ | Line chart tăng dần theo thời gian |
| Số lượng theo loại xe | Bar chart tổng |
| Lưu lượng theo chiều đi | Bảng + bar chart `line × direction` (chỉ Improved) |
| So sánh giữa các line | Chỉ hiển thị nếu có ≥2 line |
| Event log | Bảng chi tiết mọi lượt đếm |
| Nút tải xuống | Events CSV, Summary CSV, Bucket CSV |

---

### Tab 3: `Đánh giá`

Đo độ chính xác của counting so với ground truth (đếm tay).

**Bước 1 — Nhập GT bằng 1 trong 2 cách:**

- **Cách A: Upload JSON.** File có cấu trúc:
  ```json
  { "counts_by_class": { "motorcycle": 45, "car": 30, "bus": 3, "truck": 2 } }
  ```
- **Cách B: Điền tay.** Các ô `number_input` cho từng class (motorcycle, car, bus, truck, van, others, bicycle). Có hiển thị số model dự đoán bên cạnh để dễ so sánh.

> ⚠️ **Lưu ý:** phải nhập số xe **thật** bạn đếm được từ video, không phải copy số model dự đoán. Để mặc định 0 sẽ không đánh giá đúng.

**Bước 2 — Xem kết quả (tự động khi có GT):**
- Bảng per-class: `Pred / GT / |Err| / Accuracy`
- 3 metric: Overall Accuracy, MAE, MAPE (%)
- Nút tải JSON eval report

---

### Workflow ví dụ (demo giáo viên)

1. Chọn `Improved (nhóm)` trong Pipeline (sidebar)
2. Chọn weights `runs/.../train_v8s_ft_vnv3/weights/best.pt`
3. Giữ mặc định: `conf=0.3, iou=0.5, imgsz=640, FP16 on`
4. Chỉnh line: `Vị trí=50%, Hướng=horizontal, Chiều đếm=both`
5. Tab 1 → upload `demo_traffic.mp4` → chọn `⚖ So sánh 2 pipeline`
6. Side A giữ `yolov8s.pt`, Side B giữ `best.pt` fine-tune → bấm chạy
7. Xem bảng so sánh + bar chart per-class ở cuối tab
8. Chuyển Tab 2 → xem flow rate, peak hour (thử đổi "Chia thời gian" giữa 1 phút và 5 phút)
9. Chuyển Tab 3 → điền GT (từ video đếm tay) → xem MAE/MAPE

---

## File deliverable chính

| Loại | Đường dẫn |
|---|---|
| Báo cáo Markdown | [docs/comparison_report/comparison_report.md](docs/comparison_report/comparison_report.md) |
| Báo cáo Word | [docs/comparison_report/BAO_CAO_BASELINE_VS_IMPROVED.docx](docs/comparison_report/BAO_CAO_BASELINE_VS_IMPROVED.docx) |
| Notebook end-to-end | [notebooks/vn_traffic_ai_full_project.ipynb](notebooks/vn_traffic_ai_full_project.ipynb) |
| 8 chart PPT | [docs/comparison_report/charts/](docs/comparison_report/charts/) |
| Best weights | `runs/detect/.../train_v8s_ft_vnv3/weights/best.pt` |

---

## Phân công 4 người

| Người | Module | Deliverable |
|---|---|---|
| 1 | `src/detection/` | Train YOLOv8, xuất `best.pt`, đánh giá mAP |
| 2 | `src/tracking/` | ByteTrack wrapper + line/zone counter |
| 3 | `src/stats/` | Aggregate events CSV + biểu đồ per class × thời gian |
| 4 | `src/evaluation/` | Ground truth thủ công + Accuracy/MAE/MAPE + tracking proxy + pipeline integration |

Chi tiết: [docs/TEAM_ASSIGNMENT.md](docs/TEAM_ASSIGNMENT.md).

---

## Hạn chế đã biết & hướng phát triển

**Hạn chế:**
- Motorcycle chỉ 0.85% bbox train → mAP@0.5:0.95 = 0.609 (thấp hơn class khác).
- Truck confusion cao (mAP 0.695) do UA-DETRAC gộp `van → truck`.
- Improved model kém generalize ra video out-of-distribution (recall giảm mạnh khi test video screen recording độ phân giải thấp).
- Chưa đo IDF1/MOTA chuẩn do thiếu GT tracking per-frame.

**Hướng mở rộng:**
- Fine-tune với `freeze=10` (chỉ train head, giữ backbone COCO) → tăng generalization.
- Class-balanced sampling / focal loss.
- Test đa dạng video VN (5-10 phút mỗi video).
- Deploy edge (Jetson) test real-time.

---

## Bài học chính

> "**Same model, better data → better result**" — cùng kiến trúc YOLOv8s (không đổi params), chỉ thay data + training strategy → mAP xe máy tăng lên 0.949.

Nhưng cần cẩn thận: fine-tune sâu trên dataset hẹp có thể làm **mất generalization** khi test out-of-distribution. Đây là trade-off điển hình của specialist vs generalist model.
