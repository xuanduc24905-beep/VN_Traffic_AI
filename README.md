# VN Traffic AI — Phát hiện, theo vết và đếm phương tiện giao thông Việt Nam

Hệ thống end-to-end: **video giao thông VN → detection → tracking → counting → thống kê + đánh giá**.
Đề tài coursework (chủ đề 10) với **methodology research-honest**: video-aware split, phát hiện và sửa data leak, so sánh thẳng thắn các iteration (gồm cả negative result).

Repo: https://github.com/xuanduc24905-beep/VN_Traffic_AI

> ⚠️ **Số liệu dưới đây là con số có điều kiện, không phải "tốt tuyệt đối".** Xem [§ Giới hạn của bộ eval hiện tại](#-giới-hạn-của-bộ-eval-hiện-tại-đọc-trước-khi-tin-số) trước khi trích dẫn mAP50 = 0.782.

---

## Câu chuyện dự án (tóm tắt)

Dự án đi qua **7 iteration (V1 → V7)** với một bước ngoặt methodology:

| Giai đoạn | Iteration | mAP50 | Ghi chú |
|---|---|---|---|
| **Trước phát hiện leak** | V1–V4 | 0.946–0.954 ❌ | Random split → 100% video-level leak (train/val/test cùng video) |
| **Sau rebuild honest split** | V5 | **0.665** ✓ | Video-aware split, 0% leak — con số thật |
| | V5+ (imgsz=960) | **0.782** ⚠️ | "Winner" trong-domain SOICT; xem caveat bên dưới |
| | V6 (pseudo-label vn_cctv) | 0.761 | **Negative result** — pseudo amplify class bias |
| | V7 (DETRAC bustruck + imgsz=1280) | 0.756 | Motorcycle best (0.872) nhưng car/truck giảm |

> **Insight chính:** gap 0.289 mAP50 giữa V3 leaky (0.954) và V5 honest (0.665) chính xác là "leak inflation" — model memorize scene/background, không generalize. Report đầy đủ: [docs/report/BAO_CAO_PHAN_TICH_DU_AN.md](docs/report/BAO_CAO_PHAN_TICH_DU_AN.md).

---

## Kết quả cross-evaluation (4 model × 3 test set)

| Model | V5 test (VN urban, honest) | vn_cctv OOD (Đà Nẵng) | DETRAC test (cross-domain) |
|---|---|---|---|
| COCO baseline (yolov8s zero-shot) | 0.557 | — | — |
| V5 honest (imgsz=640) | 0.665 | 0.181 | 0.548 |
| **V5+ (imgsz=960)** | **0.782** ⭐ | 0.215 | 0.637 |
| V6 (pseudo vn_cctv) | 0.761 | — | — |
| V7 (DETRAC + imgsz=1280) | 0.756 | 0.187 | 0.523 |

Chi tiết: [results/v5_cross_eval/FINAL_COMPARISON.md](results/v5_cross_eval/FINAL_COMPARISON.md) · [FINAL_COMPARISON.json](results/v5_cross_eval/FINAL_COMPARISON.json).

> **Lưu ý khi đọc bảng:** "V5 test" là 2 video SOICT cam 09+10 — **cùng dataset** với train. Con số 0.782 phản ánh "tốt trong-domain SOICT24", không nên hiểu là "tốt trên giao thông VN nói chung". Số OOD 0.215 (vn_cctv) chỉ từ **1 video Đà Nẵng** → variance cao, n=1. Xem [§ Giới hạn của bộ eval hiện tại](#-giới-hạn-của-bộ-eval-hiện-tại-đọc-trước-khi-tin-số).

**Per-class (V5+ trên V5 test):**

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| motorcycle | 0.78 | 0.80 | 0.841 | 0.47 |
| car | 0.88 | 0.82 | **0.884** | 0.56 |
| bus | 0.77 | 0.68 | 0.716 | 0.52 |
| truck | 0.74 | 0.63 | 0.687 | 0.49 |
| **all** | 0.79 | 0.73 | **0.782** | 0.51 |

---

## ⚠️ Giới hạn của bộ eval hiện tại (đọc trước khi tin số)

Sau khi sửa video-leak ở V5, bộ eval **vẫn còn mấy loại bias chưa xử lý**. Thừa nhận rõ để người đọc hiểu con số 0.782 có điều kiện gì:

| Vấn đề | Mô tả | Hệ quả |
|---|---|---|
| **Same-source bias** | V5 train/val/test đều từ SOICT24 (cam 01-06+08 train, 07 val, 09-10 test). Cùng project ghi hình → cùng camera, encoding, góc, style label. | Cam 09 "unseen" nhưng nhìn vẫn giống cam 01. mAP 0.782 chủ yếu đo "nội suy trong SOICT", không phải generalize ra giao thông VN nói chung. |
| **Same-labeler bias** | SOICT24 label bởi 1 nhóm với convention cố định (ngưỡng che khuất, có box kính không, v.v.). | Model học style label chứ không chỉ học "xe". |
| **Test set tuning (p-hacking gián tiếp)** | V5 → V5+ → V6 → V7 đều eval trên **cùng 1 test set**, chọn "V5+ winner" sau khi thử 4 config. | Information từ test set đã leak vào quyết định về model → mAP báo cáo là optimistic. |
| **OOD n=1** | `vn_cctv` 29k frames nhưng **chỉ 1 video Đà Nẵng**. | Số OOD 0.215 variance rất cao, không có statistical power. |
| **Author tự chọn split** | Người train model cũng chọn cam nào vào test → unconscious bias (có thể vô tình chọn video "trông quen"). | Không có blinded split. |
| **Không có confidence interval** | mAP báo cáo là 1 số đơn, không có std / bootstrap CI. | Không biết gap V5+ (0.782) vs V7 (0.756) có significant không, hay chỉ là noise. |

### Nếu muốn eval "công tâm thật sự" (recipe)

Để nâng cấp lên eval chuẩn research, cần:

1. **Blinded split by someone ngoài team train model**. Người đó chia 70/15/15 theo frame count, **stratify theo nguồn** (SOICT day/night/Cantho/VNv3) và **theo class** (≥N bbox mỗi class trong test). Commit `split_manifest.json` + `git tag` trước khi training.
2. **Lock test set — chỉ chạm 1 lần.** Val tune bao nhiêu lần cũng được; test chỉ eval sau khi đã chốt model cuối cùng. Test set hiện tại đã bị eval 4 lần → **coi như "cháy"**, phải sinh set mới nếu muốn con số sạch.
3. **≥2 held-out test sources, không 1.**
   - `test_internal` (SOICT unseen videos — như hiện tại)
   - `test_external` — **nguồn hoàn toàn khác** (ví dụ YouTube VN tự quay 5-10 phút, hoặc dataset VN khác)
4. **Multi-seed train.** V5+ train 3 lần với seed khác nhau → báo cáo mean ± std. Nếu std > gap V5+ vs V7 (0.026) thì "V5+ thắng V7" là noise, không phải signal.
5. **Bootstrap CI** cho mAP (resample test set 1000 lần) → báo cáo `0.78 ± 0.03` thay vì `0.782`.
6. **Re-label nhỏ** — đưa 100-200 ảnh test cho teammate không train model label lại, đo IoU inter-annotator. Nếu <0.7 thì "label gốc nhiễu ~0.3 mAP" là trần đo được.

**Thực tế cho scope coursework:** chưa làm hết được. **Bắt buộc** nên thừa nhận điểm 1-2 trong báo cáo và pre-register trước iteration tiếp theo.

---

## Methodology — Honest split (quan trọng nhất)

### Phát hiện data leak (V1–V4)
Script [scripts/data_prep/audit_leak.py](scripts/data_prep/audit_leak.py) phát hiện **12/15 video (80%) rò rỉ sang cả 3 split**:
- SOICT cam 01: train=950 frames, val=276, test=126 — **cùng 1 video**
- VNv3: 1,235 train + 156 val + 156 test — **cùng 1 video chia 3 phần**
- Tổng: 1,779 val frames + 1,053 test frames ≡ 100% từ video đã có trong train

→ Model học "nhớ scene" chứ không học "nhận xe".

### Rebuild V5 — Video-aware split
[scripts/data_prep/rebuild_v5_by_video.py](scripts/data_prep/rebuild_v5_by_video.py) chia theo `video_id`:
- **Train:** 11 videos (10 SOICT/Cantho + 1 VNv3) → 2,833 frames
- **Val:** 3 videos (SOICT cam 07, Cantho 2) → 1,466 frames (video chưa thấy)
- **Test:** 2 videos (SOICT cam 09, 10) → 1,500 frames (video chưa thấy)
- Verify: `audit_leak.py` fail hard nếu còn overlap. Manifest reproducible ở [data/merged_v5/split_manifest.json](data/merged_v5/split_manifest.json).

### Safety gate (V7)
V7 pipeline tích hợp **leak audit** thành Phase 4b.5 — training sẽ fail nếu detect leak. Kết quả audit lưu ở [docs/v3_report/leak_audit_v7.json](docs/v3_report/leak_audit_v7.json).

---

## Tech stack

- **Detection:** Ultralytics YOLOv8s (11.2M params, 28.6 GFLOPs) — imgsz 640 / 960 / 1280 tùy iteration
- **Tracking:** ByteTrack (default) / BoT-SORT — built-in Ultralytics
- **Counting:** line/zone crossing + direction (cross-product) + ID-based deduplication
- **Stats & Viz:** pandas + matplotlib
- **UI:** Streamlit dashboard (single pipeline / compare 2 pipelines side-by-side)
- **Evaluation:** mAP + Accuracy/MAE/MAPE + tracking proxy (id_stability, overcount_rate)

---

## Cấu trúc thư mục

```
vn_traffic_ai/
├── baseline/                              # Code baseline giáo viên
├── configs/counting_zones.json            # Line/zone đếm per video
├── data/
│   ├── data.yaml                          # Schema 4 lớp (gốc)
│   ├── data_v7.yaml                       # Schema V7 (có DETRAC bustruck)
│   ├── merged_v5/                         # V5 honest split (symlinks)
│   ├── merged_v7/                         # V7 split (V5 + DETRAC bustruck)
│   ├── detrac_bustruck_subset/            # DETRAC bus/van only cho V7
│   ├── pseudo_vn_cctv/                    # V5+ predictions trên vn_cctv (dùng cho V6)
│   └── raw_*/                             # Dataset gốc (gitignored)
├── docs/
│   ├── report/
│   │   ├── BAO_CAO_PHAN_TICH_DU_AN.md     # Báo cáo 10 chương
│   │   ├── BAO_CAO_PHAN_TICH_DU_AN.docx
│   │   └── assets/{diagrams,yolo_format}/ # Sơ đồ pipeline + YOLO format
│   ├── v3_report/                         # Hồ sơ phát hiện leak + audit
│   ├── comparison_report/                 # Báo cáo cũ (baseline vs improved)
│   └── training_report/
├── notebooks/
│   ├── error_analysis.ipynb               # V5+ error breakdown (6 chiều)
│   ├── final_report.ipynb                 # Tổng hợp kết quả
│   └── full_project.ipynb                 # End-to-end demo
├── scripts/
│   ├── data_prep/                         # import, merge, audit_leak, rebuild_v5
│   ├── training/
│   │   ├── v5_honest.py                   # V5 baseline
│   │   ├── v5_plus_960.py                 # V5+ (imgsz=960) — WINNER
│   │   └── v6_pseudo_lbl.py               # V6 (pseudo-label — failed)
│   ├── phase4_v7/                         # V7 pipeline (4a import → 4b build → audit → 4c train → 4d eval)
│   │   └── run_pipeline.sh
│   ├── eval/eval_all_models_full.py       # Cross-eval 4 model × 3 test set
│   ├── report/                            # Render figures cho báo cáo
│   ├── demo/                              # Demo scripts
│   └── utils/
├── src/
│   ├── detection/                         # train/evaluate/infer
│   ├── tracking/                          # ByteTrack wrapper + counter
│   ├── stats/                             # aggregator + visualize
│   ├── evaluation/                        # metrics (mAP + tracking proxy)
│   └── pipeline/run.py                    # End-to-end CLI
├── ui/streamlit_app.py                    # Dashboard (single / compare mode)
├── results/
│   ├── v5_cross_eval/                     # Bảng so sánh 4 model × 3 test set
│   ├── error_analysis_v5plus/             # Predictions pkl cho analysis
│   ├── tables/                            # Events/eval CSV + JSON
│   └── figures/
├── weights/yolov8s.pt                     # COCO pretrained baseline
├── runs/detect/                           # Training outputs (gitignored)
│   ├── train_v8s_v5_honest/weights/best.pt
│   ├── train_v8s_v5plus_960/weights/best.pt  # ⭐ WINNER
│   ├── train_v8s_v6_pseudo/weights/best.pt
│   └── train_v8s_v7_1280/weights/best.pt
├── tests/
├── requirements.txt
└── run_all.sh
```

---

## Cài đặt

### Yêu cầu
- Ubuntu 20.04+ / Windows 10/11 WSL2 / macOS 12+
- Python 3.10 hoặc 3.11
- NVIDIA GPU ≥6 GB VRAM (CUDA 11.8+); CPU chạy được nhưng chậm ~30×
- Disk ~10 GB (code + weights + dataset nhỏ, chưa tính raw)

### Setup
```bash
git clone https://github.com/xuanduc24905-beep/VN_Traffic_AI.git
cd VN_Traffic_AI

# Conda (khuyến nghị)
conda create -n yolov8_ft python=3.11 -y
conda activate yolov8_ft
pip install -r requirements.txt

# ffmpeg (bắt buộc cho Streamlit preview H.264 inline)
sudo apt install ffmpeg -y          # Ubuntu/WSL
# brew install ffmpeg               # macOS
```

### Kiểm tra CUDA
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Weights
Repo **không commit `.pt` file lớn**. Hai cách:
- **Baseline `yolov8s.pt`** — Ultralytics tự tải khi chạy lần đầu.
- **Best `best.pt` (V5+)** — train lại theo mục "Training" bên dưới (~2h trên RTX 3500 Ada 12GB) hoặc tải từ release.

Đặt vào:
```
weights/yolov8s.pt                                      # Baseline
runs/detect/train_v8s_v5plus_960/weights/best.pt        # Winner (V5+ 960)
runs/detect/train_v8s_v7_1280/weights/best.pt           # V7 alt (motorcycle focus)
```

---

## Quick Start

### Cách 1 — Streamlit dashboard (demo)
```bash
streamlit run ui/streamlit_app.py
```
Mở http://localhost:8501 → upload video → chọn weights → chạy. Chi tiết tab/sidebar ở mục "Streamlit dashboard" bên dưới.

### Cách 2 — Pipeline từ terminal
```bash
python -m src.pipeline.run \
    --video data/test_videos/raw/demo_traffic.mp4 \
    --weights runs/detect/train_v8s_v5plus_960/weights/best.pt \
    --tracker bytetrack.yaml \
    --counting-config configs/counting_zones.json \
    --video-key demo_traffic \
    --out-csv results/tables/demo_events.csv \
    --out-video results/videos/demo_out.mp4
```

### Cách 3 — Cross-eval 4 model × 3 test set
```bash
python scripts/eval/eval_all_models_full.py
```
Output: `results/v5_cross_eval/FINAL_COMPARISON.{md,json}` + per-model folder.

### Cách 4 — Baseline gốc (code giáo viên)
```bash
python baseline/10_traffic_counting.py \
    --video data/test_videos/raw/demo_traffic.mp4 \
    --model yolov8s.pt --line-y 474 --no-show
```

---

## Dataset

**Schema 4 lớp:** `0: motorcycle · 1: car · 2: bus · 3: truck`

**Nguồn (sau khi rebuild V5 honest):**

| Nguồn | Vai trò | Train frames | Val/Test |
|---|---|---|---|
| SOICT24 (10 cam day + 3 cam night) | Chủ lực VN urban | ~2,100 | cam 07, 09, 10 → 2,966 frames unseen |
| Vehicle Vietnam-CanTho v19 (Roboflow) | motorcycle VN | 678 | — |
| Vietnamese vehicle v3 (Roboflow) | boost motorcycle | 1,547 | — |
| **DETRAC bus/truck subset** (V7 only) | bus/truck cross-domain | +550 | — |
| **Tổng V5 train** | | **2,833** | **val 1,466 / test 1,500** |

**OOD test set:** vn_cctv (Đà Nẵng, 29k frames, 1 video) — hoàn toàn ngoài training domain.

Chi tiết: [docs/report/BAO_CAO_PHAN_TICH_DU_AN.md §3-4](docs/report/BAO_CAO_PHAN_TICH_DU_AN.md).

---

## Training

### V5 honest (baseline — imgsz=640)
```bash
python scripts/training/v5_honest.py
```
Config: yolov8s fresh COCO, imgsz=640, batch=16, 40 epochs. Output: `runs/detect/train_v8s_v5_honest/`.

### V5+ (winner — imgsz=960)
```bash
python scripts/training/v5_plus_960.py
```
Config: yolov8s COCO, imgsz=960, batch=16, cosine LR, 40 epochs (early stop ~26ep). Output: `runs/detect/train_v8s_v5plus_960/`.

### V6 pseudo-label (negative result, để tham khảo)
```bash
# Phải chạy V5+ trước để sinh pseudo-label
python scripts/training/v6_pseudo_lbl.py
```
Kết quả: pseudo-label trên vn_cctv (conf>0.75) nhưng amplify class bias (car 80%, thiếu bus/truck) → V5 test giảm 2.7%.

### V7 DETRAC bustruck (imgsz=1280)
```bash
bash scripts/phase4_v7/run_pipeline.sh
```
Chạy tuần tự: 4a import DETRAC → 4b filter bustruck → **4b.5 audit leak (fail hard)** → 4c train imgsz=1280 → 4d eval.

---

## Đánh giá — 3 tầng

### Tầng 1: Detection (mAP) — Honest cross-eval
Bảng cross-eval 4 model × 3 test set đã trình bày ở đầu README. Xem [results/v5_cross_eval/FINAL_COMPARISON.md](results/v5_cross_eval/FINAL_COMPARISON.md).

### Tầng 2: Counting (task-level)
MAE / MAPE / Overall accuracy trên video thật, so sánh baseline vs best model.

### Tầng 3: Tracking proxy
Vì không có GT tracking per-frame (label chi phí cao cho scope coursework), dùng proxy metric trong [src/evaluation/metrics.py](src/evaluation/metrics.py):
- `track_count` — unique track_id qua line
- `overcount_rate = (pred - gt) / gt`
- `id_stability = 1 - |overcount_rate|` — proxy IDF1

→ Phân biệt được pipeline **miss xe** (recall thấp) vs **ID switch** (overcount) — 2 loại lỗi khác nhau.

---

## Error analysis V5+

Notebook: [notebooks/error_analysis.ipynb](notebooks/error_analysis.ipynb) — breakdown 6 chiều: class × size × source.

**Insights chính:**
- **FN motorcycle small objects** chiếm 61% false negative — xe máy xa camera, bbox <32×32.
- **FP ở scene dày đặc** — nhiều xe chồng lắp, NMS gộp nhầm.
- **OOD gap lớn:** V5+ 0.782 trên VN urban → 0.215 trên vn_cctv (Đà Nẵng góc camera khác, chất lượng CCTV thấp hơn).

---

## Streamlit dashboard — hướng dẫn chi tiết

### Khởi chạy
```bash
conda activate yolov8_ft
streamlit run ui/streamlit_app.py
```
Mặc định `http://localhost:8501`. Port bị chiếm thì `--server.port 8502`.

**Yêu cầu tiên quyết:**
1. Đã `pip install -r requirements.txt` (streamlit, ultralytics, opencv-python, pandas).
2. Đã cài `ffmpeg` (transcode H.264 inline).
3. Có ≥1 file `.pt` ở `weights/` hoặc `runs/detect/*/weights/best.pt`.

### Sidebar — cấu hình chung

| Nhóm | Ý nghĩa |
|---|---|
| **Pipeline** (radio) | `Improved (nhóm)` hoặc `Baseline (giáo viên)`. Compare mode mỗi bên chọn riêng. |
| **Model weights** (dropdown) | Quét `weights/` + `runs/detect/*/weights/best.pt`. Mặc định ưu tiên V5+. |
| **Tracker** (radio) | `bytetrack.yaml` (nhanh) / `botsort.yaml` (chính xác hơn khi che). |
| **Confidence** (0.1–0.9, mặc định 0.3) | Thấp → bắt nhiều, có box rác. Cao → ít FP, dễ miss. |
| **IoU NMS** (0.1–0.9, mặc định 0.5) | Scene xe sát → tăng 0.6-0.7. |
| **Image size** (320/416/512/640/768/960) | Nhỏ → nhanh; lớn → bắt xe nhỏ (V5+ dùng 960). |
| **FP16** | Nhanh 1.3-1.5× trên Ada/Ampere/Turing. Tắt nếu CPU/GPU cũ. |
| **Line position / direction / mode** | % vị trí line + ngang/dọc + both/ltr/rtl. |

### Tab 1 — Chạy pipeline
Upload `.mp4/.avi/.mov` → chọn chế độ:
- **A: Chạy 1 pipeline** — dùng config sidebar, progress bar theo frame.
- **B: So sánh 2 pipeline** — side A/B mỗi bên chọn weights riêng, chạy tuần tự. Mặc định: A=`yolov8s.pt` (COCO), B=`best.pt` V5+/V7. Output: 2 video side-by-side + bảng chênh lệch + bar chart per-class + speedup.

### Tab 2 — Thống kê
Chia bucket thời gian (30s/1p/5p/15p/30p/1h) → metrics tổng + flow rate (xe/phút, xe/giờ) + peak period + heatmap bucket×class + cumulative line chart + per-class bar + line×direction (chỉ Improved) + event log + download CSV.

### Tab 3 — Đánh giá
Nhập GT counting bằng 2 cách:
- **Upload JSON:** `{"counts_by_class": {"motorcycle": 45, "car": 30, ...}}`
- **Điền tay:** number_input per class (có hiển thị pred bên cạnh)

Output: bảng per-class (Pred/GT/|Err|/Acc) + Overall Accuracy / MAE / MAPE + download eval JSON.

---

## Troubleshooting

| Lỗi | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Chạy từ root + `PYTHONPATH=.` |
| `CUDA out of memory` | Giảm `batch` (16→8) hoặc `imgsz` (960→640) |
| Streamlit video output trắng | Cài `ffmpeg` |
| `FileNotFoundError: .../best.pt` | Train lại hoặc chọn model khác trong dropdown |
| V7 train fail "leak detected" | Audit script phát hiện split overlap — fix split trước khi train (feature intended, không phải bug) |

---

## Deliverable chính

| Loại | Đường dẫn |
|---|---|
| Báo cáo đầy đủ (10 chương) | [docs/report/BAO_CAO_PHAN_TICH_DU_AN.md](docs/report/BAO_CAO_PHAN_TICH_DU_AN.md) · [.docx](docs/report/BAO_CAO_PHAN_TICH_DU_AN.docx) |
| Cross-eval 4 model × 3 test | [results/v5_cross_eval/FINAL_COMPARISON.md](results/v5_cross_eval/FINAL_COMPARISON.md) |
| Hồ sơ phát hiện leak | [docs/v3_report/DATA_LEAK_FINDINGS.md](docs/v3_report/DATA_LEAK_FINDINGS.md) · [leak_audit_summary.md](docs/v3_report/leak_audit_summary.md) |
| Notebook error analysis | [notebooks/error_analysis.ipynb](notebooks/error_analysis.ipynb) |
| Best weights | `runs/detect/train_v8s_v5plus_960/weights/best.pt` |
| Pipeline V7 (self-contained) | [scripts/phase4_v7/](scripts/phase4_v7/) |

---

## Phân công 4 người

| Người | Module | Deliverable |
|---|---|---|
| 1 | `src/detection/` + `scripts/training/` | Train V5/V5+/V6/V7, mAP, cross-eval |
| 2 | `src/tracking/` | ByteTrack/BoT-SORT wrapper + line/zone counter |
| 3 | `src/stats/` | Aggregate CSV + biểu đồ per class × thời gian |
| 4 | `src/evaluation/` + `scripts/data_prep/` | GT counting + MAE/MAPE + **leak audit** + tracking proxy |

Chi tiết: [docs/TEAM_ASSIGNMENT.md](docs/TEAM_ASSIGNMENT.md).

---

## Hạn chế đã biết

**Về eval methodology** (xem chi tiết ở [§ Giới hạn của bộ eval hiện tại](#-giới-hạn-của-bộ-eval-hiện-tại-đọc-trước-khi-tin-số)):
- Same-source bias (SOICT train/test), test set đã bị tuning gián tiếp 4 lần, OOD n=1, không có blinded split, không có confidence interval, không multi-seed.

**Về model performance:**
- **OOD generalization yếu:** V5+ 0.782 (VN urban SOICT) → 0.215 (vn_cctv Đà Nẵng). Mô hình fine-tune hẹp chưa đủ cover domain khác.
- **Motorcycle small objects:** 61% FN đến từ xe máy xa camera (<32×32 px) — cần imgsz lớn hơn hoặc dataset ảnh HD gần hơn.
- **Truck/van confusion:** UA-DETRAC gộp van vào truck → mô hình V7 regress trên class truck (-14% so với V5+).
- **Chưa có IDF1/MOTA chuẩn:** cần GT tracking per-frame, chi phí label quá cao cho scope coursework. Dùng tracking proxy thay thế.
- **Pseudo-label naive thất bại (V6):** cần cân bằng class trước khi self-training, hoặc dùng confidence-weighted sampling.

---

## Bài học chính

> **Honest methodology > inflated numbers.** V3 leaky 0.954 nhìn đẹp hơn V5 honest 0.665, nhưng V5 là con số thật — V3 chỉ là model memorize scene. Nếu không rebuild split, mọi so sánh sau đó đều vô nghĩa.

> **Fix 1 bias không có nghĩa là hết bias.** V5 sửa video-leak nhưng vẫn còn same-source, test-tuning, OOD n=1. Thừa nhận trong README còn hơn giấu đi — người đọc có quyền biết con số 0.782 có điều kiện gì (xem [§ Giới hạn của bộ eval hiện tại](#-giới-hạn-của-bộ-eval-hiện-tại-đọc-trước-khi-tin-số)).

> **Resolution > data volume (trong scope này).** V5 → V5+ chỉ đổi imgsz từ 640 → 960, mAP50 nhảy từ 0.665 → 0.782 (+17.6%). Thêm data (V6 pseudo-label) và cross-domain (V7 DETRAC) đều không bằng.

> **Negative result cũng là result.** V6 fail cho insight quan trọng: pseudo-label naive amplify class bias — document lại để lần sau không lặp lỗi.
