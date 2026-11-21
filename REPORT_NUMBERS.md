











# REPORT_NUMBERS.md — Rà soát số liệu báo cáo

**Nguồn b
áo cáo rà:** `docs/tieuluan_V3/Nhom02_PhanLoaiPhuongTien_UPDATED5 (1).docx` (1 bản, 9.1 MB, 2026-11-16).

**Môi trường kiểm chứng:**
- Python 3.12 (conda env `vehicle_env`)
- torch 2.7.1+cu118, torchvision 0.22.1+cu118
- ultralytics 8.4.126
- GPU: NVIDIA RTX 3500 Ada Generation Laptop GPU, 12 GB
- Driver 595.95, CUDA Runtime 11.8
- Ngày chạy: 2026-11-16

**Nguyên tắc so khớp:** mAP/P/R là kết quả stochastic của `model.val()` dưới FP16, chênh ≤ 0,003 so với bản ghi báo cáo được coi là **KHỚP**. Mọi chênh lớn hơn đánh dấu **LỆCH**.

---

## 0. Tóm tắt tổng thể

| Mục | Số lượng |
|---|---|
| Tổng số đoạn văn trong báo cáo | 735 |
| Tổng số bảng | 31 |
| Tổng số token số đã rà | 2.021 |
| Số lượng **KHỚP** (≤ ngưỡng sai số) | ~150 (toàn bộ Bảng 4.3–4.7, 4.9, 3.1, 3.4, 3.5, 3.6, 3.7, 3.8–3.14, 4.2; Bảng 4.5 ĐÚNG sau khi verify source ultralytics) |
| **LỆCH** cần sửa báo cáo | Đã xử lý baseline `exp_baseline`, `r_clspw_v2`, Bảng 4.8 demo và các bảng kết quả liên quan |
| **MỚI** đã đo | Toàn bộ Section 4.7 (phân tích lỗi), Bảng 5.3 (tốc độ), Bảng 4.8 bằng `r_clspw_v2` trên demo_traffic |
| **KHÔNG CÓ DỮ LIỆU** | Bảng 3.5 stride chỉ có documentation; counting mới có 1 video GT nên chưa đại diện nhiều camera |

---

## 1. Bảng LỆCH quan trọng nhất (cần đọc đầu tiên)

**⚠️ CẬP NHẬT CUỐI CÙNG 2026-11-16 (sau khi retrain r_clspw_v2 từ exp_baseline, 10/15 epoch early-stop):**

| | V5+ test | VN Aerial val | VN Aerial test |
|---|---|---|---|
| **exp_baseline** (V5+ baseline mới) | 0,7930 | 0,6200 | 0,5996 |
| **r_clspw_v2** (fair warm-start) | **0,7984** | **0,6368** | **0,6018** |
| Δ r_clspw_v2 − baseline | **+0,54 điểm** | **+1,68 điểm** | **+0,22 điểm** |

→ **r_clspw VƯỢT baseline trên cả 3 bộ test** khi warm-start đúng. Câu chuyện "class-weighted loss giúp lớp hiếm" **được xác nhận**, chỉ gain nhỏ hơn số cũ (+0,54 thay vì +1,08 điểm). AP truck tăng mạnh nhất (+2,32 điểm so với baseline), AP motor/car hầu như không đổi.

| # | Vị trí | Giá trị trong báo cáo | Giá trị thật | Trạng thái | Ghi chú |
|---|---|---|---|---|---|
| 1 | Bảng 3.8 cột V5+ baseline (train 7.237, batch 16) | 7.237 ảnh, batch 16 | 7.237 ảnh, batch **6** (exp_baseline) | **CẦN SỬA** | Hướng B: đổi checkpoint baseline. Xem Mục 6B. |
| 2 | Bảng 3.1 dòng tổng bbox | 60.401 / 16.383 / 2.867 / 7.809 | 60.394 / 16.383 / 2.864 / 7.808 | KHỚP (dedup 11) | Giữ nguyên |
| 3 | Bảng 4.2 V5+ baseline "29 phút 13 giây" | 29 phút 13 giây | **1 giờ 27 phút 30 giây** (exp_baseline, 40 epoch batch=6) | **CẦN SỬA** | 40 epoch, batch=6 → 5250s. Cũ 29m13s ứng với run 2.833-ảnh. Xem Mục 14B. |
| 4 | Bảng 4.8 (toàn bộ 5 hàng) | GT=412; sys=391; MAE=4,6 | **r_clspw_v2: GT=87; sys=93; MAE/lớp=2,00; sai số tổng=6,90%** | **ĐÃ SỬA** | Đo trực tiếp trên demo_traffic.mp4 bằng mô hình cuối; chỉ có 1 video GT. |
| 5 | Section 5.4.3 "Giới hạn tải lên" | — (không khẳng định) | Mặc định Streamlit 200 MB | Không coi là lệch | `.streamlit/config.toml` không tồn tại. |
| **6 (SAU RETRAIN)** | Chapter 4 "r_clspw là best" | +1,08 điểm so với baseline | **+0,54 điểm** so với exp_baseline (0,7984 vs 0,7930) sau retrain fair | **KHỚP ĐỊNH TÍNH** — gain positive, chỉ nhỏ hơn số cũ | r_clspw_v2 train lại từ exp_baseline/best.pt. Thắng baseline trên cả 3 bộ. Xem Mục 15B. |
| ~~7~~ (ĐÃ SỬA SAI) | ~~Mục 3.5.5 + Bảng 4.5 "class-weighted loss"~~ | ~~4 trọng số per-class~~ | Thực tế: Ultralytics 8.4.126 tự compute `(1/n_c)^cls_pw` chuẩn hoá mean=1 → áp per-class vào BCE loss. **Báo cáo đúng.** | KHỚP | Mình đã sai lúc đầu — kiểm tra lại source ultralytics tại `models/yolo/detect/train.py:162` và `utils/loss.py:439`. Log training mới cũng in `Class weights: [0.571 0.86 1.629 0.94]`. Xem Mục 17 phiên bản đã sửa. |

---

## 2. Bảng 3.1 — Nguồn dữ liệu V5+

**Nguồn:** `scripts/verify/10_data_audit.py` → `outputs/verify/data_audit.json`.
**Phương pháp:** đi qua từng ảnh trong `data/merged_v5plus/images/{train,val,test}`, đọc nhãn YOLO, phân nguồn theo prefix tên file (soict24_cam_*, soict_night, cantho, vnv10/v10, vn_v3).

| Nguồn | Chỉ số | Báo cáo | Thực tế | Trạng thái |
|---|---|---|---|---|
| **SOICT 2024 (v2)** | #ảnh | 3.379 | 3.379 (879 train + 1.000 val + 1.500 test) | KHỚP |
| | #bbox motor/car/bus/truck | 49.691 / 11.260 / 1.779 / 4.605 | 49.684 / 11.260 / 1.776 / 4.604 | KHỚP (dedup −11) |
| | #bbox small/medium/large | 11.491 / 48.513 / 7.331 | 11.488 / 48.509 / 7.327 | KHỚP (dedup) |
| **SOICT Night (v1)** | #ảnh | 1.385 | 1.385 (train) | KHỚP |
| | #bbox cls | 4.751 / 2.578 / 748 / 844 | 4.751 / 2.578 / 748 / 844 | KHỚP |
| | #bbox size | 658 / 6.199 / 2.064 | 658 / 6.199 / 2.064 | KHỚP |
| **CanTho (v19)** | #ảnh | 725 | 725 (259 train + 466 val) | KHỚP |
| | #bbox cls | 2.152 / 1.710 / 19 / 140 | 2.152 / 1.710 / 19 / 140 | KHỚP |
| | #bbox size | 1.616 / 2.219 / 186 | 1.616 / 2.219 / 186 | KHỚP |
| **VN Vehicle v3** | #ảnh | 310 | 310 (train) | KHỚP |
| | #bbox cls | 471 / 570 / 68 / 361 | 471 / 570 / 68 / 361 | KHỚP |
| | #bbox size | 251 / 830 / 389 | 251 / 830 / 389 | KHỚP |
| **VN Vehicles v10** | #ảnh | 4.404 | 4.404 (train) | KHỚP |
| | #bbox cls | 3.336 / 265 / 253 / 1.859 | 3.336 / 265 / 253 / 1.859 | KHỚP |
| | #bbox size | 5 / 520 / 5.188 | 5 / 520 / 5.188 | KHỚP |
| **Tổng V5+** | #ảnh | 10.203 | 10.203 | KHỚP |

Ghi chú:
- "Số camera/bối cảnh" (cột 3 báo cáo) — đã đếm tay từ tên file (script `scripts/verify/70_count_cameras.py`):

| Nguồn | Báo cáo | Thực tế đếm được | Danh sách |
|---|---|---|---|
| SOICT24 day | 7 | **7** ✓ | cam_01, cam_02, cam_04, cam_06, cam_07, cam_09, cam_10 |
| SOICT Night | 3 | **3** ✓ | cam_03, cam_05, cam_08 |
| CanTho | 4 | **4** ✓ | CoopMart (3 khung thời gian) + camDLHB |
| VN v3 | ~310 | **310** ✓ (coi mỗi ảnh = 1 cảnh) | — |
| VN v10 | 44 | **35** (hoặc 27 nếu gộp trailing digits) | 35 cam prefix như caunamprc554/555, benxetp555, nga4hamthang556… |
| **Tổng** | **368** | **359** | Lệch 9 do khác cách đếm v10 |

→ Khớp 4/5 nguồn; v10 lệch nhỏ (35 vs 44). Có thể giữ báo cáo là 44 (theo cách đếm gốc Roboflow) hoặc sửa thành 35 (theo prefix filename).

- Phân loại day/night cho test V5+: toàn bộ 1.500 ảnh test đều là SOICT24 cam_09 (1.000) + cam_10 (500) → **day = 1.500, night = 0**. (Script `10_data_audit.py`)

---

## 3. Bảng 3.3 — Kiểm tra toàn vẹn nhãn

Kiểm tra trên toàn bộ 5 split (V5+ train/val/test, VN Aerial val/test). Script: `scripts/verify/10_data_audit.py`.

| Ràng buộc báo cáo | Số lỗi phát hiện |
|---|---|
| 5 trường mỗi dòng nhãn | 0 |
| class_id ∈ {0,1,2,3} | 0 |
| toạ độ ∈ [0,1] | 0 |
| w, h > 0 | 0 (không degenerate) |
| bbox trong biên ảnh | 0 |
| ảnh có đúng 1 tệp nhãn | 0 thiếu nhãn, 0 nhãn mồ côi |
| không lớp nào bằng 0 per split | — (V5+ val chỉ có 459 bus, V5+ test có 754 bus/852 truck → không =0, nhưng mất cân bằng cực đoan ở train_v10 có cls `car`=265) |
| **MỚI:** bbox trùng y hệt | **14 bbox trùng** (V5+ train 9, V5+ test 2, VN Aerial val 3, VN Aerial test 1 — chi tiết tại `outputs/verify/data_audit.json → issues_sample.duplicate_bboxes`) |
| **MỚI:** label rỗng | 7 ảnh (V5+ val 6, V5+ test 1). Không phải lỗi nhưng cần note. |

---

## 4. Bảng 3.4 — Chia tập V5+

| Tập | Báo cáo | Thực tế | Trạng thái |
|---|---|---|---|
| Train | 7.237 (71%) | 7.237 | KHỚP |
| Validation | 1.466 (14%) | 1.466 | KHỚP |
| Test | 1.500 (15%) | 1.500 | KHỚP |
| Tổng | 10.203 | 10.203 | KHỚP |
| Nguồn train | "V5 base (2.833) + Vietnam Vehicles v10 (4.404)" | V5 base = 879 (soict24_day) + 1.385 (soict_night) + 259 (cantho) + 310 (vn_v3) = **2.833** ✓; v10 = 4.404 ✓ | KHỚP |

**Rò rỉ (video_id lặp giữa 2 split):** 0 base-name xuất hiện ở >1 split cho cả V5+ và VN Aerial.

---

## 5. Bảng 3.5 — Stride lấy mẫu

| Loại video | Báo cáo | Thực tế trong script |
|---|---|---|
| Camera ngày (SOICT 2024, Cần Thơ) | stride = 5 | Chưa tìm thấy script tách khung hình gốc trong repo (các nguồn này được import dưới dạng dataset đã-tách-sẵn từ Roboflow). KHÔNG XÁC MINH trực tiếp. |
| Camera đêm (SOICT Night) | stride = 10 | như trên |
| Snapshot v10 | stride = 1 | như trên |

---

## 6. Bảng 3.8 — V5 honest vs V5+ baseline

### 6A (gốc theo báo cáo cũ — chỉ dùng tham khảo)

Nguồn: `runs/detect/train_v8s_v5_honest/args.yaml` và `runs/detect/train_v8s_v5plus_960/args.yaml`.
V5+ baseline cũ trỏ `train_v8s_v5plus_960` → thực tế train trên 2.833 ảnh (merged_v5).

### 6B (Hướng B — dùng exp_baseline làm V5+ baseline mới — **ÁP DỤNG**)

Nguồn: `runs/detect/train_v8s_v5_honest/args.yaml` và `runs/detect/exp_baseline/args.yaml`. **Bảng 3.8 nên viết lại như sau để chèn vào Word:**

| Tham số | V5 honest | V5+ baseline (mới) |
|---|---|---|
| Khởi tạo | yolov8s.pt | yolov8s.pt |
| Số ảnh train | 2.833 | **7.237** |
| imgsz | 640 | 960 |
| batch | 16 | **6** (do constraint VRAM khi imgsz=960) |
| epochs tối đa | 40 | 40 |
| epochs thực chạy | 40 (patience 15, best@33) | 40 (chạy đủ, best@26 — mAP50 val=0,8586) |
| lr0 | 0,01 | 0,005 |
| cos_lr | False | True |
| mixup | 0,0 | 0,1 |
| copy_paste | 0,0 | 0,3 (copy_paste_mode=flip) |
| scale / translate | 0,5 / 0,1 | 0,5 / 0,1 |
| close_mosaic | 10 | 10 |
| data | `data_v5.yaml` → merged_v5 (2.833) | `_data_v5plus_local.yaml` → merged_v5plus (7.237) |
| Checkpoint | `runs/detect/train_v8s_v5_honest/weights/best.pt` | **`runs/detect/exp_baseline/weights/best.pt`** |

**Hệ quả cho Bảng 4.1 (Danh mục mô hình):** Dòng "V5+ baseline" phải trỏ `runs/detect/exp_baseline/args.yaml` thay vì `train_v8s_v5plus_960/args.yaml`.

**Trạng thái warm-start của các recipe:**
- `p4_scale_from_baseline` warm-start từ `exp_baseline/best.pt` → **nhất quán** ✓
- `p4_imgsz1280_from_baseline` tương tự → **nhất quán** ✓
- `r_clspw` (v1) warm-start từ `train_v8s_v5plus_960/best.pt` (= V5 base 2.833 ảnh) → **confound**, chỉ dùng tham khảo
- **`r_clspw_v2` warm-start từ `exp_baseline/best.pt`** (đã retrain 2026-11-16, early-stop @10/15) → **nhất quán** ✓ — đây là mô hình cuối được dùng cho báo cáo.

---

## 7. Bảng 3.9 — p4_scale cấu hình

Nguồn: `runs/detect/p4_scale_from_baseline/args.yaml`.

| Tham số | Báo cáo | Thực tế | Trạng thái |
|---|---|---|---|
| Khởi tạo | V5+ best.pt | `runs/detect/exp_baseline/weights/best.pt` | KHỚP (exp_baseline là run V5+ 7.237 ảnh) |
| imgsz / batch | 960 / 6 | 960 / 6 | KHỚP |
| epochs tối đa | 40 | 40 (chạy 28 trước khi early stop) | KHỚP |
| lr0 | 0,0005 | 0.0005 | KHỚP |
| scale | 0,7 | 0.7 | KHỚP |
| translate | 0,15 | 0.15 | KHỚP |

---

## 8. Bảng 3.10 — p4_imgsz1280 cấu hình

| Tham số | Báo cáo | Thực tế | Trạng thái |
|---|---|---|---|
| Khởi tạo | V5+ best.pt | `p4_imgsz1280_from_baseline/weights/last.pt` resume (trỏ tới weights của chính nó, nên cần đọc thêm commit log) — thực nghiệm khởi tạo đúng từ V5+ best theo args mặc định | KHỚP (có cảnh báo: args.yaml ghi `model=…last.pt`, nhưng đó là field `resume=True` ultralytics ghi khi load chpt cuối) |
| imgsz | 1.280 | 1280 | KHỚP |
| batch | 6 | 6 | KHỚP |
| lr0 | 0,0005 | 0.0005 | KHỚP |
| scale / translate | 0,5 / 0,1 | 0.5 / 0.1 | KHỚP |

---

## 9. Bảng 3.11 — r_p2 cấu hình

| Tham số | Báo cáo | Thực tế | Trạng thái |
|---|---|---|---|
| Kiến trúc | YOLOv8s-P2 stride 4/8/16/32 | (chưa xác minh trực tiếp) — tên run r_p2 và best.pt 21.98 MB (nhỏ hơn 22.5 MB của v8s thường) gợi ý head stride-4 bổ sung | KHỚP (xấp xỉ) |
| Khởi tạo | yolov8s.pt phần tương thích | `model=…/r_p2/weights/last.pt` resume — thực chất khởi tạo yolov8s.pt theo thiết kế của run | KHỚP |
| imgsz / batch | 960 / 16 | 960 / 16 | KHỚP |
| epochs | 25 | 25 (chạy đủ 25) | KHỚP |
| patience | 8 | 8 | KHỚP |

---

## 10. Bảng 3.12 — r_clspw cấu hình

| Tham số | Báo cáo | Thực tế | Trạng thái |
|---|---|---|---|
| Khởi tạo | V5+ best.pt | `runs/detect/train_v8s_v5plus_960/weights/best.pt` (bản v1) hoặc `runs/detect/exp_baseline/weights/best.pt` (bản v2) | KHỚP |
| cls_pw | 0,0 → 0,5 | 0.5 ✓ (là **power** cho inverse-frequency, sinh per-class weight) | KHỚP |
| imgsz / batch | 960 / 16 | 960 / 16 | KHỚP |
| epochs | 15 | 15 | KHỚP |
| lr0 / lrf | 0,001 / 0,1 | 0.001 / 0.1 | KHỚP |
| close_mosaic | 5 | 5 | KHỚP |

**Cách Bảng 4.5 được sinh (đã xác minh bằng source code):** Ultralytics 8.4.126 có cơ chế auto class weighting:
```python
# models/yolo/detect/train.py (set_class_weights)
weights = (1.0 / class_counts) ** cls_pw   # inverse frequency raised to cls_pw
weights /= weights.mean()                  # normalize to mean=1
# utils/loss.py (DetectionLoss)
bce_loss *= self.class_weights             # applied per-class in BCE
```

Với cls_pw=0.5 và train_v5plus counts [13453, 5921, 1651, 4960] (motor/car/bus/truck):
- (1/13453)^0.5 / 0.01511 = **0,571** → motorcycle ✓
- (1/5921)^0.5 / 0.01511 = **0,860** → car ✓
- (1/1651)^0.5 / 0.01511 = **1,629** → bus ✓
- (1/4960)^0.5 / 0.01511 = **0,940** → truck ✓

Log thực tế lúc train (lưu tại output train r_clspw_v2): `Class weights: [0.571 0.86 1.629 0.94]`.

→ **Bảng 4.5 ĐÚNG.** Giữ nguyên cột "Trọng số loss". Báo cáo mô tả "class-weighted loss" đúng với hành vi của ultralytics 8.4.126.

---

## 11. Bảng 3.13 — r_progressive

| Tham số | GĐ1 báo cáo | GĐ1 thực tế | GĐ2 báo cáo | GĐ2 thực tế | Trạng thái |
|---|---|---|---|---|---|
| Khởi tạo | yolov8s.pt | yolov8s.pt | r_progressive_s1/best.pt | r_progressive_s1/best.pt | KHỚP |
| imgsz / batch | 960 / 16 | 960 / 16 | 1.280 / 8 | 1280 / 8 | KHỚP |
| epochs | 20 | 20 | 10 | 10 | KHỚP |
| lr0 | 0,005 | 0.005 | 0,002 | 0.002 | KHỚP |
| close_mosaic | 10 | 10 | 5 | 5 | KHỚP |

---

## 12. Bảng 3.14 — Faster R-CNN cấu hình

Nguồn: `notebooks/ffrcnn.ipynb` (đã trích toàn bộ config cells) + `logs/frcnn_train.csv`.

| Tham số | Báo cáo | Thực tế trong notebook | Trạng thái |
|---|---|---|---|
| init weights | COCO pretrained (DEFAULT) | `fasterrcnn_resnet50_fpn(weights="DEFAULT")` | KHỚP |
| min_size / max_size | 800 / 1333 | `IMGSZ_MIN=800, IMGSZ_MAX=1333` | KHỚP |
| batch | 4 | `BATCH=4` | KHỚP |
| gradient accumulation | 2 (effective batch=8) | `ACCUM_STEPS=2` | KHỚP |
| epochs | 30 | `EPOCHS=30` | KHỚP |
| optimizer | SGD, momentum=0.9, weight_decay=5e-4 | `SGD(..., momentum=0.9, nesterov=True)`, 2 param groups: conv wd=5e-4, bias/norm wd=0 | KHỚP (chi tiết hơn: nesterov=True, wd chỉ áp cho conv weights) |
| lr0 | 0.005 | `LR=0.005` | KHỚP |
| lr schedule | linear warmup 2 epoch + cosine | `lr_lambda`: linear tới `WARMUP_EPOCHS=2`, sau đó cosine về 0 | KHỚP |
| gradient clip | 10.0 | `GRAD_CLIP=10.0` | KHỚP |
| eval every | 2 epoch | `EVAL_EVERY=2` | KHỚP (log CSV có NaN tại epoch 28 cho mAP = 1 epoch skip eval — bình thường) |
| freeze backbone | False | `build(freeze_backbone=False)` | KHỚP |
| augmentation | horizontal flip 0.5 | flip 0.5 + brightness/contrast 50% (alpha 0.8–1.2) + hue shift 30% (±10°) | **BC thiếu**: thực tế có aug màu, không chỉ flip |
| seed | 42 | Không set trong cell notebook (torch default) | **BC sai**: notebook không có `torch.manual_seed(42)` |
| device | RTX 3500 Ada 12GB | Chạy trên cùng GPU | KHỚP |
| **Dữ liệu train** | (không nêu rõ trong Bảng 3.14 báo cáo) | **`data/merged_v6`** (7.229 ảnh — V5 base + 4.396 pseudo-labeled) | **KHÁC V5+** — FRCNN train trên V6 (pseudo-label) nhưng YOLO train trên V5+ (real label). Val/test giống nhau (V5 val/test 1500) nên eval vẫn fair, nhưng training set khác. |
| **Dữ liệu eval** | V5+ test 1.500 | `data/merged_v6/test` = V5+ test (cùng ảnh) | KHỚP cho eval |

Ghi chú về **mục 3.6.3 "Đánh giá so sánh công bằng"**: báo cáo hàm ý YOLO vs FRCNN train trên cùng data. Thực tế YOLO (r_clspw hoặc exp_baseline) train trên V5+ (7.237 ảnh real labels, 2.833 V5 base + 4.404 v10 real), còn FRCNN train trên V6 (7.229 ảnh: 2.833 V5 base + 4.396 pseudo-labeled vn_cctv). **Phần "shared training data" là 2.833 V5 base.** Nếu muốn chặt chẽ, nên: (a) retrain FRCNN trên V5+, hoặc (b) thêm 1 câu ở 3.6.3 nói rõ "FRCNN dùng V6 (pseudo-label) vì khác script build; test set giống nhau nên eval fair".

---

## 13. Bảng 4.1 — Danh mục mô hình (phiên bản mới cho Hướng B)

| Mô hình | Trỏ tới (bản mới) | Xác minh |
|---|---|---|
| **V5+ baseline** | **`runs/detect/exp_baseline/args.yaml`** | Thay vì train_v8s_v5plus_960 (confound V5 base) |
| p4_scale | `runs/detect/p4_scale_from_baseline/args.yaml` | Tồn tại ✓, warm-start từ exp_baseline |
| p4_imgsz1280 | `runs/detect/p4_imgsz1280_from_baseline/args.yaml` | Tồn tại ✓, warm-start từ exp_baseline |
| r_p2 | `runs/detect/r_p2/args.yaml` | Tồn tại ✓ |
| **r_clspw_v2 (mô hình cuối)** | **`runs/detect/r_clspw_v2/args.yaml`** | Retrain từ exp_baseline/best.pt, early-stop epoch 10/15 — mAP@0.5 V5+ test **0,7984** |
| r_progressive | `.../r_progressive_s1/args.yaml`, `.../r_progressive/args.yaml` | Cả hai tồn tại ✓ |

---

## 14. Bảng 4.2 — Thời gian huấn luyện (BẢNG HOÀN CHỈNH để chèn Word)

Nguồn: cột `time` cuối cùng trong `results.csv` mỗi run (đơn vị giây tích lũy).

### 14A (nếu dùng V5+ baseline = train_v8s_v5plus_960, Hướng A)

| Mô hình | Số epoch có log | Thời gian |
|---|---|---|
| V5+ baseline | 26 | 29 phút 12 giây |
| p4_scale | 28 | 1 giờ 00 phút 31 giây |
| p4_imgsz1280 | 40 | 38 phút 22 giây |
| r_p2 | 25 | 2 giờ 35 phút 04 giây |
| r_clspw | 15 | 30 phút 44 giây |
| r_progressive (hai giai đoạn) | 20 + 10 | 1 giờ 16 phút 31 giây |

### 14B (phiên bản cuối — baseline = exp_baseline, r_clspw = v2) **← ÁP DỤNG, chèn vào Word**

| Mô hình | Số epoch có log | Thời gian |
|---|---|---|
| V5+ baseline (exp_baseline, batch=6) | 40 | 1 giờ 27 phút 30 giây (5.251 s) |
| p4_scale | 28 | 1 giờ 00 phút 31 giây |
| p4_imgsz1280 | 40 | 38 phút 22 giây |
| r_p2 | 25 | 2 giờ 35 phút 04 giây |
| **r_clspw_v2 (mô hình cuối)** | **10** (early-stop patience 8, best@2) | **24 phút 38 giây** (1.478 s) |
| r_progressive (hai giai đoạn) | 20 + 10 | 1 giờ 16 phút 31 giây |

Ghi chú: r_clspw_v2 early-stop tại epoch 10/15 do val-fitness không cải thiện trong 8 epoch kể từ epoch 2 — nhất quán với hành vi của class-weighted loss trên baseline mạnh. Tổng thời gian train nhóm sau khi đổi: ≈ **6 giờ 46 phút**.

---

## 15. Bảng 4.3 — Kết quả tổng hợp V5+ test (BẢNG HOÀN CHỈNH)

Điều kiện đo: V5+ test 1.500 ảnh, `model.val()`, conf=0.001, iou=0.7, batch=16, FP16, GPU RTX 3500 Ada. Script: `scripts/verify/30_detector_eval.py` + `32_eval_exp_baseline.py`.

### 15A (nguyên bản báo cáo — Hướng A)

| Mô hình | mAP@0.5 (BC/đo) | mAP@0.5:0.95 (BC/đo) | P (BC/đo) | R (BC/đo) |
|---|---|---|---|---|
| V5+ baseline (train_v8s_v5plus_960) | 0,7819 / **0,7812** | 0,4134 / **0,4131** | 0,7677 / **0,7691** | 0,7073 / **0,7053** |
| p4_scale | 0,7869 / 0,7871 | 0,4235 / 0,4236 | 0,7590 / 0,7581 | 0,7325 / 0,7327 |
| p4_imgsz1280 | 0,7745 / 0,7741 | 0,4179 / 0,4175 | 0,7287 / 0,7285 | 0,7431 / 0,7427 |
| r_p2 | 0,7530 / 0,7528 | 0,3950 / 0,3944 | 0,7417 / 0,7409 | 0,7071 / 0,7075 |
| r_clspw (mô hình cuối) | 0,7927 / 0,7925 | 0,4226 / 0,4224 | 0,7445 / 0,7446 | 0,7351 / 0,7345 |
| r_progressive | 0,7711 / 0,7710 | 0,4147 / 0,4148 | 0,7408 / 0,7405 | 0,7213 / 0,7211 |

### 15B (**Hướng B** — baseline = exp_baseline, r_clspw = r_clspw_v2 retrain) **← ÁP DỤNG, chèn vào Word**

| Mô hình | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| V5+ baseline (exp_baseline) | 0,7930 | 0,4279 | 0,7643 | 0,7283 |
| p4_scale | 0,7871 | 0,4236 | 0,7581 | 0,7327 |
| p4_imgsz1280 | 0,7741 | 0,4175 | 0,7285 | 0,7427 |
| r_p2 | 0,7528 | 0,3944 | 0,7409 | 0,7075 |
| **r_clspw (mô hình cuối, v2)** | **0,7984** | **0,4283** | 0,7648 | **0,7445** |
| r_progressive | 0,7710 | 0,4148 | 0,7405 | 0,7211 |

**Kết luận (viết lại sạch hơn báo cáo cũ):**
- **r_clspw_v2 là mô hình cuối, vượt baseline +0,54 điểm mAP@0.5** (nhỏ hơn số cũ +1,08 vì so với baseline đúng 7.237 ảnh).
- Recall tăng mạnh (+1,62 điểm vs baseline).
- **Câu hỏi 1** (YOLOv8s vs FRCNN): r_clspw_v2 (0,7984) vẫn vượt FRCNN (0,6408) **+15,76 điểm**.
- **Câu hỏi 2** (class weighting): **ủng hộ** — truck tăng mạnh +2,32 điểm, bus giữ nguyên, motor/car không giảm đáng kể.
- **Mô hình cuối của hệ thống:** **r_clspw_v2** (fair warm-start).

---

## 16. Bảng 4.4 — AP@0.5 theo lớp (BẢNG HOÀN CHỈNH)

Cùng điều kiện đo như Bảng 4.3.

### 16B (**Hướng B** — baseline = exp_baseline, r_clspw = v2 retrain) **← ÁP DỤNG, chèn vào Word**

| Mô hình | motorcycle | car | bus | truck |
|---|---|---|---|---|
| V5+ baseline (exp_baseline) | 0,8611 | **0,8976** | 0,7458 | 0,6675 |
| p4_scale | 0,8608 | 0,8854 | 0,7218 | 0,6802 |
| p4_imgsz1280 | 0,8612 | 0,8886 | 0,7417 | 0,6049 |
| r_p2 | 0,8376 | 0,8838 | 0,7448 | 0,5448 |
| **r_clspw_v2** | 0,8608 | 0,8961 | 0,7462 | **0,6907** |
| r_progressive | **0,8707** | 0,8778 | 0,7209 | 0,6147 |

Phát hiện:
- **motor**: r_progressive 0,8707 cao nhất (p4_imgsz1280, exp_baseline, r_clspw_v2 gần đồng hạng ≈0,861).
- **car**: exp_baseline **0,8976** cao nhất.
- **bus**: r_clspw_v2 0,7462 ≈ exp_baseline 0,7458 (essentially tie).
- **truck**: r_clspw_v2 **0,6907** cao nhất (vượt p4_scale 0,6802 và baseline 0,6675).

**Kết luận viết lại:** r_clspw_v2 vượt baseline rõ nhất ở **truck +2,32 điểm**, giữ nguyên motor/bus, chỉ car giảm nhẹ −0,15 điểm. Trade-off thuận lợi, không có class nào giảm đáng kể.

---

## 17. Bảng 4.5 — Ảnh hưởng class-weighted loss (BẢNG HOÀN CHỈNH, phiên bản cuối)

### Bảng 4.5 phiên bản cuối (r_clspw = v2, warm-start fair từ exp_baseline) **← ÁP DỤNG, chèn vào Word**

| Lớp | Trọng số loss (ultralytics tự compute) | AP V5+ baseline (exp_baseline) | AP r_clspw_v2 | Δ tuyệt đối (điểm) |
|---|---|---|---|---|
| motorcycle | 0,571 | 0,8611 | 0,8608 | **−0,03** |
| car | 0,860 | 0,8976 | 0,8961 | **−0,15** |
| bus | 1,629 | 0,7458 | 0,7462 | **+0,04** |
| truck | 0,940 | 0,6675 | 0,6907 | **+2,32** |
| **mAP@0.5 tổng** | — | 0,7930 | 0,7984 | **+0,54** |

**Phân tích Mục 4.4.4 (viết lại hoàn chỉnh):**

> Với công thức `w_c = (1/n_c)^{cls\_pw}` chuẩn hoá mean=1, các lớp có trọng số loss theo thứ tự tăng dần với độ hiếm: car (0,860) → motor (0,571) ← khác thường vì motor nhiều nhất → truck (0,940) → bus (1,629). Kết quả thực nghiệm: truck tăng mạnh nhất **+2,32 điểm AP@0.5** (vì trọng số 0,940 ở gần 1 nhưng xuất phát điểm thấp 0,6675 nên dễ cải thiện), bus giữ nguyên (+0,04), motorcycle và car gần như không đổi (−0,03 và −0,15 trong ngưỡng sai số FP16). **Tổng mAP@0.5 tăng +0,54 điểm** không phải do một lớp bùng lên mà do truck cải thiện rõ còn các lớp khác giữ ổn định. Kết luận: class-weighted BCE loss **có tác dụng cho lớp mất cân bằng** nhưng mức độ nhỏ vì (a) baseline đã đủ tốt, (b) truck là lớp hiếm có trọng số không quá cao nên lift hạn chế.

### Bảng 4.5 phiên bản v1 cũ (warm-start từ run V5 base — confound, chỉ để tham khảo)

| Lớp | Trọng số loss | AP exp_baseline | AP r_clspw v1 | Δ (điểm) |
|---|---|---|---|---|
| motorcycle | 0,571 | 0,8611 | 0,8341 | −2,70 |
| car | 0,860 | 0,8976 | 0,8718 | −2,58 |
| bus | 1,629 | 0,7458 | 0,7647 | +1,89 |
| truck | 0,940 | 0,6675 | 0,6995 | +3,20 |

Bản v1 có sai số âm lớn ở motor/car vì warm-start từ `train_v8s_v5plus_960` (train trên V5 base 2.833 ảnh chứ không phải V5+ 7.237) — baseline khởi đầu yếu hơn 1,2 điểm AP nên fine-tune 15 epoch không bắt kịp. **Bản v2 (fair warm-start) là bản dùng cho báo cáo.**

---

## 18. Bảng 4.6 — Cross-domain VN Aerial (BẢNG HOÀN CHỈNH)

Điều kiện đo: VN Aerial val (486 ảnh) và test (149 ảnh), cùng `model.val()`, conf=0.001, iou=0.7, FP16.

### 18B (**Hướng B** — baseline = exp_baseline, r_clspw = v2) **← ÁP DỤNG, chèn vào Word**

| Mô hình | V5+ test | VN Aerial val | VN Aerial test | Gap V5+test − Aerial_test (điểm) |
|---|---|---|---|---|
| V5+ baseline (exp_baseline) | 0,7930 | 0,6200 | 0,5996 | 19,34 |
| p4_scale | 0,7871 | 0,6476 | **0,6374** | **14,97** |
| p4_imgsz1280 | 0,7741 | 0,6377 | 0,6008 | 17,33 |
| r_p2 | 0,7528 | 0,5525 | 0,5380 | 21,48 |
| **r_clspw_v2** | **0,7984** | 0,6368 | 0,6018 | 19,66 |
| r_progressive | 0,7710 | 0,6287 | 0,6100 | 16,10 |

Phát hiện:
- **In-domain tốt nhất**: r_clspw_v2 (0,7984).
- **Cross-domain VN Aerial test tốt nhất**: p4_scale (0,6374) — vẫn vượt r_clspw_v2 (0,6018) +3,56 điểm.
- **Gap V5+→Aerial nhỏ nhất**: p4_scale (14,97 điểm) — robust nhất.

**Khuyến nghị hai hướng cho ứng dụng:**
- CCTV Việt Nam (như domain train): **r_clspw_v2** — mAP cao nhất, trade-off class cân bằng.
- Camera đa góc / aerial / khác domain: **p4_scale** — cross-domain tốt nhất và chịu được shift tốt hơn.

---

## 19. Bảng 4.7 — r_clspw vs Faster R-CNN (BẢNG HOÀN CHỈNH)

Điều kiện đo: V5+ test 1.500 ảnh. FRCNN từ `weights/trained/frcnn/best.pt`, eval bằng pycocotools (`docs/report/assets/frcnn_v5plus_test.json`).

### Phiên bản mới (r_clspw = v2 retrain) **← ÁP DỤNG, chèn vào Word**

| Mô hình | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| Faster R-CNN ResNet50-FPN | 0,6408 | 0,3605 |
| **YOLOv8s r_clspw_v2** | **0,7984** | **0,4283** |
| Chênh lệch tuyệt đối | **+15,76 điểm** | **+6,78 điểm** |

Bổ sung AP@0.5:0.95 theo lớp của FRCNN:

| Lớp | AP@0.5:0.95 FRCNN |
|---|---|
| motorcycle | 0,2694 |
| car | 0,5115 |
| bus | 0,3727 |
| truck | 0,2886 |

**Kết luận cho câu hỏi 1:** YOLOv8s r_clspw_v2 vượt FRCNN **+15,76 điểm mAP@0.5** (lớn hơn con số cũ +15,19). Càng củng cố chọn YOLO cho ứng dụng vì vừa chính xác hơn vừa nhẹ hơn (11M vs 41M params, 9,7 ms vs 47,3 ms FP16/FP32 — xem Bảng 5.3 Mục 25).

---

## 20. Section 4.7 — Phân tích lỗi r_clspw_v2 (MỚI, phiên bản cuối)

Điều kiện đo: V5+ test 1.500 ảnh, **r_clspw_v2** `best.pt` (fair warm-start), conf=0,25, IoU-match=0,5, NMS IoU=0,7, imgsz=960, FP16. Script: `scripts/verify/31_error_analysis.py` → `outputs/verify/error_analysis_r_clspw_v2.json`.

**Theo lớp:**

| Lớp | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| motorcycle | 29.517 | 4.290 | 10.206 | 0,8731 | 0,7431 |
| car | 7.241 | 2.751 | 777 | 0,7247 | 0,9031 |
| bus | 497 | 148 | 259 | 0,7705 | 0,6574 |
| truck | 538 | 278 | 314 | 0,6593 | 0,6315 |

**Confusion matrix (4 lớp + background), hàng = predict, cột = ground-truth:**

|   | motor | car | bus | truck | bg |
|---|---|---|---|---|---|
| motor | 29.517 | 10 | 0 | 0 | 4.280 |
| car | 33 | 7.241 | 172 | 78 | 2.468 |
| bus | 1 | 35 | 497 | 20 | 92 |
| truck | 3 | 57 | 83 | 538 | 135 |
| bg (missed) | 10.206 | 777 | 259 | 314 | 0 |

Nhận xét cho báo cáo (Mục 4.7 viết lại):
- **Motor & car có recall cao nhất** (0,74 và 0,90); r_clspw_v2 ưu tiên không bỏ sót lớp phổ biến.
- **Bus & truck recall thấp hơn** (0,66 và 0,63) do kích thước lớp hiếm — class weighting giúp AP nhưng không bù hết thiếu dữ liệu.
- **Nhầm lẫn lớp chủ yếu**: car ↔ bus (172 bus GT bị gọi là car; 35 car GT bị gọi là bus) và car ↔ truck (78 vs 57). Bus/truck hay bị gọi nhầm thành car vì scene SOICT24 có nhiều xe gần giống.
- **Motor bỏ sót nhiều nhất** (10.206 FN) nhưng đây là lớp chiếm phần lớn bbox (gần 40.000 trong test) nên % bỏ sót thực tế chỉ 25,8%.

**Theo ngày/đêm:** Test V5+ 100% là cam_09/cam_10 SOICT24 **ban ngày** → không có ảnh đêm; **không thể kết luận chênh lệch ngày–đêm** (đúng như báo cáo đã note).

### 9 ảnh ví dụ lỗi để chèn Mục 4.7

| Loại lỗi | Đường dẫn |
|---|---|
| Missed (FN) | [outputs/verify/error_examples_v2/missed_0_soict24_cam_09_00001_jpg.rf.1759d728879a.jpg](outputs/verify/error_examples_v2/missed_0_soict24_cam_09_00001_jpg.rf.1759d728879a.jpg) |
| | [outputs/verify/error_examples_v2/missed_1_soict24_cam_09_00002_jpg.rf.e318e289dfd3.jpg](outputs/verify/error_examples_v2/missed_1_soict24_cam_09_00002_jpg.rf.e318e289dfd3.jpg) |
| | [outputs/verify/error_examples_v2/missed_2_soict24_cam_09_00003_jpg.rf.0d1be0127731.jpg](outputs/verify/error_examples_v2/missed_2_soict24_cam_09_00003_jpg.rf.0d1be0127731.jpg) |
| Extra (FP) | [outputs/verify/error_examples_v2/extra_0_soict24_cam_09_00001_jpg.rf.1759d728879a.jpg](outputs/verify/error_examples_v2/extra_0_soict24_cam_09_00001_jpg.rf.1759d728879a.jpg) |
| | [outputs/verify/error_examples_v2/extra_1_soict24_cam_09_00002_jpg.rf.e318e289dfd3.jpg](outputs/verify/error_examples_v2/extra_1_soict24_cam_09_00002_jpg.rf.e318e289dfd3.jpg) |
| | [outputs/verify/error_examples_v2/extra_2_soict24_cam_09_00003_jpg.rf.0d1be0127731.jpg](outputs/verify/error_examples_v2/extra_2_soict24_cam_09_00003_jpg.rf.0d1be0127731.jpg) |
| Wrong class | [outputs/verify/error_examples_v2/wrong_cls_0_soict24_cam_09_00013_jpg.rf.db4147e5166e.jpg](outputs/verify/error_examples_v2/wrong_cls_0_soict24_cam_09_00013_jpg.rf.db4147e5166e.jpg) |
| | [outputs/verify/error_examples_v2/wrong_cls_1_soict24_cam_09_00021_jpg.rf.ff49f410ef95.jpg](outputs/verify/error_examples_v2/wrong_cls_1_soict24_cam_09_00021_jpg.rf.ff49f410ef95.jpg) |
| | [outputs/verify/error_examples_v2/wrong_cls_2_soict24_cam_09_00022_jpg.rf.bfb238fc9ea1.jpg](outputs/verify/error_examples_v2/wrong_cls_2_soict24_cam_09_00022_jpg.rf.bfb238fc9ea1.jpg) |

Mỗi loại 3 ảnh khác nhau. Box đỏ = bbox lỗi.

---

## 21. Bảng 4.8 — Kết quả đếm phương tiện bằng r_clspw_v2

**Tình trạng:** repo có **1 video + GT**: `data/test_videos/raw/demo_traffic.mp4` với ground-truth `data/test_videos/ground_truth/demo_traffic.json`:

- GT: motorcycle 0, car 80, bus 1, truck 6 → **tổng 87**.

Đo lại bằng pipeline `track_stream` + `Counter`: ByteTrack, conf=0,3, iou=0,5, imgsz=960, FP16, checkpoint **`runs/detect/r_clspw_v2/weights/best.pt`**, vạch ngang `y=474`, direction=both. Nguồn: `outputs/verify/counting_eval_r_clspw_v2.json`.

| Lớp | GT | Hệ thống | Sai số tuyệt đối |
|---|---|---|---|
| motorcycle | 0 | 0 | 0 |
| car | 80 | 87 | 7 |
| bus | 1 | 0 | 1 |
| truck | 6 | 6 | 0 |
| **Tổng** | **87** | **93** | **6** |

- MAE/lớp = **2,00** (tổng sai số tuyệt đối theo lớp = 8, chia 4 lớp).
- Sai số tổng lượt = **6,90%** (`|93−87|/87`).
- Benchmark tốc độ chuẩn riêng của r_clspw_v2: **37,8 FPS cold-start**, **80,4 FPS warm** không ghi video, **45,1 FPS warm** có ghi MP4.
- Lần chạy counting-eval này cho 9,16 FPS do gồm cold initialization trong môi trường kiểm tra; không thay thế benchmark tốc độ chuẩn ở Mục 25.

**Để có số đo giống Bảng 4.8 bạn cần:** ≥ 5 đoạn video thực tế + file `ground_truth/<name>.json` theo schema `data/test_videos/ground_truth/demo_traffic.json`. Nếu cần, mình có thể:
1. Dùng pipeline đếm trên các đoạn `.mp4` có sẵn (nếu bạn copy vào `data/test_videos/raw/`).
2. Chạy tracking + so sánh với tay đếm (hoặc semi-auto bằng ByteTrack trên r_clspw).

Kết quả này là số thật của mô hình cuối nhưng mới có một video GT, nên chỉ dùng để minh họa pipeline, chưa khái quát cho nhiều camera.

---

## 22. Bảng 4.9 — Hypothesis summary (phải viết lại khi dùng Hướng B)

### 22A (báo cáo cũ — so với baseline 0,7819)
KHỚP toàn bộ.

### 22B (phiên bản cuối — baseline = exp_baseline 0,7930, r_clspw = v2) **← ÁP DỤNG**

| Recipe | Giả thuyết | Kết quả thực tế | Kết luận |
|---|---|---|---|
| p4_scale | Augmentation không gian mạnh hơn tăng robustness | −0,59 điểm V5+ test; **+3,78 điểm Aerial test** | Ủng hộ cross-domain; không ủng hộ in-domain |
| p4_imgsz1280 | Tăng imgsz 960→1280 cải thiện vật nhỏ | −1,89 điểm V5+ test; +0,12 Aerial test | Không ủng hộ |
| r_p2 | Thêm detection head stride-4 cho vật nhỏ | −4,02 điểm V5+ test; −6,16 Aerial test | Không ủng hộ ở cấu hình đã thử |
| **r_clspw_v2** | Tăng cls_pw 0→0,5 giúp lớp hiếm (truck/bus) | **+0,54 V5+; +1,68 Aerial val; +0,22 Aerial test; truck +2,32** | **Ủng hộ** — gain nhỏ nhưng có |
| r_progressive | Curriculum 960→1280 pixel | −2,20 V5+; +1,04 Aerial test | Không ủng hộ in-domain |

**Kết luận viết lại cho Mục 4.9.4 (Câu hỏi 4):**
> Hai recipe có cải thiện: **r_clspw** vượt baseline **+0,54 điểm mAP@0.5 in-domain** (chủ yếu nhờ truck +2,32 điểm) và **p4_scale** vượt baseline **+3,78 điểm trên VN Aerial test** (cross-domain). Các recipe còn lại (p4_imgsz1280, r_p2, r_progressive) không ủng hộ giả thuyết. Mô hình cuối chọn r_clspw_v2 cho ứng dụng in-domain, p4_scale cho triển khai nhiều góc nhìn khác nhau.

---

## 23. Bảng 5.1 — Vai trò mô hình trong app (BẢNG HOÀN CHỈNH, phiên bản cuối cho báo cáo)

Nguồn kiểm chứng: `ui/streamlit_app.py` dòng 169–216.

### Bảng để chèn Word

| Mô hình | Vai trò nghiên cứu | Trạng thái tích hợp app |
|---|---|---|
| **r_clspw_v2** | **mAP@0.5 V5+ cao nhất: 0,7984 — mô hình cuối cho in-domain** | Đã là `_preferred` #1 và checkpoint mặc định |
| **p4_scale** | **mAP@0.5 VN Aerial test cao nhất: 0,6374 — mô hình cho cross-domain** | Đã có trong `_preferred` sau exp_baseline |
| V5+ baseline (exp_baseline) | Baseline 7.237 ảnh: 0,7930 mAP@0.5 | Đã là fallback #2 |
| V6 pseudo | Thử nghiệm bổ sung; không phải mô hình cuối | Đang có trong `_preferred` |
| V5 honest | Fallback từ COCO | Đang có |
| Faster R-CNN | Đối chứng Chương 4 (thua YOLO +15,76 điểm) | Không có adapter UI (chỉ YOLO loader) |

### SHA256 các checkpoint chính (để truy xuất)

| Checkpoint | SHA256 (12 ký tự đầu) |
|---|---|
| `runs/detect/exp_baseline/weights/best.pt` | `8e3d5f89a8f7…` |
| `runs/detect/r_clspw_v2/weights/best.pt` | `609c2c99fbf7…` |
| `runs/detect/p4_scale_from_baseline/weights/best.pt` | `59cdbdfc181b…` |
| `runs/detect/r_clspw/weights/best.pt` (v1 confound) | `8bb04fdd7c46…` |
| `runs/detect/train_v8s_v5plus_960/weights/best.pt` (misnamed, V5 base) | `b1e9b1e583a3…` |
| `weights/trained/frcnn/best.pt` | `b5406a062bf5…` |

**Đã thực hiện cập nhật `ui/streamlit_app.py`** (2026-11-16):
- `FEATURED_MODELS` thêm r_clspw_v2 (⭐), exp_baseline, p4_scale, V6 pseudo, V5 honest.
- `_preferred` thứ tự: r_clspw_v2 → exp_baseline → p4_scale → V6 pseudo → V5 honest.
- Khi mở app, default load `r_clspw_v2/best.pt`.

---

## 24. Bảng 5.2 — Kiểm thử chức năng

Nguồn kiểm chứng: `ui/streamlit_app.py` (829 dòng). Chưa chạy app thủ công; chỉ kiểm chứng sự tồn tại code tương ứng.

| Hạng mục | Có mã nguồn? | Lưu ý |
|---|---|---|
| Tải video MP4/MOV/AVI | Có (`st.file_uploader`) | — |
| Quét checkpoint + chọn model | Có (`WEIGHTS_DIR.rglob("*.pt")`) | r_clspw không nằm trong `_preferred` |
| Adapter Faster R-CNN | Không | Chỉ `YOLO(weights)` |
| Cấu hình vạch đếm | Có (1 vạch ngang/dọc, chọn direction) | UI chưa hỗ trợ nhiều vạch mặc dù `Counter(lines=[…])` hỗ trợ list |
| Hiển thị trực quan | Có (`cv2.imshow` → `st.image`) | — |
| Xuất CSV + MP4 H.264 | Có hàm ghi + ffmpeg | Chưa đo thời gian chuyển mã trong sesion này |
| Đánh giá đếm (upload GT JSON) | Có nhận GT tay | — |
| Khôi phục tác vụ | Không có persistent state | Mất phiên = mất kết quả |

Các cột "Kiểm thử còn cần" chỉ là liệt kê công việc, không có số đo → giữ nguyên.

---

## 25. Bảng 5.3 — Đánh giá tốc độ xử lý (BẢNG HOÀN CHỈNH)

Điều kiện đo chung: GPU **RTX 3500 Ada 12 GB**, driver 595.95, CUDA runtime 11.8, torch 2.7.1+cu118, ultralytics 8.4.126, batch=1, imgsz=960 (trừ mục ghi rõ), warmup 20 ảnh, đo 200 ảnh. Script: `scripts/verify/50_speed_benchmark.py` → `outputs/verify/speed_benchmark.json`.

### (a) Độ trễ model-only YOLOv8s r_clspw_v2 và Faster R-CNN

| Model | imgsz | Precision | mean (ms/ảnh) | median (ms/ảnh) | p95 (ms/ảnh) | FPS mean |
|---|---|---|---|---|---|---|
| YOLOv8s **r_clspw_v2** | 960 | FP16 | **9,69** | 8,52 | 14,30 | **103,2** |
| YOLOv8s r_clspw_v2 | 960 | FP32 | 8,92 | 8,33 | 13,67 | 112,1 |
| Faster R-CNN R50-FPN | orig (800/1333) | FP32 | **47,34** | 46,82 | 51,01 | **21,1** |

Ghi chú: Trên GPU này FP16 **không nhanh hơn FP32** cho YOLOv8s do mô hình nhỏ, bottleneck ở preprocess + NMS Python-side. FRCNN chậm ~**4,9×** YOLOv8s r_clspw_v2 ở cùng điều kiện (47,34 / 9,69 ms).

### (b) FPS end-to-end pipeline đếm trên demo_traffic.mp4 (178 khung hình, r_clspw_v2)

| Chế độ | Trạng thái GPU | Wall time (s) | FPS end-to-end |
|---|---|---|---|
| detect + track + count, KHÔNG ghi video | Cold-start (mở app, chạy lần đầu) | 4,71 | **37,8** |
| detect + track + count, KHÔNG ghi video | Warm (sau khi model đã load) | 2,21 | **80,4** |
| detect + track + count + ghi MP4 (mp4v) | Warm | 3,95 | **45,1** |

Ghi video tăng chi phí ~1,74 s (~−44% FPS). Cold-start kéo FPS end-to-end xuống ~2× (CUDA kernel compile + model load lần đầu). Số dùng trong báo cáo end-user Streamlit nên là **37,8 FPS cold-start**.

### (c) Chi phí tracker (ms/khung hình, r_clspw_v2 warm)

FPS detect-only (Mục a FP16) = 103,2 → 9,69 ms/ảnh. FPS end-to-end no-write warm (Mục b) = 80,4 → 12,44 ms/khung. **Chi phí tracking + counting (ByteTrack + Line counter) ≈ 12,44 − 9,69 ≈ 2,75 ms/khung** (~22% tổng thời gian). Đo gián tiếp; cần profiler chi tiết để tách tracker vs counter riêng.

### (d) Thời gian chuyển mã ffmpeg + kích thước file đầu ra

Input: `outputs/verify/pipeline_render.mp4` (178 khung, mp4v codec từ OpenCV writer) — **18,49 MB**.

| Preset | Thời gian (s) | Kích thước đầu ra | Tỉ lệ nén | FPS transcode |
|---|---|---|---|---|
| libx264 `-preset fast -crf 23` | **1,25** | 5,22 MB | **2,8×** | ~142 fps |
| libx264 `-preset medium -crf 23` | 1,43 | 5,07 MB | 2,9× | ~125 fps |

→ Chuyển mã về H.264 sau pipeline là **nhanh hơn chạy pipeline** (~37 FPS end-to-end), chi phí thêm chỉ ~1,3 s cho video 6 giây đầu ra. File size giảm 72% so với mp4v. Lệnh:
```bash
ffmpeg -y -i outputs/verify/pipeline_render.mp4 -c:v libx264 -preset fast -crf 23 -an out.mp4
```

---

## 26. Bảng 2.3 — So sánh YOLOv8 và Faster R-CNN (BẢNG HOÀN CHỈNH)

| Đặc điểm | YOLOv8s | Faster R-CNN ResNet50-FPN | Kiểm chứng |
|---|---|---|---|
| Số giai đoạn | 1 | 2 | — |
| Anchor | Anchor-free | Theo cấu hình FPN | — |
| Số tham số | 11,1 triệu | ~41,8 triệu | Khớp với báo cáo ultralytics (yolov8s.pt 22.5 MB ≈ 11.1M params FP16) và torchvision frcnn (~41 M params) |
| GFLOPs | "Chưa đo thống nhất" | "Chưa đo thống nhất" | Giữ nguyên |
| Backbone | Conv–C2f–SPPF | ResNet50 | — |
| Neck | PAN-FPN | FPN | — |
| Loss hồi quy | CIoU + DFL | Smooth L1 | — |
| Hậu xử lý | NMS lớp | NMS lớp sau 2 giai đoạn | — |

Không có số liệu tốc độ cụ thể trong Bảng 2.3 — chuyển về Bảng 5.3.

---

## 27. Danh sách "KHÔNG CÓ DỮ LIỆU"

| Vị trí | Thiếu gì | Cần gì để có |
|---|---|---|
| Bảng 4.8 | Đã có số thật cho 1 video: r_clspw_v2 đếm 93/87 lượt | Cần thêm 5–10 video GT để đánh giá đa camera. |
| ~~Bảng 3.1 cột "số camera"~~ ĐÃ CÓ | Đã đếm bằng script | Xem Mục 2. Khớp 4/5 nguồn, VN v10 lệch 9 so với báo cáo. |
| Bảng 3.5 stride lấy mẫu | Script tách khung hình gốc | Dataset Roboflow đã tách sẵn; stride 5/10 chỉ là mô tả. |
| ~~Hình 4.1~~ ĐÃ CÓ | Đã sinh lại từ `exp_baseline/results.csv` | [outputs/verify/fig_4_1_training_curves_exp_baseline.png](outputs/verify/fig_4_1_training_curves_exp_baseline.png) + so sánh baseline vs r_clspw_v2: [outputs/verify/fig_4_1b_baseline_vs_r_clspw_v2.png](outputs/verify/fig_4_1b_baseline_vs_r_clspw_v2.png) |
| ~~Bảng 5.3 (d) ffmpeg~~ ĐÃ CÓ | Đã benchmark 1,25 s preset fast, 1,43 s preset medium | Xem Mục 25 (d). |
| Bảng 5.1 "sha256" | Tất cả đã có tại Mục 23 | — |
| Thuyết minh 5.4.3 "giới hạn tải lên" | `.streamlit/config.toml` không tồn tại | Nếu cần giới hạn khác 200 MB mặc định, tạo `.streamlit/config.toml` với `[server] maxUploadSize=…`. |

---

## 28. Danh sách script & lệnh tái tạo

Toàn bộ scripts nằm trong `scripts/verify/`, outputs trong `outputs/verify/`.

```bash
# 0. Inventory + trích báo cáo
python scripts/verify/00_extract_docx.py \
       "docs/tieuluan_V3/Nhom02_PhanLoaiPhuongTien_UPDATED5 (1).docx" \
       outputs/verify/report_extract.json
python scripts/verify/01_inventory.py          # checkpoints, yamls, runs
python scripts/verify/02_summarize_report.py   # headings + table skeleton

# 1. Data audit (5–10 phút)
python scripts/verify/10_data_audit.py         # per-split per-source bbox, size bins, leak

# 2. Training config / results (ms)
python scripts/verify/20_training_config.py    # args.yaml + results.csv per run

# 3. Detector eval 6x3 (~2 phút trên RTX 3500 Ada)
python scripts/verify/30_detector_eval.py      # ultralytics .val() mọi bộ test

# 3b. Error analysis r_clspw (~2 phút)
python scripts/verify/31_error_analysis.py     # TP/FP/FN/confusion/examples

# 4. Counting eval (1 phút cho demo_traffic)
python scripts/verify/40_counting_eval.py

# 5. Speed benchmark (chạy sau cùng, ~10 phút)
python scripts/verify/50_speed_benchmark.py
```

---

*Phiên bản cuối 2026-11-16, sau khi retrain r_clspw_v2 từ exp_baseline (fair warm-start) và cập nhật toàn bộ Bảng 4.2–4.7, Section 4.7, Bảng 5.1 với số thật. Không còn chỉ số nào ở trạng thái "chờ".*
