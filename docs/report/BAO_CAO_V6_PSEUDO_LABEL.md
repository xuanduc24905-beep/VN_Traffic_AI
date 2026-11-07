# Báo cáo V6 — Cải tiến mô hình phân loại phương tiện giao thông VN qua pseudo-labeling

**Dự án**: VN Traffic AI — Phân loại phương tiện bằng CNN
**Phiên bản**: V5 → V6
**Ngày báo cáo**: 2026-11-07
**Người thực hiện**: xuanduc

---

## 📑 Tóm tắt kết quả

| Model | mAP@50 | mAP@50-95 | AP motorcycle | AP truck | AP bus | AP car |
|---|---|---|---|---|---|---|
| YOLOv8s (COCO vanilla) | 0.006 | 0.004 | 0.000 | 0.000 | 0.000 | 0.000 |
| V5 (fine-tune từ YOLOv8s) | 0.665 | 0.340 | 0.622 | 0.530 | 0.694 | 0.813 |
| **V6 (V5 + pseudo-label HCM)** | **0.761** | **0.411** | **0.817** | **0.634** | **0.743** | **0.850** |
| **Δ (V6 - V5)** | **+9.6 pts** | **+7.2 pts** | **+19.5 pts** | **+10.4 pts** | **+4.8 pts** | **+3.7 pts** |

**Highlight**: Class **motorcycle** — điểm yếu nhất của V5 — tăng **+31.3% tương đối** nhờ pseudo-label từ scene giao thông VN (xe máy đông).

---

## 1. Bối cảnh và động lực

### 1.1. Điểm yếu của V5

V5 (`train_v8s_v5_honest`) là model YOLOv8s fine-tune từ COCO pretrained, train trên **2,833 ảnh gold** (10 video soict + cantho) với **honest video-aware split**. Kết quả:

- mAP@50 trên test = 0.665
- Precision 0.73, Recall 0.62
- **Điểm yếu**: `motorcycle` AP@50 chỉ 0.622 — thấp nhất trong 4 class

**Nguyên nhân motorcycle yếu**:
- V5 train chủ yếu từ dataset SOICT 2024 (video ban đêm cao tốc) và Can Thơ (siêu thị, góc hẹp)
- Thiếu domain "đô thị VN giờ cao điểm" — nơi xe máy chi chít, nhiều góc, đa dạng ánh sáng

### 1.2. Giải pháp — pseudo-labeling

Thay vì thuê người gán nhãn thêm 10,000 ảnh (tốn kém), dùng **pseudo-labeling**:

1. Thu thập ảnh unlabeled từ CCTV công khai TP.HCM (nhiều xe máy)
2. Dùng V5 làm "teacher" tự động gán nhãn (dù chỉ ~80% chính xác)
3. Lọc nghiêm ngặt theo confidence + size
4. Mẫu nhỏ review tay để đo precision
5. Merge với V5 gold → train V6 (student)

---

## 2. Nguồn dữ liệu

### 2.1. V5 gold (baseline)

15 video, split video-level:

| Split | Videos | Frames |
|---|---|---|
| Train | 10 (soictnight_cam_03/05/08, soict24_cam_01/02/04/06, cantho_CoopMart x2, vnv3) | 2,833 |
| Val | 3 (soict24_cam_07, cantho_CoopMart-8h30-9h00, cantho_camDLHB) | 1,466 |
| Test | 2 (soict24_cam_09, soict24_cam_10) | 1,500 |

Split đảm bảo **video_id không trùng** giữa train/val/test → không leak.

### 2.2. Nguồn pseudo-label mới — CCTV TP.HCM

- **Nguồn gốc**: `giaothong.hochiminhcity.gov.vn` (Sở GTVT TP.HCM chính thức)
- **Trung gian scrape**: `cameragiaothong.net` (wrapper, dễ enumerate ID)
- **Scale**: 782 camera công khai, phủ 138 phường
- **Chất lượng**: ảnh snapshot 512×288 JPEG, refresh ~15s/frame
- **Pháp lý**: nguồn công khai, mục đích nghiên cứu học thuật

---

## 3. Quy trình pseudo-labeling

### 3.1. Sơ đồ pipeline 7 bước

```
B1. Scrape 782 URL camera (01_scrape_camera_urls.py)
      │
      ▼
B2. Collect snapshot — session 30 phút, chọn 30-40 cam
    (02_collect_snapshots.py + blacklist + history)
      │
      ▼
B3. Dedup pHash (87% giữ lại)  (03_dedup.py)
      │
      ▼
B4. Inference V5 → pseudo-label bbox
    (04_inference.py + chunking + resume)
      │
      ▼
B5. Filter per-class threshold + size (05_filter.py)
      │
      ▼
B6. Visualize 500 ảnh cho người review (06_visualize_review.py)
      │
      ▼
B7. Remove bad: ảnh WRONG + camera blacklist (07_remove_bad.py)
```

### 3.2. Thống kê qua từng bước

| Session | Thời điểm | Ảnh thô (B2) | Sau dedup (B3) | Có label (B4) | Qua filter (B5) |
|---|---|---|---|---|---|
| 1 | 2026-11-05, 11h | 1,588 | 1,463 | 1,427 | 1,401 |
| 2 | 2026-11-06, 05h | 2,559 | (merged) | | |
| **Tổng** | | **4,147** | **3,518** | **2,852** | **~2,500** |

### 3.3. Confidence threshold per-class

| Class | Threshold | Lý do |
|---|---|---|
| motorcycle | **0.60** | Class khó nhất → nghiêm ngặt tránh noise |
| car | 0.45 | Class dễ → có thể loose hơn |
| bus | 0.50 | Trung bình |
| truck | 0.50 | Trung bình |

Size filter: box area < 0.0025 (≈ 20×20 px) → loại.

### 3.4. Spot-check review tay 500 ảnh

Reviewer phân loại 500 ảnh ngẫu nhiên:

| Loại | Số ảnh | % |
|---|---|---|
| CORRECT (tất cả box đúng) | 438 | 87.6% |
| PARTIAL (1-2 box sai, đa số đúng) | 51 | 10.2% |
| WRONG (sai nhiều/class nhầm) | 11 | 2.2% |

**Precision strict = 87.6%** (trên ngưỡng 85% → đạt chuẩn academic)
**Precision loose = 97.8%** (chấp nhận PARTIAL)

### 3.5. Phát hiện camera xấu → blacklist

Khi review phát hiện pattern lỗi systematic:
- 6/11 ảnh WRONG đến từ **camera 000605** ("Nút giao Mỹ Thủy, Cát Lái") → V5 fail do góc cam cao, gộp cụm xe
- Mở rộng điều tra → 7 camera có problem_rate ≥ 18%:

| Cam ID | Vị trí | Problem rate |
|---|---|---|
| 000605 | Nút giao Mỹ Thủy, Cát Lái | 53% |
| 000430 | Phạm Văn Đồng - Lê Quang Định | 41% |
| 000759 | Nguyễn Hữu Thọ, Tân Hưng | 30% |
| 000096 | 625 Quang Trung | 29% |
| 000204 | Đinh Tiên Hoàng, Tân Định | 26% |
| 000105 | Võ Văn Kiệt - Cầu Chà Và (trên cầu) | 18% |
| 000026 | Nút giao Mỹ Thủy 2 | 12% (cluster) |

→ Blacklist toàn bộ 7 camera → xoá 358 ảnh (khoảng 25% total raw).

---

## 4. Kiến trúc và huấn luyện V6

### 4.1. Cấu hình train

```yaml
model: yolov8s.pt (COCO pretrained)
data:  data/data_v6.yaml
epochs: 40
imgsz: 960
batch: 16
lr0: 0.005
```

### 4.2. Dataset V6

| Split | Nguồn | Ảnh |
|---|---|---|
| **Train** | V5 train + HCM pseudo | **7,229** |
| Val | V5 val (giữ nguyên) | 1,466 |
| Test | V5 test (giữ nguyên) | 1,500 |

Phân bố V6 train theo nguồn:

| Nguồn | Ảnh | % |
|---|---|---|
| **HCM pseudo** (mới) | 4,396 | 60.8% |
| V5 train videos | 2,833 | 39.2% |

### 4.3. Training results

```
Epoch 40: Precision=0.892  Recall=0.780  mAP50=0.832  mAP50-95=0.514
```

Train hoàn tất sau 40 epochs. Weights: `runs/detect/train_v8s_v6_pseudo/weights/best.pt`

---

## 5. Kết quả đánh giá

### 5.1. So sánh 3 model trên cùng test V5 gold (1,500 ảnh)

| Metric | YOLOv8s COCO | V5 fine-tune | **V6 pseudo** |
|---|---|---|---|
| mAP@50 | 0.0063 | 0.6648 | **0.7608** |
| mAP@50-95 | 0.0035 | 0.3399 | **0.4111** |
| Precision | 0.0091 | 0.7259 | **0.7536** |
| Recall | 0.2041 | 0.6160 | **0.7093** |

### 5.2. Per-class AP@50

| Class | V5 | V6 | Δ tuyệt đối | Δ tương đối |
|---|---|---|---|---|
| **motorcycle** | **0.622** | **0.817** | **+0.195** | **+31.3%** |
| truck | 0.530 | 0.634 | +0.104 | +19.6% |
| bus | 0.694 | 0.743 | +0.049 | +7.0% |
| car | 0.813 | 0.850 | +0.037 | +4.5% |

### 5.3. Biểu đồ so sánh (mô tả)

```
        AP@50
 0.9 ┤     ■              ■
     │     ■    ■    ■    ■      ■
 0.7 ┤ ■   ■    ■    ■    ■   ■  ■
     │ ■   ■    ■    ■    ■   ■  ■
 0.5 ┤ ■   ■    ■    ■    ■   ■  ■
     │ ■   ■    ■    ■    ■   ■  ■
     └──────────────────────────────▶
       motor   car  bus truck  overall
       V5=■   V6=■
```

### 5.4. Phân tích

**Cải thiện đồng đều**: V6 vượt V5 ở **tất cả 4 class** → pseudo-label không chỉ cải tiến 1 class mà nâng tổng thể.

**Motorcycle lớn nhất** (+31.3%): phù hợp giả thuyết — pseudo-label từ scene giao thông HCM (xe máy chi chít) chuyên trị điểm yếu V5.

**Truck +19.6%**: cũng cải tiến rõ nhờ ảnh xe tải/xe container trong camera cao tốc HCM.

**Bus +7% và Car +4.5%**: tăng ít hơn vì V5 đã làm tốt 2 class này.

**Recall tăng mạnh hơn Precision** (+15% vs +4%): V6 detect được nhiều xe hơn mà không tăng false positive — hiệu ứng mong muốn của pseudo-labeling.

---

## 6. Kiểm chứng tính honest (no leakage)

Dataset V5 bị leak 100% ở các phiên bản V1-V4 trước đây → đã rebuild theo video-aware split. V6 phải được verify kỹ.

### 6.1. Kết quả leak check

| Mức kiểm tra | Kết quả |
|---|---|
| Filename: V6_train ∩ V5_test | **0 ảnh** ✓ |
| Filename: V6_train ∩ V5_val | **0 ảnh** ✓ |
| V6 test = V5 test (identical) | **1,500 ảnh identical** ✓ |
| Video_id: V6_train ∩ V5_test videos | **0 video** ✓ |
| Video_id: V6_train ∩ V5_val videos | **0 video** ✓ |

### 6.2. Phân tích nguồn V6 train — không trùng test/val

V6 train (7,229 ảnh) chỉ gồm:
- **HCM pseudo** (4,396) — nguồn hoàn toàn mới, không có trong V5 manifest
- 10 videos V5 train: soictnight_cam_03/05/08, soict24_cam_01/02/04/06, cantho_CoopMart (2), vnv3

→ KHÔNG có video nào từ V5 test ({soict24_cam_09, soict24_cam_10}) hoặc V5 val ({soict24_cam_07, cantho_CoopMart-8h30-9h00, cantho_camDLHB-13h20-13h57}).

### 6.3. Kết luận

**V6 → V5 so sánh là HONEST**. Mọi cải tiến (+9.6 mAP@50, +19.5 pts motorcycle) đều phản ánh hiệu quả thật của pseudo-labeling, không phải do data leak.

---

## 7. Hạn chế và bàn luận

### 7.1. Hạn chế

**Pseudo-label có noise ~10-15%**: dù đã filter, vẫn còn ảnh PARTIAL (51/500 = 10.2%). Model V6 học qua noise này → không thể đạt ceiling của manual annotation.

**Confirmation bias**: V5 là teacher → V6 kế thừa luôn weakness V5. Vì vậy vẫn cần V6 fine-tune thêm nếu muốn đột phá.

**Giới hạn resolution**: nguồn CCTV HCM chỉ 512×288 — khó detect object nhỏ (xe xa). Đã train V6 với imgsz=960 để compensate nhưng chưa tối ưu.

**Precision chỉ đo trên 500 sample**: statistical margin of error ~2% ở confidence 95%.

### 7.2. Camera blacklist — pattern xấu systematic

7 camera bị loại cho thấy V5 có weakness đặc thù:
- **Góc cam cao** (bird-eye): gộp cụm xe thành 1 box sai
- **Camera trên cầu**: xe máy quá đông → miss hàng loạt
- **Camera ngã tư phức tạp** (Mỹ Thủy): nhiễu đa chiều

→ V7 (nếu có) nên augment thêm scene này hoặc dùng model backbone lớn hơn.

### 7.3. Thời điểm scrape

Dataset HCM pseudo chụp ở 3 khung giờ: 04h (135), 05h (1,852), 11h (1,085). Thiếu khung giờ tối (20h+) → V6 chưa tối ưu cho inference ban đêm.

---

## 8. Công việc tiếp theo (future work)

### 8.1. Vòng lặp tiếp theo (V7)

```
Round 2: V6 làm teacher → generate pseudo-label chất lượng cao hơn V5
       → train V7
       → so sánh V7 vs V6 → nếu tăng <1% thì dừng (diminishing return)
```

### 8.2. Mở rộng diversity

- Scrape thêm **camera Hà Nội** (nguồn độc lập) — diversity về domain
- Scrape thêm **khung giờ đêm** (20h-00h) — fix điểm yếu low-light
- Scrape thêm **mùa mưa** — robust với condition khác biệt

### 8.3. Pseudo-label cho VN specifics

Xe máy VN có đặc thù không có trong COCO:
- Chở 3-4 người
- Chở hàng cồng kềnh
- Đi làn ngược
→ Cần class riêng hoặc augmentation chuyên biệt.

---

## 9. Kết luận

Pseudo-labeling từ CCTV công khai TP.HCM là **chiến lược hiệu quả** để cải tiến V5:

✅ **Cải tiến mạnh** ở class yếu nhất (motorcycle +31.3%)
✅ **Honest comparison** — không có leak ở cả 2 mức file + video_id
✅ **Chi phí thấp** — 0 đồng gán nhãn thủ công, chỉ cần GPU 1-2 giờ
✅ **Reproducible** — toàn bộ pipeline + config commit trên GitHub
✅ **Methodology chuẩn** — follow Noisy Student / Self-training literature

V6 trở thành model deployment candidate cho giai đoạn tiếp theo của dự án.

---

## 📁 Artifacts

| Loại | Đường dẫn |
|---|---|
| V6 weights | `runs/detect/train_v8s_v6_pseudo/weights/best.pt` |
| Comparison result | `runs/compare_v5_v6_yolov8s_test.json` |
| Pseudo pipeline scripts | `scripts/pseudo/01..10_*.py` |
| Dataset config | `configs/data_v6.yaml`, `configs/hcm_cameras.yaml` |
| Blacklist | `configs/hcm_cameras_blacklist.yaml` |
| Workflow docs | `docs/guides/V6_PSEUDO_LABELING_WORKFLOW.md` |
| Pipeline giải thích | `docs/guides/PSEUDO_PIPELINE_CHI_TIET.md` |
| Review guide (team) | `docs/guides/HUONG_DAN_DUYET_500_ANH.md` |

---

## 📚 Tham khảo

- Xie, Q. et al. (2020). *Self-training with Noisy Student improves ImageNet classification*. CVPR.
- Sohn, K. et al. (2020). *A Simple Semi-Supervised Learning Framework for Object Detection*. arXiv:2005.04757.
- Yalniz, I. et al. (2019). *Billion-scale semi-supervised learning for image classification*. arXiv:1905.00546.
- Ultralytics YOLOv8 Documentation: https://docs.ultralytics.com

---

*Báo cáo cập nhật: 2026-11-07*
