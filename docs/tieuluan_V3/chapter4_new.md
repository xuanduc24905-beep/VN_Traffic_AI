# CHƯƠNG 4  THỰC NGHIỆM VÀ ĐÁNH GIÁ

Chương này trình bày toàn bộ kết quả thực nghiệm của đề tài, bao gồm môi trường triển khai, kết quả huấn luyện, so sánh giữa các cấu hình YOLOv8s, so sánh YOLOv8s với mô hình đối chứng Faster R-CNN ResNet50-FPN, ablation study và phân tích lỗi. Mục tiêu chính là trả lời năm câu hỏi nghiên cứu đã đặt ra ở Chương 1 bằng bằng chứng thực nghiệm định lượng.

## 4.1 Môi trường và cấu hình thực nghiệm

### 4.1.1 Phần cứng

- **GPU**: NVIDIA RTX 3500 Ada Generation Laptop (12 GB VRAM, kiến trúc Ada Lovelace)
- **CPU**: Intel Core i9-13900H (14 cores, 20 threads)
- **RAM**: 32 GB DDR5
- **Lưu trữ**: NVMe SSD 1 TB (tốc độ đọc tuần tự ~3,5 GB/s — đủ để không trở thành bottleneck khi training)

### 4.1.2 Phần mềm

- **OS**: Linux Ubuntu 22.04 (WSL2 trên Windows 11)
- **Python**: 3.10.20
- **PyTorch** [29]: 2.5.1 + CUDA 12.4
- **Ultralytics** [6]: 8.4.142
- **TorchVision**: 0.20.1
- **torchmetrics**: 1.5
- **OpenCV**: 4.10
- **Streamlit**: 1.38

### 4.1.3 Thiết lập tái lập

Mỗi lần huấn luyện và đánh giá đều được gắn với:
- **Git commit hash** của mã nguồn
- **Tệp YAML cấu hình dataset** (ví dụ `data/data_v5plus.yaml`)
- **Tệp manifest split** (`data/merged_v5plus/split_manifest.json`) ghi chính xác video nào thuộc tập nào
- **Seed cố định**: `seed=42` cho Python, NumPy, PyTorch và CUDA
- **Checkpoint trọng số** (`best.pt` và `last.pt`) kèm **results.csv** chi tiết từng epoch

Tất cả các kết quả trong chương này có thể tái lập từ mã nguồn tại thư mục `scripts/` và dữ liệu tại `data/merged_v5plus/`.

### 4.1.4 Danh mục mô hình thực nghiệm

**Bảng 4.1** tổng hợp danh mục các mô hình đã huấn luyện và đánh giá trong đề tài.

| Mã mô hình | Kiến trúc | Dữ liệu train | imgsz | Init weights | Mô tả |
|---|---|---|---|---|---|
| COCO-ZS | YOLOv8s | (không train) | 960 | yolov8s.pt | Baseline zero-shot, chỉ map 4 class COCO sang 4 class VN |
| V5-honest | YOLOv8s | V5 (2.833 ảnh) | 640 | yolov8s.pt | Baseline minimal, config đơn giản |
| V5+ | YOLOv8s | V5+ (7.237 ảnh) | 960 | yolov8s.pt | Baseline V5+ với mixup, cos_lr |
| p4_scale | YOLOv8s | V5+ | 960 | V5+ best.pt | Finetune với scale=0.7, translate=0.15 (mô hình cuối) |
| p4_imgsz1280 | YOLOv8s | V5+ | 1280 | V5+ best.pt | Finetune với imgsz lớn hơn (ablation) |
| V6-pseudo | YOLOv8s | V6 pseudo (11.633 ảnh) | 960 | V5+ best.pt | Thử nghiệm self-training pseudo-labeling |
| FRCNN | Faster R-CNN ResNet50-FPN | V5+ | 800–1333 | COCO pretrained | Mô hình đối chứng |

## 4.2 Kết quả huấn luyện

### 4.2.1 Đường cong loss và mAP theo epoch

Mỗi mô hình YOLOv8s được train trong 40 epoch (trừ Faster R-CNN train 30 epoch) với cơ chế early stopping khi mAP@0.5 trên validation không cải thiện trong 15 epoch liên tiếp (`patience=15`).

**Hình 4.1** Đường cong loss và mAP theo epoch của mô hình V5+ 960px. Trục ngang là số epoch (0–40), các trục dọc lần lượt là box_loss, cls_loss, dfl_loss (giảm dần) và precision, recall, mAP@0.5, mAP@0.5:0.95 (tăng dần).

Quan sát từ đường cong training:
- **V5 honest** hội tụ nhanh, mAP@0.5 val đạt ~0.70 sau epoch 15 và plateau quanh 0.72 đến epoch 40.
- **V5+ 960px** cần nhiều epoch hơn để hội tụ do imgsz lớn hơn, mAP@0.5 val đạt ~0.80 ở epoch 25 và plateau quanh 0.86 ở epoch 35–40.
- **p4_scale** (warm-start từ V5+) hội tụ nhanh trong 10 epoch đầu do đã có kiến thức V5+, sau đó cải thiện chậm hơn do augmentation mạnh. mAP@0.5 val đạt 0.855 ở epoch 30.
- **Faster R-CNN** hội tụ nhanh về mAP@0.5 val ~0.93 ở epoch 15, plateau ở epoch 25–30 (lưu ý: val mAP của FRCNN chênh lệch lớn với test do khác biệt distribution giữa val và test).

### 4.2.2 Thời gian huấn luyện thực tế

**Bảng 4.2** Thời gian huấn luyện thực tế của các mô hình trên GPU RTX 3500 Ada 12 GB.

| Mô hình | Epochs chạy | Thời gian | Thời gian/epoch |
|---|---|---|---|
| V5 honest (640) | 40 | 22 phút | 33 giây |
| V5+ 960px | 40 | 1 giờ 48 phút | 2 phút 42 giây |
| p4_scale (finetune) | 40 | 1 giờ 55 phút | 2 phút 53 giây |
| p4_imgsz1280 (finetune) | 40 | 2 giờ 30 phút | 3 phút 45 giây |
| V6 pseudo | 40 | 2 giờ 50 phút | 4 phút 15 giây |
| Faster R-CNN | 30 | 4 giờ 25 phút | 8 phút 50 giây |

Faster R-CNN mất thời gian gần gấp 3× YOLOv8s do kiến trúc hai giai đoạn phức tạp hơn, GFLOPs cao hơn (~134 vs 28,6) và không có fused optimization của Ultralytics.

## 4.3 So sánh YOLOv8s và Faster R-CNN trên tập test V5+

### 4.3.1 Kết quả tổng hợp

Tất cả mô hình được đánh giá trên cùng tập test V5+ gồm 1.500 ảnh bằng `torchmetrics.detection.MeanAveragePrecision` để đảm bảo cùng cách tính mAP, tránh chênh lệch do thư viện.

**Bảng 4.3** So sánh các mô hình trên test V5+ (1.500 ảnh, 4 lớp).

| Mô hình | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| COCO-ZS (zero-shot) | 0,557 | ~0,28 | 0,61 | 0,51 |
| V5 honest (640) | 0,665 | 0,34 | 0,71 | 0,62 |
| **V5+ 960px** | **0,782** | **0,413** | **0,768** | **0,707** |
| **p4_scale** (cuối) | **0,787** | **0,423** | 0,759 | 0,732 |
| p4_imgsz1280 | 0,774 | 0,418 | 0,729 | 0,743 |
| V6 pseudo | 0,761 | 0,383 | 0,742 | 0,715 |
| **Faster R-CNN** | **0,641** | **0,361** | — | — |

**Nhận xét quan trọng:**
- Mô hình cuối **p4_scale** đạt mAP@0.5 = 0,787, vượt mô hình đối chứng Faster R-CNN (0,641) **14,6 điểm phần trăm** (tương đương +22,8%).
- Về mAP@0.5:0.95 (strict metric), p4_scale (0,423) vượt FRCNN (0,361) **6,2 điểm**, chứng tỏ YOLOv8s không chỉ phát hiện được nhiều mà còn **định vị chính xác hơn**.
- Tiến trình cải thiện của YOLOv8s qua ba giai đoạn: COCO-ZS → V5 honest → V5+ → p4_scale tương ứng **+10,8%, +11,7%, +0,5%** trên mAP@0.5.
- V6 pseudo (self-training) **không cải thiện** so với V5+ (0,761 < 0,782), cho thấy naive pseudo-labeling có thể làm xấu đi kết quả do khuếch đại nhiễu nhãn.

### 4.3.2 So sánh per-class AP

**Bảng 4.4** AP@0.5 theo từng lớp trên test V5+.

| Mô hình | motorcycle | car | bus | truck |
|---|---|---|---|---|
| V5+ 960px | 0,841 | 0,884 | 0,716 | 0,687 |
| **p4_scale** | **0,862** | 0,885 | 0,722 | 0,679 |
| p4_imgsz1280 | 0,862 | 0,889 | **0,742** | 0,605 |
| Faster R-CNN (mAP@0.5:0.95) | 0,269 | 0,511 | 0,373 | 0,289 |

Quan sát quan trọng:
- **motorcycle**: p4_scale cải thiện +2,1% so với V5+ (0,862 vs 0,841). Logic hợp lý vì `scale=0.7` giúp detect xe máy ở xa (kích thước nhỏ).
- **truck**: p4_imgsz1280 bị **drop -8,2%** so với V5+ (0,605 vs 0,687). Bài học: imgsz lớn hơn không phải lúc nào cũng tốt — xe tải lớn có thể bị cắt/méo khi augment ở imgsz cao.
- **bus**: p4_imgsz1280 vẫn là tốt nhất (+2,5%), nhưng không bù đắp được sự sụt giảm của truck.
- **Faster R-CNN** bị thua rõ rệt ở **motorcycle** (0,269 mAP@50-95), lớp đông nhất và nhỏ nhất, phản ánh hạn chế của FPN truyền thống trong việc phát hiện đối tượng nhỏ dày đặc.

### 4.3.3 So sánh val và test gap

**Bảng 4.5** Khoảng cách val–test của ba mô hình YOLOv8s chính.

| Mô hình | val mAP@0.5 | test mAP@0.5 | Gap (val - test) |
|---|---|---|---|
| V5+ 960px | 0,857 | 0,782 | **0,075** |
| p4_scale | 0,855 | 0,787 | **0,068** |
| p4_imgsz1280 | 0,852 | 0,774 | 0,078 |

**Insight**: p4_scale có gap val–test **nhỏ nhất** (0,068) → **tổng quát hóa tốt nhất**. Điều này củng cố giả thuyết rằng augmentation mạnh hơn (scale=0.7, translate=0.15) giúp mô hình giảm overfit lên tập val và robust hơn trên test.

### 4.3.4 So sánh tốc độ FPS

**Bảng 4.6** Tốc độ xử lý trên GPU RTX 3500 Ada 12GB, đo trên 100 ảnh test sau 20 ảnh warm-up.

| Mô hình | Precision | imgsz | Latency (ms/image) | FPS |
|---|---|---|---|---|
| YOLOv8s (V5+/p4_scale) | FP16 | 960 | 18 | **55,6** |
| YOLOv8s | FP32 | 960 | 26 | 38,5 |
| YOLOv8s | FP16 | 1280 | 32 | 31,3 |
| Faster R-CNN ResNet50-FPN | FP32 | 800/1333 | 185 | **5,4** |

YOLOv8s nhanh hơn Faster R-CNN **~10× trên cùng phần cứng**, phù hợp cho xử lý video thời gian thực (≥25 FPS). Faster R-CNN cho thấy rõ hạn chế về tốc độ do kiến trúc hai giai đoạn.

**Kết luận §4.3**: p4_scale là mô hình tốt nhất tổng thể (cao nhất mAP, generalize tốt nhất, tốc độ cao), vượt mô hình đối chứng Faster R-CNN cả về độ chính xác lẫn tốc độ.

## 4.4 Ablation study

### 4.4.1 Ablation về độ phân giải đầu vào

So sánh V5 honest (imgsz=640) với V5+ 960px cho thấy ảnh hưởng của imgsz:

| imgsz | mAP@0.5 test | AP motorcycle | AP bus | AP truck |
|---|---|---|---|---|
| 640 (V5 honest) | 0,665 | 0,72 | 0,59 | 0,51 |
| 960 (V5+) | 0,782 | 0,841 | 0,716 | 0,687 |
| **Δ** | **+11,7%** | +12,1% | +12,6% | +17,7% |

**Kết luận**: Tăng imgsz từ 640 lên 960 cải thiện **đồng đều tất cả các lớp**, gain mạnh nhất ở truck (+17,7%). Xe tải thường ở vị trí xa hoặc kích thước lớn — imgsz cao giúp cả hai trường hợp.

### 4.4.2 Ablation về augmentation

So sánh V5+ baseline (aug chuẩn) với p4_scale (scale=0.7, translate=0.15):

| Cấu hình | mAP@0.5 | mAP@0.5:0.95 | AP motorcycle | Gap val-test |
|---|---|---|---|---|
| V5+ (scale=0.5, translate=0.1) | 0,782 | 0,413 | 0,841 | 0,075 |
| p4_scale (scale=0.7, translate=0.15) | 0,787 | **0,423** | **0,862** | **0,068** |

**Kết luận**: Augmentation mạnh hơn mang lại cải thiện đồng đều trên cả ba trục (mAP, per-class motorcycle, generalization gap). Đặc biệt mAP@0.5:0.95 tăng +2,4% cho thấy mô hình định vị chính xác hơn.

### 4.4.3 Ablation về self-training (V6 pseudo)

Thử nghiệm bổ sung 4.396 ảnh pseudo-label từ CCTV HCM vào tập train (V6 pseudo):

| Dataset | Số ảnh train | mAP@0.5 test | Ghi chú |
|---|---|---|---|
| V5+ (không pseudo) | 7.237 | **0,782** | Baseline |
| V6 pseudo | 11.633 | 0,761 | **Fail**: drop -2,1% |

**Phân tích thất bại**: Pseudo labels từ CCTV HCM (precision ~87,6% khi review manually) đã làm **khuếch đại class imbalance** trong tập train. CCTV HCM chủ yếu là xe máy (>85%), khiến mô hình học quá mạnh về motorcycle và yếu hơn ở bus/truck. Đây là **bài học về naive self-training**: cần cân bằng phân bố lớp trước khi bổ sung pseudo labels.

### 4.4.4 Ablation về kích thước finetune

So sánh hai cách finetune từ V5+ best.pt:

| Cấu hình | imgsz | mAP@0.5 test | mAP@0.5:0.95 |
|---|---|---|---|
| V5+ baseline (chuẩn) | 960 | 0,782 | 0,413 |
| p4_scale (aug mạnh) | 960 | **0,787** | **0,423** |
| p4_imgsz1280 (imgsz lớn) | 1280 | 0,774 | 0,418 |

**Kết luận**: Chuyển sang imgsz=1280 **không giúp** so với 960 — thậm chí giảm mAP@0.5 (-0,8%). Chi phí tính toán tăng 1,78× nhưng kết quả xấu đi. Nguyên nhân có thể do augmentation mặc định được calibrate cho imgsz=960 không còn phù hợp ở imgsz cao hơn.

## 4.5 Kết quả trên tập ngoài phân phối (Đà Nẵng)

### 4.5.1 Mô tả tập OOD

Một video CCTV Đà Nẵng (29.195 khung hình, ~20 phút) được tách riêng làm tập ngoài phân phối (out-of-distribution — OOD) để đánh giá khả năng tổng quát hóa cross-domain. Video này không xuất hiện trong bất kỳ tập train/val/test của V5+, đến từ camera giám sát ở khu vực Đà Nẵng với các đặc điểm khác biệt so với dữ liệu V5+ (chủ yếu từ TP.HCM và Cần Thơ).

### 4.5.2 Đánh giá định tính

Do không có nhãn chi tiết bounding box cho toàn bộ video Đà Nẵng, đánh giá được thực hiện ở hai mức:
- **Đánh giá định tính**: chạy p4_scale trên video Đà Nẵng, xuất video chú thích và quan sát các phân đoạn mẫu mỗi 5 phút.
- **Đánh giá định lượng counting**: đếm tay số phương tiện qua vạch ảo trên 5 đoạn 1 phút (trình bày ở §4.6).

**Quan sát định tính** từ video chú thích:
- Mô hình phát hiện tốt **xe máy và ô tô con** (chiếm >90% số detection) với confidence ổn định ở mức 0,7–0,9.
- **Xe buýt** có nhiều detection hơn thực tế (false positive), có thể do nhầm với xe khách nhỏ hoặc xe van dài.
- **Xe tải** đôi khi bị bỏ sót (false negative) ở các cảnh nghiêng hoặc che khuất một phần.
- Chất lượng detection giảm nhẹ ở các phân đoạn ánh sáng chuyển tiếp (bình minh/hoàng hôn).

### 4.5.3 Mức suy giảm cross-domain

Vì chưa có nhãn chuẩn chi tiết cho Đà Nẵng, không thể báo cáo chính xác mức suy giảm mAP@0.5 cross-domain. Đây được ghi nhận là **hạn chế của đề tài** và đề xuất cho hướng phát triển: cần gán nhãn (ít nhất) 500 khung hình mẫu từ Đà Nẵng để có đánh giá OOD định lượng.

## 4.6 Kết quả đếm phương tiện

### 4.6.1 Quy trình thu thập ground truth

Đánh giá hệ thống đếm được thực hiện trên **5 đoạn video 1 phút** lấy từ các video test của V5+ (3 đoạn ngày từ SOICT 2024 và 2 đoạn đêm từ SOICT Night). Mỗi đoạn:
- Đặt một vạch đếm ngang qua làn đường chính
- **Hai thành viên nhóm độc lập** đếm số phương tiện từng lớp đi qua vạch
- Trường hợp chênh lệch được **thành viên thứ ba** xem lại và quyết định
- Các trường hợp không chắc chắn (xe bị che khuất hoàn toàn khi đi qua vạch) được **loại bỏ khỏi ground truth**

Tổng số phương tiện đã đếm tay: **412 lượt** (motorcycle 298, car 86, bus 11, truck 17).

### 4.6.2 So sánh số đếm hệ thống vs ground truth

**Bảng 4.7** So sánh số đếm hệ thống (p4_scale + ByteTrack) vs ground truth đếm tay trên 5 đoạn video.

| Lớp | Ground truth (tổng) | Hệ thống (tổng) | MAE/đoạn | MAPE |
|---|---|---|---|---|
| motorcycle | 298 | 282 | 3,2 | 5,4% |
| car | 86 | 82 | 0,8 | 4,7% |
| bus | 11 | 12 | 0,2 | 9,1% |
| truck | 17 | 15 | 0,4 | 11,8% |
| **Tổng** | **412** | **391** | **4,6** | **5,1%** |

**Nhận xét**:
- Lớp **motorcycle** và **car** (số đếm lớn) có MAPE thấp (~5%) — hệ thống đếm ổn định.
- Lớp **bus** và **truck** (số đếm nhỏ) có MAPE cao hơn (9–12%) nhưng chủ yếu do mẫu số nhỏ, không phản ánh sai số thực tế quy mô lớn.
- Tổng sai số ~5,1% được coi là chấp nhận được cho ứng dụng thống kê cơ cấu phương tiện.

### 4.6.3 So sánh ngày và đêm

**Bảng 4.8** So sánh chất lượng đếm theo điều kiện ánh sáng.

| Điều kiện | GT total | Hệ thống | MAPE |
|---|---|---|---|
| 3 đoạn ban ngày | 312 | 299 | 4,2% |
| 2 đoạn ban đêm | 100 | 92 | 8,0% |

Ban đêm MAPE cao gần gấp 2× ban ngày do:
- Phát hiện yếu hơn khi xe chỉ có đèn pha/đèn hậu thấy rõ (phần thân xe tối)
- Xe máy nhỏ dễ bị miss khi di chuyển qua khu vực ít ánh sáng
- ByteTrack đôi khi mất track do detection không liên tục

## 4.7 Phân tích lỗi

### 4.7.1 Confusion matrix

**Hình 4.2** Confusion matrix 4×4 của mô hình p4_scale trên test V5+. Trục hàng là lớp ground truth, trục cột là lớp dự đoán. Ô (i, j) ghi số detection có nhãn thật i được dự đoán là j.

Quan sát từ confusion matrix:
- Đường chéo chính tập trung các giá trị lớn → mô hình phần lớn phân loại đúng lớp.
- Nhầm lẫn chủ yếu ở cặp **truck ↔ bus** (xe tải nhỏ vs xe buýt nhỏ có hình dáng tương tự khi nhìn từ phía sau hoặc phía trước)
- Nhầm lẫn thứ hai ở cặp **car ↔ truck** (xe pickup có thùng hàng dễ nhầm với xe tải nhỏ)
- Ít nhầm **motorcycle ↔ các lớp khác** (hình dáng khác biệt rõ)

### 4.7.2 Phân loại lỗi theo kích thước object

**Bảng 4.9** Phân loại TP/FP/FN theo kích thước bounding box (chuẩn COCO).

| Kích thước | TP | FP | FN | Recall |
|---|---|---|---|---|
| Small (<32² px) | 1.205 | 318 | 420 | 74,1% |
| Medium (32²–96² px) | 2.890 | 412 | 311 | 90,3% |
| Large (≥96² px) | 745 | 95 | 42 | 94,7% |

**Insight**:
- Đối tượng **small** có recall thấp nhất (74,1%) — là nguyên nhân chính của FN tổng thể
- Augmentation `scale=0.7` của p4_scale đã cải thiện nhưng chưa giải quyết triệt để
- Hướng cải thiện: dùng kiến trúc P2 head (stride 4) hoặc multi-scale training

### 4.7.3 Phân loại lỗi theo điều kiện ánh sáng

**Bảng 4.10** mAP@0.5 phân theo nguồn dữ liệu (proxy cho điều kiện).

| Nguồn test | Số ảnh | mAP@0.5 | AP motorcycle |
|---|---|---|---|
| SOICT 2024 (ngày TP.HCM) | 780 | **0,821** | 0,885 |
| SOICT Night (đêm) | 420 | 0,742 | 0,815 |
| Cantho (ngày Cần Thơ) | 180 | 0,789 | 0,862 |
| VNv3 (hỗn hợp) | 120 | 0,768 | 0,833 |

Ban đêm là điều kiện khó nhất (mAP 0,742, giảm 8% so với ngày TP.HCM). Nguyên nhân:
- Độ tương phản thấp giữa phương tiện và nền
- Nhiễu từ đèn pha, đèn phanh khiến mô hình mất feature stable
- Kích thước object nhỏ do khoảng cách xa hơn (camera đêm thường góc rộng)

### 4.7.4 Ví dụ minh họa thất bại

**Hình 4.3** Các trường hợp False Negative điển hình của p4_scale trên test V5+:
(a) Xe máy nhỏ bị che khuất bởi xe tải phía trước — mô hình chỉ detect được xe tải.
(b) Nhóm 3 xe máy đi song song — mô hình chỉ detect 1 xe (lỗi dense scene).
(c) Xe tải ở vị trí bị cắt một phần ở biên ảnh — recall giảm do bbox incomplete.
(d) Xe máy chạy trong bóng cây — confidence < 0,25 nên bị lọc ra.

**Hình 4.4** Các trường hợp False Positive điển hình:
(a) Biển quảng cáo có hình xe → nhầm thành ô tô.
(b) Xe van dài → detect thành bus do backbone học hình dáng hộp dài.
(c) Phần tối trên đường → detect nhầm thành xe máy (low confidence).

## 4.8 Thảo luận — Trả lời các câu hỏi nghiên cứu

### Câu hỏi 1: So sánh YOLOv8s và Faster R-CNN

Trên cùng tập test V5+ gồm 1.500 ảnh với 4 lớp, mô hình **YOLOv8s (p4_scale) vượt trội Faster R-CNN** ở mọi chỉ số:
- mAP@0.5: 0,787 vs 0,641 (**+14,6%**)
- mAP@0.5:0.95: 0,423 vs 0,361 (**+6,2%**)
- Tốc độ: 55,6 FPS vs 5,4 FPS (**~10× nhanh hơn**)

YOLOv8s đạt lợi thế đặc biệt ở lớp motorcycle (lớp đông nhất, kích thước nhỏ nhất), nơi anchor-free và head decoupled tỏ ra hiệu quả hơn so với anchor-based của Faster R-CNN. Faster R-CNN mạnh hơn ở các lớp hiếm có kích thước lớn (bus, truck) theo mAP@0.5:0.95 (bus 0,373 vs YOLO 0,42, truck 0,289 vs YOLO 0,34 theo chuẩn mAP@50-95 per class) nhưng không đủ bù đắp sự thua kém ở các lớp đông.

### Câu hỏi 2: Ảnh hưởng của hyperparameter

Việc tăng độ phân giải từ 640 lên 960 cải thiện mAP@0.5 **+11,7%** (0,665 → 0,782), gain mạnh nhất ở truck (+17,7%). Áp dụng cosine learning rate schedule và augmentation không gian mạnh hơn (scale=0,7) cải thiện thêm **+0,5%** mAP@0.5 và **+2,4%** mAP@0.5:0.95, đồng thời **giảm gap val–test** (0,075 → 0,068) chứng tỏ cải thiện generalization.

### Câu hỏi 3: Phân tích lỗi

Lỗi chủ yếu xuất hiện ở:
- **Đối tượng nhỏ** (recall 74,1% so với 94,7% ở large) — nguyên nhân chính của FN
- **Nhầm lẫn truck ↔ bus** (hình dáng tương tự khi nhìn trực diện)
- **Điều kiện ban đêm** (mAP giảm 8% so với ban ngày)
- **Dense scene** với nhiều xe máy song song

### Câu hỏi 4: Đạt mục tiêu mAP≥0,75

Mô hình cuối **p4_scale đạt mAP@0.5 = 0,787 > 0,75**, **đạt mục tiêu**. Mức cải thiện so với COCO zero-shot là **+41,3%** tuyệt đối (0,557 → 0,787), so với V5 honest baseline là **+18,3%** (0,665 → 0,787).

### Câu hỏi 5: Hệ thống đếm

Hệ thống đếm đạt **MAPE trung bình 5,1%** trên 5 đoạn video kiểm thử (412 phương tiện đếm tay). Lỗi đếm tập trung ở **lớp motorcycle vào ban đêm** (MAPE 8,0% cho đêm vs 4,2% ngày) do mô hình detection yếu hơn trong điều kiện ánh sáng thấp. Các lớp hiếm (bus, truck) có MAPE cao hơn (9–12%) nhưng chủ yếu do mẫu số nhỏ, không phản ánh sai số thực tế quy mô lớn.

## 4.9 Tiểu kết chương

Chương này đã trình bày đầy đủ kết quả thực nghiệm:
- **Môi trường**: GPU RTX 3500 Ada 12GB, Ultralytics 8.4.142, PyTorch 2.5.1
- **Kết quả tổng hợp**: p4_scale đạt mAP@0.5 = 0,787 trên test V5+, vượt Faster R-CNN +14,6% và nhanh hơn ~10×
- **Ablation**: imgsz 640→960 cải thiện +11,7%; aug mạnh hơn cải thiện +0,5% mAP và giảm gap val–test
- **OOD Đà Nẵng**: định tính tốt, chưa có đánh giá định lượng do thiếu nhãn
- **Counting**: MAPE 5,1% trung bình, 4,2% ngày và 8,0% đêm
- **Lỗi chính**: đối tượng nhỏ, nhầm truck↔bus, điều kiện đêm, dense scene

Các kết quả này xác nhận hệ thống đáp ứng mục tiêu đặt ra và cung cấp nền tảng cho Chương 5 về xây dựng ứng dụng demo.
