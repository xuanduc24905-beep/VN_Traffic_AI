# CHƯƠNG 4  KẾT QUẢ THỰC NGHIỆM VÀ THẢO LUẬN

Chương này kiểm chứng năm cải tiến đã nêu ở §3.5 trên cùng tập test V5+ độc lập, đánh giá khả năng tổng quát hóa trên VN Aerial, đối chiếu mô hình cuối với Faster R-CNN và báo cáo kết quả đếm. Mọi chỉ số YOLO trong các bảng phát hiện được lấy từ `docs/report/assets/cross_domain_results.json`; số Faster R-CNN lấy từ `docs/report/assets/frcnn_v5plus_test.json`.

## 4.1 Môi trường và cấu hình thực nghiệm

### 4.1.1 Phần cứng và phần mềm

Thực nghiệm chạy trên GPU NVIDIA RTX 3500 Ada Laptop 12 GB. Môi trường chính gồm Python 3.10, PyTorch 2.5.1 và Ultralytics 8.4.142. Tất cả recipe YOLO dùng seed=42, optimizer SGD, AMP và checkpoint tốt nhất của từng run.

### 4.1.2 Dữ liệu đánh giá

Tập test V5+ có 1.500 ảnh chỉ đến từ hai camera SOICT24 ban ngày: cam_09 gồm 1.000 ảnh và cam_10 gồm 500 ảnh. Hai camera này được giữ độc lập với train/validation ở cấp video. Bộ cross-domain VN Aerial gồm 486 ảnh validation và 149 ảnh test từ góc nhìn trên cao; kiểm tra MD5 xác nhận 0 ảnh trùng với V5+.

### 4.1.3 Danh mục mô hình

| Mô hình | Thay đổi chính so với V5+ | Cấu hình thực |
|---|---|---|
| V5+ baseline | YOLOv8s, 960 pixel | `runs/detect/train_v8s_v5plus_960/args.yaml` |
| p4_scale | scale=0,7; translate=0,15; warm-start | `runs/detect/p4_scale_from_baseline/args.yaml` |
| p4_imgsz1280 | đầu vào 1.280 pixel; warm-start | `runs/detect/p4_imgsz1280_from_baseline/args.yaml` |
| r_p2 | thêm detection head stride 4 | `runs/detect/r_p2/args.yaml` |
| r_clspw | cls_pw=0,5; class-weighted BCE | `runs/detect/r_clspw/args.yaml` |
| r_progressive | curriculum 960→1.280 pixel | `runs/detect/r_progressive_s1/args.yaml`, `runs/detect/r_progressive/args.yaml` |

## 4.2 Quá trình huấn luyện

### 4.2.1 Đường cong loss

Ba thành phần box loss, classification loss và DFL của các run đều được lưu theo epoch trong `results.csv`. Các đường cong được dùng để kiểm tra hội tụ và phát hiện dao động, còn quyết định cuối dựa trên checkpoint tốt nhất thay vì epoch cuối.

**Hình 4.1** Đường cong box loss, classification loss, DFL và mAP trên validation của baseline V5+.

`r_clspw` đạt mAP@0.5 validation tốt nhất ở epoch 10/15; `r_p2` đạt tốt nhất ở epoch 19/25. Với progressive, giai đoạn 1 học ở 960 pixel trong 20 epoch và giai đoạn 2 tiếp tục ở 1.280 pixel trong 10 epoch. Kết quả test ở §4.3 cho thấy validation cao không tự động bảo đảm tổng quát hóa tốt nhất trên camera chưa thấy.

### 4.2.2 Thời gian huấn luyện ghi trong results.csv

| Mô hình | Số epoch có log | Thời gian tích lũy |
|---|---|---|
| V5+ baseline | 26 | 29 phút 13 giây |
| p4_scale | 28 | 1 giờ 00 phút 31 giây |
| p4_imgsz1280 | 40 | 38 phút 23 giây |
| r_p2 | 25 | 2 giờ 35 phút 04 giây |
| r_clspw | 15 | 30 phút 44 giây |
| r_progressive, hai giai đoạn | 20 + 10 | 1 giờ 16 phút 32 giây |

Các thời gian trên là giá trị được ghi trong từng `results.csv`; một số run có resume hoặc early stopping nên không nên dùng bảng này như benchmark tốc độ phần cứng tuyệt đối. `r_clspw` có chi phí fine-tuning thấp vì warm-start từ V5+ và chỉ chạy 15 epoch.

## 4.3 So sánh năm cải tiến trên test V5+

### 4.3.1 Kết quả tổng hợp

**Bảng 4.1** Kết quả trên test V5+ gồm 1.500 ảnh.

| Mô hình | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---:|---:|---:|---:|
| V5+ baseline | 0,7819 | 0,4134 | **0,7677** | 0,7073 |
| p4_scale | 0,7869 | **0,4235** | 0,7590 | 0,7325 |
| p4_imgsz1280 | 0,7745 | 0,4179 | 0,7287 | **0,7431** |
| r_p2 | 0,7530 | 0,3950 | 0,7417 | 0,7071 |
| **r_clspw, mô hình cuối** | **0,7927** | 0,4226 | 0,7445 | 0,7351 |
| r_progressive | 0,7711 | 0,4147 | 0,7408 | 0,7213 |

`r_clspw` đạt mAP@0.5 cao nhất. So với `p4_scale`, mức tăng là **+0,58 điểm tuyệt đối (0,74% tương đối)**. `p4_scale` vẫn nhỉnh hơn rất nhỏ ở mAP@0.5:0.95: 0,4235 so với 0,4226. Vì mục tiêu chính đã xác định là mAP@0.5 và `r_clspw` cải thiện hai lớp ít mẫu, `r_clspw` được chọn làm mô hình cuối.

**Hình 4.2** So sánh mAP@0.5 của baseline và các recipe kiến trúc, loss, curriculum trên test V5+.

### 4.3.2 AP@0.5 theo lớp

**Bảng 4.2** AP@0.5 theo bốn lớp trên test V5+.

| Mô hình | motorcycle | car | bus | truck |
|---|---:|---:|---:|---:|
| V5+ baseline | 0,8406 | 0,8839 | 0,7164 | 0,6868 |
| p4_scale | 0,8616 | 0,8851 | 0,7217 | 0,6793 |
| p4_imgsz1280 | 0,8624 | **0,8888** | 0,7418 | 0,6048 |
| r_p2 | 0,8385 | 0,8840 | 0,7449 | 0,5448 |
| **r_clspw** | 0,8354 | 0,8719 | **0,7647** | **0,6990** |
| r_progressive | **0,8720** | 0,8777 | 0,7209 | 0,6137 |

Kết quả thể hiện trade-off rõ ràng. `r_clspw` không đứng đầu ở motorcycle hoặc car nhưng đạt AP cao nhất cho bus và truck, qua đó nâng mAP tổng. `r_progressive` tốt nhất ở motorcycle nhưng giảm mạnh ở truck. `r_p2` cải thiện bus so với baseline nhưng giảm truck đủ lớn để làm mAP tổng thấp nhất.

## 4.4 Ablation study từng cải tiến

### 4.4.1 Ảnh hưởng augmentation không gian: p4_scale

So với V5+ baseline, `p4_scale` tăng mAP@0.5 từ 0,7819 lên 0,7869, tương ứng **+0,50 điểm tuyệt đối (0,64% tương đối)**; mAP@0.5:0.95 tăng 1,01 điểm và recall tăng 2,52 điểm. Precision giảm 0,87 điểm. Giả thuyết tăng biến thiên không gian được ủng hộ ở recall và mAP. Tuy nhiên recipe đồng thời warm-start, giảm learning rate và đổi batch, vì vậy không thể quy toàn bộ chênh lệch cho riêng `scale` hoặc `translate`.

### 4.4.2 Ảnh hưởng độ phân giải: p4_imgsz1280

`p4_imgsz1280` đạt mAP@0.5 = 0,7745, thấp hơn baseline **0,75 điểm tuyệt đối** dù recall tăng từ 0,7073 lên 0,7431. AP truck giảm từ 0,6868 xuống 0,6048. Như vậy tăng đầu vào lên 1.280 pixel không cải thiện mục tiêu tổng thể trong cấu hình đã thử; lợi ích recall đi kèm nhiều false positive hơn, thể hiện qua precision giảm còn 0,7287.

### 4.4.3 Ảnh hưởng kiến trúc P2: r_p2

`r_p2` đạt 0,7530 mAP@0.5, thấp hơn baseline **2,89 điểm tuyệt đối**. AP bus tăng 2,84 điểm nhưng AP truck giảm 14,20 điểm. Kết quả không ủng hộ giả thuyết rằng thêm head stride 4 tự động cải thiện toàn bộ bài toán vật nhỏ. Đây là negative result quan trọng: tăng độ phân giải đặc trưng cũng làm bài toán tối ưu phức tạp hơn, và 25 epoch từ phần trọng số COCO tương thích chưa bù được lợi thế warm-start của các recipe khác.

### 4.4.4 Ảnh hưởng class-weighted loss: r_clspw

Với `cls_pw=0,5`, `r_clspw` tăng mAP@0.5 **1,08 điểm tuyệt đối** so với V5+ baseline và **0,58 điểm tuyệt đối** so với `p4_scale`. AP bus tăng từ 0,7164 lên 0,7647 (**+4,83 điểm**), AP truck tăng từ 0,6868 lên 0,6990 (**+1,22 điểm**). Đổi lại AP motorcycle và car giảm nhẹ. Mẫu biến thiên này phù hợp với cơ chế class weighting đã mô tả ở §3.5.5: loss ưu tiên tương đối hơn cho lớp ít mẫu thay vì tối đa hóa lớp phổ biến.

**Bảng 4.3** Trọng số classification loss và biến thiên AP@0.5 so với V5+.

| Lớp | Trọng số loss | AP V5+ | AP r_clspw | Δ tuyệt đối |
|---|---:|---:|---:|---:|
| motorcycle | 0,571 | 0,8406 | 0,8354 | -0,52 điểm |
| car | 0,860 | 0,8839 | 0,8719 | -1,20 điểm |
| bus | 1,629 | 0,7164 | 0,7647 | +4,83 điểm |
| truck | 0,940 | 0,6868 | 0,6990 | +1,22 điểm |

### 4.4.5 Ảnh hưởng progressive curriculum: r_progressive

`r_progressive` đạt AP motorcycle cao nhất 0,8720 nhưng mAP@0.5 tổng chỉ 0,7711, thấp hơn baseline 1,08 điểm tuyệt đối. AP truck giảm 7,31 điểm so với baseline. Giả thuyết curriculum 960→1.280 giúp đồng đều các lớp không được ủng hộ; cấu hình này thiên về motorcycle và không giải quyết mất cân bằng lớp.

### 4.4.6 V6 pseudo-labeling

V6 bổ sung 4.396 ảnh pseudo-label nhưng đạt mAP@0.5 = 0,7608, thấp hơn V5+ baseline **2,11 điểm tuyệt đối**. AP motorcycle/car giảm, còn bus tăng từ 0,7164 lên 0,7427. Kết quả cho thấy bổ sung dữ liệu tự gán nhãn không bảo đảm cải thiện nếu phân bố nguồn và chất lượng nhãn không được kiểm soát. V6 không được chọn làm mô hình cuối.

## 4.5 Đánh giá cross-domain trên VN Aerial

### 4.5.1 Kết quả định lượng

**Bảng 4.4** mAP@0.5 trong miền và cross-domain. Gap là V5+ test trừ VN Aerial test.

| Mô hình | V5+ test | VN Aerial val | VN Aerial test | Gap đến Aerial test |
|---|---:|---:|---:|---:|
| V5+ baseline | 0,7819 | 0,6141 | 0,5994 | 18,26 điểm |
| **p4_scale** | 0,7869 | **0,6483** | **0,6373** | **14,97 điểm** |
| p4_imgsz1280 | 0,7745 | 0,6374 | 0,6015 | 17,29 điểm |
| r_p2 | 0,7530 | 0,5516 | 0,5369 | 21,61 điểm |
| **r_clspw** | **0,7927** | 0,6135 | 0,5676 | 22,51 điểm |
| r_progressive | 0,7711 | 0,6297 | 0,6115 | 15,96 điểm |

**Hình 4.3** So sánh mAP@0.5 trên V5+ test và hai phân hoạch VN Aerial.

### 4.5.2 Phân tích trade-off

`r_clspw` tốt nhất trong miền nhưng không tốt nhất cross-domain. `p4_scale` đạt mAP cao nhất ở cả VN Aerial validation và test, đồng thời có gap nhỏ nhất trong bảng. Kết quả cho thấy augmentation không gian mạnh mang lại độ bền trước thay đổi góc nhìn tốt hơn, còn class weighting tối ưu phân bố lớp của V5+ có thể làm tăng phụ thuộc vào miền huấn luyện. Vì mục tiêu chính của đề tài là test V5+, `r_clspw` vẫn là mô hình mặc định; khi triển khai trên camera aerial, `p4_scale` là lựa chọn thực nghiệm tốt hơn.

Faster R-CNN chưa được đánh giá trên VN Aerial nên không có số cross-domain để so sánh và không được đưa vào Bảng 4.3.

## 4.6 So sánh r_clspw với Faster R-CNN

**Bảng 4.5** So sánh trên cùng test V5+.

| Mô hình | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|
| Faster R-CNN ResNet50-FPN | 0,6408 | 0,3605 |
| **YOLOv8s r_clspw** | **0,7927** | **0,4226** |
| Chênh lệch tuyệt đối | **+15,19 điểm** | **+6,20 điểm** |

`r_clspw` vượt Faster R-CNN 15,19 điểm mAP@0.5 và 6,20 điểm mAP@0.5:0.95 trên cùng 1.500 ảnh. Asset Faster R-CNN hiện chỉ có mAP tổng và AP@0.5:0.95 theo lớp; Precision, Recall và kết quả VN Aerial **chưa đo**, vì vậy chương này không suy diễn các chỉ số đó.

## 4.7 Phân tích lỗi của r_clspw

### 4.7.1 Theo lớp

AP@0.5 của `r_clspw` lần lượt là 0,8354 cho motorcycle, 0,8719 cho car, 0,7647 cho bus và 0,6990 cho truck. Truck vẫn là lớp khó nhất dù được cải thiện so với baseline. Bus hưởng lợi lớn nhất từ class-weighted loss. Đây là bằng chứng định lượng duy nhất theo lớp có trong asset đánh giá chuẩn.

### 4.7.2 Theo kích thước và điều kiện

Phân rã TP/FP/FN của `r_clspw` theo small/medium/large và theo điều kiện ánh sáng **chưa đo định lượng**. Tập test chỉ gồm cam_09 và cam_10 SOICT24 ban ngày nên không thể dùng nó để kết luận chênh lệch ngày–đêm. Các phân tích tiếp theo cần xuất prediction ở ngưỡng cố định, ghép với ground truth theo IoU≥0,5, rồi tổng hợp theo diện tích bounding box và camera.

**Hình 4.4** Confusion matrix của r_clspw trên test V5+; placeholder chờ xuất trực tiếp từ run r_clspw.

**Hình 4.5** Ví dụ false positive và false negative của r_clspw; placeholder chờ chọn mẫu từ prediction đã ghép ground truth.

## 4.8 Kết quả đếm phương tiện

Hệ thống kết hợp mô hình mặc định `r_clspw` với ByteTrack và logic giao cắt vạch. Bộ kiểm thử đếm gồm năm đoạn video một phút lấy từ nguồn của cam_09 và cam_10 ban ngày, không trùng các frame train/validation. Ground truth được đếm tay và đối chiếu theo lớp.

**Bảng 4.6** Kết quả đếm tổng hợp.

| Lớp | Ground truth | Hệ thống | MAE/đoạn | MAPE |
|---|---:|---:|---:|---:|
| motorcycle | 298 | 282 | 3,2 | 5,4% |
| car | 86 | 82 | 0,8 | 4,7% |
| bus | 11 | 12 | 0,2 | 9,1% |
| truck | 17 | 15 | 0,4 | 11,8% |
| **Tổng** | **412** | **391** | **4,6** | **5,1%** |

MAPE tổng là 5,1%. Bus và truck có MAPE cao hơn do mẫu số nhỏ; vì vậy cần đọc đồng thời sai số tuyệt đối. Các chỉ số tracking chuẩn như IDF1, MOTA và HOTA chưa đo do chưa có ground-truth track ID theo khung hình.

## 4.9 Thảo luận và trả lời câu hỏi nghiên cứu

### 4.9.1 Câu hỏi 1 — YOLOv8s và Faster R-CNN

Trên cùng test V5+, `r_clspw` đạt 0,7927 mAP@0.5 so với 0,6408 của Faster R-CNN, chênh 15,19 điểm tuyệt đối. Ở mAP@0.5:0.95, chênh lệch là 6,20 điểm. Kết quả tốc độ model-only chưa được đo lại trong asset đánh giá chuẩn; hiệu năng end-to-end của ứng dụng được báo cáo riêng ở Chương 5.

### 4.9.2 Câu hỏi 2 — Ảnh hưởng của các chiến lược cải tiến

Trong năm recipe, class-weighted loss cho kết quả mAP@0.5 tốt nhất; augmentation không gian đứng thứ hai nhưng tổng quát hóa cross-domain tốt nhất. Tăng ảnh lên 1.280 pixel và progressive curriculum không tăng mAP tổng. P2 là negative result rõ nhất. Do một số recipe đồng thời thay đổi khởi tạo, learning rate hoặc batch, kết quả được diễn giải ở mức recipe thay vì khẳng định quan hệ nhân quả của một tham số đơn lẻ.

**Bảng 4.7** Tổng hợp kết quả kiểm chứng các giả thuyết.

| Recipe | Giả thuyết chính | Kết quả V5+ | Kết luận |
|---|---|---|---|
| p4_scale | Augmentation mạnh tăng robustness | +0,50 điểm mAP@0.5 | Ủng hộ một phần; có confound warm-start/LR |
| p4_imgsz1280 | Độ phân giải cao tăng mAP | -0,75 điểm | Không ủng hộ |
| r_p2 | Head stride 4 giúp vật nhỏ | -2,89 điểm | Không ủng hộ trong cấu hình đã thử |
| r_clspw | Class weighting giúp lớp hiếm | +1,08 điểm; bus/truck tăng | Ủng hộ |
| r_progressive | Curriculum 960→1.280 tăng mAP | -1,08 điểm | Không ủng hộ |

### 4.9.3 Câu hỏi 3 — Lỗi phổ biến

Truck là lớp có AP thấp nhất. Phân tích định lượng theo kích thước, điều kiện sáng và loại FP/FN chưa có trong asset hiện tại, nên chưa đủ bằng chứng để kết luận tần suất từng kiểu lỗi. Báo cáo giữ trạng thái “chưa đo” thay vì suy diễn từ quan sát rời rạc.

### 4.9.4 Câu hỏi 4 — Mục tiêu mAP@0.5 ≥ 0,75

`r_clspw` đạt 0,7927, vượt ngưỡng mục tiêu 0,75. So với YOLOv8s-COCO zero-shot 0,0922, mức tăng là **+70,06 điểm tuyệt đối**. So với V5 honest 0,6648, mức tăng là **+12,79 điểm tuyệt đối (19,25% tương đối)**.

### 4.9.5 Câu hỏi 5 — Hệ thống đếm

Hệ thống đạt MAPE tổng 5,1% trên năm đoạn kiểm thử với 412 lượt ground truth. Sai số phần trăm cao hơn ở bus và truck cần được hiểu cùng số mẫu nhỏ. Chất lượng tracker mới được đánh giá gián tiếp qua counting, chưa có chỉ số tracking chuẩn.

### 4.9.6 Tiểu kết chương

Kết quả xác nhận `r_clspw` là mô hình cuối cho test V5+ với mAP@0.5 = 0,7927. `p4_scale` là lựa chọn tốt hơn khi ưu tiên cross-domain aerial. P2, tăng độ phân giải đơn thuần, progressive curriculum và pseudo-labeling không cải thiện mục tiêu mAP tổng trong cấu hình đã thử. Việc báo cáo đồng thời positive và negative results giúp làm rõ giới hạn áp dụng của từng chiến lược.
