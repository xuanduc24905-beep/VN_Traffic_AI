# CHƯƠNG 3  PHƯƠNG PHÁP ĐỀ XUẤT

Chương này trình bày phương pháp xây dựng hệ thống phát hiện, theo vết và đếm phương tiện giao thông Việt Nam được đề xuất trong đồ án. Nội dung tập trung mô tả thiết kế cuối cùng gồm: (i) tổng quan hệ thống hai luồng, (ii) thiết kế dữ liệu V5+ và quy ước nhãn, (iii) chiến lược chia tập theo video kèm kiểm soát rò rỉ, (iv) tiền xử lý và tăng cường dữ liệu, (v) cấu hình huấn luyện YOLOv8s (mô hình chính), (vi) cấu hình huấn luyện Faster R-CNN ResNet50-FPN (mô hình đối chứng), (vii) thuật toán theo vết và đếm bằng ByteTrack, (viii) các tiêu chí đánh giá áp dụng. Các thí nghiệm khảo sát (ablation) và kết quả định lượng được trình bày trong Chương 4.

## 3.1 Tổng quan hệ thống

### 3.1.1 Kiến trúc hai luồng

Hệ thống gồm hai quy trình xử lý tách biệt chạy ở hai thời điểm khác nhau nhưng cùng dựa trên một mô hình phát hiện YOLOv8s đã được fine-tune.

**Luồng huấn luyện (offline)** là giai đoạn chuẩn bị, không yêu cầu xử lý thời gian thực. Mục đích là tạo ra mô hình có khả năng nhận dạng bốn loại phương tiện trên dữ liệu giao thông Việt Nam. Luồng gồm bốn bước chính:
- Chuẩn hóa ảnh và nhãn từ năm nguồn dữ liệu Roboflow về cùng schema 4 lớp.
- Chia tập theo video_id và kiểm tra rò rỉ tự động để đảm bảo mỗi video chỉ xuất hiện trong đúng một tập.
- Fine-tune YOLOv8s từ trọng số pretrained COCO qua ba giai đoạn tăng dần (V5 honest 640px → V5+ 960px → p4_scale finetune), đồng thời fine-tune Faster R-CNN ResNet50-FPN trên cùng dữ liệu để đối chứng.
- Chọn checkpoint tốt nhất theo mAP@0.5 trên tập validation.

**Luồng suy luận (online)** là giai đoạn ứng dụng, dùng mô hình đã huấn luyện để xử lý video thực tế. Luồng gồm năm bước:
- Đọc video theo khung hình liên tiếp.
- Phát hiện phương tiện trên từng khung hình bằng YOLOv8s.
- Theo vết các phương tiện qua các khung hình liên tiếp bằng ByteTrack [19], gán track ID bền vững cho mỗi phương tiện.
- Phát hiện giao cắt vạch đếm ảo khi quỹ đạo phương tiện cắt qua.
- Xuất kết quả dưới dạng tệp CSV sự kiện và video chú thích.

### 3.1.2 Pipeline xử lý video

Pipeline suy luận có thể biểu diễn bằng sơ đồ sau:

**Hình 3.1** Pipeline xử lý video gồm bốn thành phần chính: Video đầu vào → YOLOv8s detect → ByteTrack gán track ID → Counter phát hiện giao cắt vạch → Xuất CSV sự kiện và video chú thích.

YOLOv8s là mạng CNN chính thực hiện phát hiện phương tiện trên từng khung hình độc lập. ByteTrack là thuật toán theo vết (không phải mạng CNN thứ hai) liên kết các detection liên tiếp dựa trên IoU, điểm tin cậy và Kalman filter [27]. Module Counter triển khai thuật toán phát hiện giao cắt đoạn thẳng với cơ chế chống đếm trùng.

## 3.2 Dữ liệu và quy ước nhãn

### 3.2.1 Nguồn dữ liệu

Bộ dữ liệu V5+ của đề tài được xây dựng bằng cách hợp nhất năm nguồn dữ liệu công khai từ Roboflow, tất cả đều tập trung vào phương tiện giao thông Việt Nam.

**Bảng 3.1** Thống kê các nguồn dữ liệu V5+ với số ảnh, số khung hình sau lấy mẫu, phân bố lớp và kích thước bounding box theo chuẩn COCO (small: diện tích < 32², medium: 32² ≤ diện tích < 96², large: ≥ 96²).

| Nguồn | Số ảnh | Camera/scene | Mô tả nội dung | Phân bố lớp chính | Kích thước bbox |
|---|---|---|---|---|---|
| SOICT 2024 (v2) | ~2.100 | Camera ngày TP.HCM | Ảnh ngã tư nhiều phương tiện | Motorcycle (62%), car (28%), bus (5%), truck (5%) | Medium (55%), small (35%), large (10%) |
| SOICT Night (v1) | ~1.500 | Camera đêm (nguồn SOICT 2024) | Cảnh đêm có ánh đèn pha | Motorcycle (68%), car (25%), bus (4%), truck (3%) | Small (45%), medium (45%), large (10%) |
| Vehicle Vietnam-CanTho (v19) | ~1.200 | Camera Cần Thơ | Đường phố Cần Thơ | Motorcycle (70%), car (22%), bus (4%), truck (4%) | Medium (50%), small (40%), large (10%) |
| Vietnamese Vehicle v3 | ~950 | Đa nguồn | Ảnh hỗn hợp nhiều camera | Motorcycle (60%), car (27%), bus (7%), truck (6%) | Medium (55%), small (30%), large (15%) |
| Vietnam Vehicles v10 (visionvn) | ~4.400 | 776 camera (snapshot CCTV) | Snapshot từ camera giám sát | Motorcycle (75%), car (20%), bus (3%), truck (2%) | Small (50%), medium (40%), large (10%) |
| **Tổng V5+** | **10.203** | **>800 scene/camera** | Hợp nhất 5 nguồn | Motorcycle (69%), car (24%), bus (4%), truck (3%) | Small (42%), medium (48%), large (10%) |

Năm nguồn dữ liệu được phân phối theo giấy phép CC BY 4.0 (Roboflow public), cho phép sử dụng lại với điều kiện trích dẫn nguồn. Các trích dẫn cụ thể được ghi trong phần Tài liệu tham khảo.

**Hình 3.2** Ví dụ ảnh từ bộ dữ liệu SOICT Night (camera giám sát giao thông ban đêm thuộc nguồn SOICT 2024). 

**Hình 3.3** Ví dụ ảnh từ bộ dữ liệu SOICT 2024 (camera ngày TP.HCM).

**Hình 3.4** Ví dụ ảnh từ bộ dữ liệu Vehicle Vietnam-CanTho.

**Hình 3.5** Ví dụ ảnh từ bộ dữ liệu Vietnamese Vehicle v3.

**Hình 3.6** Ví dụ ảnh từ bộ dữ liệu Vietnam Vehicles v10 (visionvn, snapshot CCTV).

### 3.2.2 Schema nhãn 4 lớp

Tất cả nguồn dữ liệu được quy đổi về cùng một schema 4 lớp phương tiện phù hợp với đặc thù giao thông Việt Nam:

**Bảng 3.2** Schema nhãn 4 lớp.

| ID | Tên lớp | Mô tả |
|---|---|---|
| 0 | motorcycle | Xe máy, xe tay ga, mô tô |
| 1 | car | Ô tô con (sedan, hatchback, SUV, pickup nhỏ) |
| 2 | bus | Xe buýt (bus đô thị, bus liên tỉnh) |
| 3 | truck | Xe tải các loại (bao gồm container) |

Các lớp khác xuất hiện trong dữ liệu nguồn (bicycle, auto-rickshaw, van, person, xe ba gác) được **loại bỏ hoàn toàn** chứ không gộp sang lớp gần nhất về ngữ nghĩa. Nguyên tắc này nhằm tránh gây nhầm nhãn — ví dụ ánh xạ van sang truck có thể làm mô hình học nhầm hình dáng (van gần với SUV/minivan hơn là xe tải thùng). Các ảnh có nhãn gốc không ánh xạ được sau khi loại bỏ bị giữ lại dưới dạng ảnh nền (background image) để đóng góp vào việc học negative samples.

Quy đổi cụ thể cho nguồn Vietnam Vehicles v10 (có 6 lớp gốc):
- `container`, `xe_tai` → `truck` (ID 3)
- `oto` → `car` (ID 1)
- `xe_khach` → `bus` (ID 2)
- `xe_may` → `motorcycle` (ID 0)
- `xe_ba_gac` → **loại bỏ** (không có lớp tương đương trong schema 4 lớp)

### 3.2.3 Định dạng nhãn YOLO

Mỗi ảnh đi kèm một tệp nhãn .txt cùng tên trong thư mục `labels/` song song với thư mục `images/`. Mỗi dòng trong tệp nhãn biểu diễn một bounding box theo định dạng chuẩn YOLO [11]:

```
class_id  cx  cy  w  h
```

Trong đó `class_id ∈ {0, 1, 2, 3}` là ID lớp theo schema đã định, `(cx, cy)` là tọa độ tâm bounding box và `(w, h)` là chiều rộng và chiều cao bounding box. Cả bốn giá trị được chuẩn hóa về khoảng `[0, 1]` theo kích thước ảnh gốc. Chuẩn hóa cho phép tái sử dụng cùng tệp nhãn với bất kỳ độ phân giải resize nào mà không cần tính toán lại.

**Hình 3.7** Minh họa hệ tọa độ YOLO trên một ảnh thực từ bộ SOICT 2024: gốc tọa độ ở góc trên trái, trục +X hướng sang phải, trục +Y hướng xuống dưới; tâm bounding box `(cx, cy)` và kích thước `(w, h)` đều chuẩn hóa về `[0, 1]`.

### 3.2.4 Chiến lược kiểm tra chất lượng dữ liệu

Phương pháp đề xuất áp dụng chiến lược kiểm tra hai tầng độc lập trước mỗi lần huấn luyện. Cả hai tầng đều phải đạt chuẩn mới cho phép bắt đầu training (fail-hard).

**Tầng 1 — kiểm tra toàn vẹn nhãn** (`scripts/data_prep/validate_labels.py`): kiểm tra bảy ràng buộc ở cấp độ từng bounding box được tổng hợp trong **Bảng 3.3**.

| # | Ràng buộc | Lỗi phổ biến phát hiện |
|---|---|---|
| 1 | Mỗi dòng nhãn có đúng 5 trường (class_id, cx, cy, w, h) | Nhãn thiếu trường hoặc dư ký tự |
| 2 | class_id ∈ {0, 1, 2, 3} | ID vượt phạm vi do quy đổi lớp sai |
| 3 | 0 ≤ cx, cy, w, h ≤ 1 | Tọa độ không chuẩn hóa |
| 4 | w > 0 và h > 0 | Bounding box degenerate (chiều âm hoặc bằng 0) |
| 5 | cx - w/2 ≥ 0 và cx + w/2 ≤ 1 (và tương tự cho y) | Bounding box vượt ra ngoài biên ảnh |
| 6 | Mỗi tệp ảnh có đúng một tệp nhãn tương ứng | Ảnh thiếu nhãn hoặc nhãn thiếu ảnh |
| 7 | Phân bố lớp per split không có lớp nào = 0 | Mất cân bằng cực đoan |

Kết quả validation được lưu vào tệp JSON (ví dụ `label_validation_v5plus.json`) kèm phân bố lớp để làm bằng chứng tái lập.

**Tầng 2 — kiểm tra rò rỉ cấp video** (`scripts/data_prep/audit_leak.py`, trình bày chi tiết trong §3.3): đảm bảo không có video nào xuất hiện trong nhiều tập cùng lúc.

Căn cứ thiết kế hai tầng độc lập: hai tầng đo hai loại lỗi hoàn toàn khác nhau; vi phạm tầng này không bao hàm tầng kia. Ultralytics tự kiểm tra một phần ràng buộc toàn vẹn khi quét dataset lần đầu [20], nhưng chỉ warn-and-skip chứ không fail-hard. Script `validate_labels.py` bổ sung cơ chế fail-hard chủ động để kiểm soát chất lượng trước khi tốn tài nguyên GPU.

## 3.3 Chia tập theo video và kiểm soát rò rỉ

### 3.3.1 Vấn đề của chia ngẫu nhiên theo khung hình

Camera CCTV cố định quay 25–30 khung hình mỗi giây. Giữa hai khung hình liên tiếp, nền (background) gần như không đổi, lane marking và biển báo giữ nguyên, ánh sáng chỉ thay đổi rất chậm; phương tiện di chuyển vài pixel. Hai khung hình cách nhau 0,04 giây về bản chất là hai bản sao gần như y hệt. Nếu chia ngẫu nhiên theo khung hình, một khung hình vào train và khung hình kế tiếp vào test sẽ dẫn đến hiện tượng mô hình "ghi nhớ cảnh" (scene memorization) thay vì học khái niệm phương tiện.

Trong các phiên bản đầu của đề tài (V1 đến V4), do sử dụng chia ngẫu nhiên theo khung hình nên mAP@0.5 trên test đạt mức ảo ~0,95, nhưng mô hình không tổng quát hóa được sang video mới. Sau khi phát hiện hiện tượng này, đề tài rebuild lại các bộ dữ liệu từ V5 theo nguyên tắc chia cấp video.

### 3.3.2 Nguyên tắc chia theo video_id

Phương pháp đề xuất lấy `video_id` làm đơn vị chia — mỗi video chỉ xuất hiện trong đúng một tập (train HOẶC val HOẶC test). Nguyên tắc này đảm bảo mô hình được đánh giá trên các cảnh (scene) hoàn toàn chưa từng gặp trong huấn luyện.

Việc xác định `video_id` được thực hiện bằng cách parse tên file theo các regex pattern đã định nghĩa trong `audit_leak.py`, phân biệt các nguồn dữ liệu (SOICT, cantho, vnv3, v10 visionvn). Với nguồn Vietnam Vehicles v10 (snapshot CCTV), mỗi camera được coi là một `camera_id` tương đương với `video_id` của các nguồn khác.

### 3.3.3 Cấu hình chia dataset V5+

Áp dụng nguyên tắc chia theo video, bộ dữ liệu V5+ được phân chia như sau:

**Bảng 3.4** Cấu hình chia tập V5+.

| Tập | Số ảnh | Tỉ lệ | Nguồn |
|---|---|---|---|
| Train | 7.237 | 71% | V5 base (2.833) + Vietnam Vehicles v10 (4.404) |
| Validation | 1.466 | 14% | V5 val (giữ nguyên từ V5 base) |
| Test | 1.500 | 15% | V5 test (giữ nguyên từ V5 base) |
| **Tổng** | **10.203** | **100%** | |

Val và test giữ nguyên từ V5 base (không bổ sung từ v10) để các kết quả có thể so sánh công bằng với V5 honest baseline. Toàn bộ v10 (4.404 ảnh) được thêm vào train.

Tập test bao gồm cả cảnh ngày (SOICT 2024, Cần Thơ) và cảnh đêm (SOICT Night) để kết quả đánh giá không phụ thuộc duy nhất vào một điều kiện ánh sáng. Với tập test có 1.500 ảnh phủ nhiều scene khác nhau, kết quả mAP có độ tin cậy cao hơn so với chia 2 video test của V5 ban đầu.

### 3.3.4 Chiến lược lấy mẫu để giảm dư thừa

Vì các video dense (hàng nghìn khung hình liên tiếp) tạo ra sự dư thừa cao, phương pháp áp dụng lấy mẫu theo stride:

**Bảng 3.5** Bước lấy mẫu (stride) theo loại video.

| Loại video | Stride | Lý do |
|---|---|---|
| Camera ngày (SOICT 2024, Cần Thơ) | 5 khung hình | Phương tiện di chuyển nhanh, 5 frame ~ 0.2s đủ để scene đổi |
| Camera đêm (SOICT Night) | 10 khung hình | Phương tiện ít hơn, chuyển động chậm, giảm dư thừa |
| Snapshot v10 | 1 (giữ nguyên) | Đã là snapshot rời rạc, không có khái niệm video sequence |

### 3.3.5 Xác minh không rò rỉ

Sau khi chia dữ liệu, nhóm chạy script `audit_leak.py` để xác nhận không có rò rỉ. Script kiểm tra hai điều kiện:
- **Điều kiện 1**: Không có video nào xuất hiện ở hơn một tập. Số video giao nhau giữa các tập bằng 0.
- **Điều kiện 2**: Tỉ lệ khung hình của tập val và test thuộc về video đã có trong tập train bằng 0%.

Kết quả kiểm tra trên V5+ (lưu trong `leak_audit_v5plus.json`):
- Số video unique: 4.419
- Số video xuất hiện cross-split: **0** ✓
- Tỉ lệ frame leak: **0,00%** ✓

Tệp `split_manifest.json` ghi lại chính xác từng video được xếp vào tập nào. Hai tệp này đóng vai trò bằng chứng tái lập: bất kỳ ai muốn kiểm chứng đều có thể mở manifest và chạy lại audit để xác nhận.

## 3.4 Tiền xử lý và tăng cường dữ liệu

### 3.4.1 Pipeline tiền xử lý

Mỗi ảnh được xử lý qua bốn bước trước khi đưa vào mạng YOLOv8s:

1. **Letterbox resize**: ảnh được resize về kích thước mục tiêu (960 × 960) giữ nguyên tỉ lệ khung hình gốc. Phần trống được đệm màu xám (114, 114, 114) để hình dạng phương tiện không bị méo.
2. **Chuyển thứ tự kênh màu**: từ BGR (OpenCV mặc định) sang RGB (YOLOv8 yêu cầu).
3. **Chuẩn hóa pixel**: giá trị `[0, 255]` được chia 255 để đưa về `[0.0, 1.0]` float32.
4. **Cache metadata**: ở lần huấn luyện đầu, Ultralytics quét dataset và lưu metadata vào tệp `.cache` cạnh thư mục nhãn; các lần sau đọc trực tiếp từ cache.

Mọi phép biến đổi hình học (lật ngang, co giãn, tịnh tiến, mosaic [15]) đều cập nhật bounding box đồng bộ với ảnh. Augmentation chỉ áp dụng cho tập train; validation và test giữ nguyên để đo lường phản ánh đúng phân phối dữ liệu.

### 3.4.2 Lựa chọn độ phân giải đầu vào

Phương pháp đề xuất sử dụng `imgsz = 960` cho cả huấn luyện và suy luận. Căn cứ chọn:

- **Phân phối độ phân giải nguồn**: ~85% dữ liệu đến từ các bộ SOICT có kích thước gốc 1280 × 720. Kích thước 960 giữ được chi tiết của các đối tượng nhỏ (xe máy ở xa camera) mà không phóng to quá mức các ảnh nhỏ hơn.
- **Ràng buộc phần cứng**: với GPU 12 GB VRAM, kích thước 960 vẫn đủ bộ nhớ để huấn luyện với batch 16, giúp ước lượng gradient ổn định hơn so với batch 8 (nếu dùng imgsz 1280 phải giảm batch xuống 8).
- **Chi phí tính toán cân bằng**: số điểm ảnh ở kích thước 960 gấp 2,25 lần so với 640, nhưng mức cải thiện khả năng phát hiện đối tượng nhỏ được xem là xứng đáng với chi phí tăng thêm (xác nhận trong Chương 4).

### 3.4.3 Chiến lược tăng cường dữ liệu

Phương pháp kết hợp các phép tăng cường mặc định của Ultralytics YOLOv8 [20] với một số thiết lập bổ sung phù hợp với đặc thù dữ liệu giao thông Việt Nam (cảnh đông, nhiều phương tiện che khuất, mất cân bằng lớp).

**Bảng 3.6** Tham số tăng cường dữ liệu mặc định của YOLOv8.

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| hsv_h | 0.015 | Dao động Hue ±1.5% |
| hsv_s | 0.7 | Dao động Saturation ±70% |
| hsv_v | 0.4 | Dao động Value ±40% |
| degrees | 0.0 | Không xoay (giữ hướng xe thật) |
| translate | 0.1 | Dịch chuyển tối đa 10% kích thước |
| scale | 0.5 | Co giãn ±50% |
| fliplr | 0.5 | Xác suất lật ngang 50% |
| flipud | 0.0 | Không lật dọc (phi thực tế) |
| mosaic | 1.0 | Ghép 4 ảnh (mặc định on) |

**Bảng 3.7** Thiết lập bổ sung và điều chỉnh cho dữ liệu Việt Nam.

| Tham số | V5+ baseline | Mô hình cuối p4_scale | Lý do |
|---|---|---|---|
| mixup [21] | 0.1 | 0.1 | Giúp mô hình học overlap xe trong cảnh đông |
| cos_lr [22] | True | True | Cosine schedule giảm LR mượt hơn |
| close_mosaic | 10 epoch cuối | 10 epoch cuối | Tắt mosaic ở epoch cuối để học distribution gốc |
| scale | 0.5 | **0.7** | Tăng robust với xe ở xa |
| translate | 0.1 | **0.15** | Tăng robust với vị trí xe trong frame |
| copy_paste | 0.3 | **0.0** | copy_paste không có tác dụng cho bbox (chỉ segmentation) |

Validation và test chỉ áp dụng letterbox resize, chuyển kênh và chuẩn hóa; không áp dụng mixup, mosaic, flip hay bất kỳ augmentation nào.

## 3.5 Mô hình phát hiện chính YOLOv8s

### 3.5.1 Lý do chọn YOLOv8s

YOLOv8s là phiên bản small trong họ YOLOv8 của Ultralytics [6]. Lý do chọn:
- **Số tham số**: 11,1 triệu — nhỏ gọn, phù hợp triển khai trên thiết bị có cấu hình hạn chế.
- **GFLOPs**: 28,6 — chi phí tính toán thấp, cho phép đạt FPS cao trên GPU laptop.
- **Kiến trúc**: anchor-free, backbone CSPDarknet, neck PAN-FPN [23], [24], head decoupled.
- **Cân bằng tốc độ–độ chính xác**: nhanh hơn YOLOv8m/l và đủ chính xác cho bài toán 4 lớp.
- **Hệ sinh thái Ultralytics**: tích hợp sẵn ByteTrack cho tracking, hỗ trợ ONNX export, API Python đơn giản.

### 3.5.2 Chiến lược huấn luyện ba giai đoạn

Thay vì huấn luyện một lần duy nhất, đề tài áp dụng chiến lược ba giai đoạn tăng dần:

**Giai đoạn 1 — V5 honest baseline** (imgsz=640):
- Fine-tune YOLOv8s từ trọng số pretrained COCO
- imgsz = 640, batch = 32, epochs = 40, lr0 = 0.01
- Mục tiêu: baseline tối giản không có augmentation nâng cao

**Giai đoạn 2 — V5+ 960px baseline** (imgsz=960):
- Fine-tune YOLOv8s từ COCO với cấu hình nâng cao
- imgsz = 960, batch = 16, epochs = 40, lr0 = 0.005, cos_lr=True
- Augmentation: mixup=0.1, scale=0.5, translate=0.1
- Mục tiêu: khai thác độ phân giải cao hơn cho xe máy nhỏ

**Giai đoạn 3 — p4_scale finetune** (mô hình cuối):
- Fine-tune tiếp từ best.pt của V5+ 960px (warm-start)
- imgsz = 960, batch = 6, epochs = 40, **lr0 = 0.0005** (giảm 10×)
- Augmentation mạnh hơn: **scale = 0.7, translate = 0.15**, copy_paste = 0.0
- Mục tiêu: tăng robustness với scale variation

**Bảng 3.8** Cấu hình huấn luyện ba giai đoạn YOLOv8s.

| Tham số | V5 honest | V5+ 960px | p4_scale (cuối) |
|---|---|---|---|
| init | yolov8s.pt (COCO) | yolov8s.pt (COCO) | V5+ best.pt |
| imgsz | 640 | 960 | 960 |
| batch | 32 | 16 | 6 |
| epochs | 40 | 40 | 40 |
| lr0 | 0.01 | 0.005 | 0.0005 |
| lr schedule | linear | cosine | cosine |
| optimizer | SGD | SGD | SGD |
| momentum | 0.937 | 0.937 | 0.937 |
| weight_decay | 0.0005 | 0.0005 | 0.0005 |
| warmup_epochs | 3 | 3 | 3 |
| close_mosaic | 10 | 10 | 10 |
| mixup | 0 | 0.1 | 0.1 |
| scale | 0.5 | 0.5 | 0.7 |
| translate | 0.1 | 0.1 | 0.15 |
| patience | 15 | 15 | 15 |
| seed | 42 | 42 | 42 |

### 3.5.3 Hàm mất mát

YOLOv8 sử dụng ba thành phần loss kết hợp [12], [25], [26]: box loss (CIoU), classification loss (BCE với class weights), DFL loss. Trọng số mặc định `box=7.5, cls=0.5, dfl=1.5` được giữ nguyên.

### 3.5.4 Tiêu chí lựa chọn checkpoint

Checkpoint tốt nhất được chọn dựa trên **mAP@0.5 trên tập validation**, đồng thời xem xét AP của từng lớp để tránh chọn mô hình có kết quả trung bình cao nhưng yếu ở các lớp quan trọng (bus, truck). Tập test **không tham gia** quá trình chọn checkpoint, nhằm tránh selection bias.

## 3.6 Mô hình đối chứng Faster R-CNN ResNet50-FPN

Để trả lời Câu hỏi nghiên cứu 1 (so sánh YOLOv8 và Faster R-CNN trên cùng tập test), đề tài huấn luyện một mô hình Faster R-CNN ResNet50-FPN trên cùng tập dữ liệu V5+.

### 3.6.1 Cấu hình kiến trúc

Mô hình được xây dựng bằng `torchvision.models.detection.fasterrcnn_resnet50_fpn` [13] với trọng số pretrained COCO:

```python
model = fasterrcnn_resnet50_fpn(
    weights="DEFAULT",
    min_size=800,              # FPN standard: resize cạnh ngắn về 800
    max_size=1333,             # giới hạn cạnh dài
    box_score_thresh=0.05,     # ngưỡng confidence cho inference
    box_nms_thresh=0.5,        # ngưỡng IoU cho NMS
    box_detections_per_img=100,
    rpn_pre_nms_top_n_train=2000,
    rpn_post_nms_top_n_train=1000,
)
# Thay box_predictor cho 4 lớp + 1 background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=5)
```

Lưu ý quy ước nhãn: YOLO dùng ID 0–3 cho 4 lớp phương tiện; Faster R-CNN của TorchVision dùng ID 1–5 (ID 0 là background). Bộ chuyển đổi dữ liệu cộng 1 vào class_id của nhãn YOLO khi chuyển sang Faster R-CNN, và trừ 1 khi lấy kết quả ra để chuẩn hóa về schema chung khi so sánh.

### 3.6.2 Cấu hình huấn luyện

**Bảng 3.9** Cấu hình huấn luyện Faster R-CNN ResNet50-FPN.

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| init weights | COCO pretrained (DEFAULT) | Từ TorchVision model zoo |
| min_size / max_size | 800 / 1333 | Chuẩn FPN, giữ nguyên để không phá feature |
| batch | 4 | VRAM-constrained (Faster R-CNN tốn ~3× YOLO) |
| gradient accumulation | 2 | Effective batch = 8 |
| epochs | 30 | FRCNN hội tụ nhanh hơn YOLO |
| optimizer | SGD | momentum=0.9, weight_decay=0.0005 |
| lr0 | 0.005 | Linear scaling: lr = 0.02 × batch/16 |
| lr schedule | linear warmup (2 epoch) + cosine | Giảm mượt sau warmup |
| gradient clip | 10.0 | FRCNN dễ explode gradient |
| eval every | 2 epoch | Tiết kiệm thời gian |
| freeze backbone | False | Fine-tune toàn mạng |
| augmentation | horizontal flip 0.5 | Giữ tối thiểu để fair compare |
| seed | 42 | Khớp seed với YOLO |
| device | NVIDIA RTX 3500 Ada 12GB | Cùng phần cứng YOLO |

### 3.6.3 Đánh giá so sánh công bằng

Cả YOLOv8s và Faster R-CNN được đánh giá trên **cùng tập test V5+** (1.500 ảnh) bằng `torchmetrics.detection.MeanAveragePrecision` để đảm bảo cùng implementation tính mAP, tránh chênh lệch do khác thư viện (Ultralytics vs TorchVision). Chỉ số báo cáo: mAP@0.5, mAP@0.5:0.95, mAP@0.75, AP@0.5:0.95 per class.

Tốc độ FPS được đo trên cùng GPU RTX 3500 Ada với 100 ảnh test sau 20 ảnh warm-up, bao gồm tiền xử lý + inference + hậu xử lý (NMS).

## 3.7 Theo vết và đếm phương tiện

### 3.7.1 Theo vết với ByteTrack

Sau khi YOLOv8s phát hiện phương tiện ở mỗi khung hình, thuật toán ByteTrack [19] liên kết các detection giữa các khung hình liên tiếp để gán ID bền vững cho mỗi phương tiện. ByteTrack hoạt động dựa trên:
- **IoU matching** giữa detection ở frame hiện tại và track hiện có
- **Confidence score** để phân biệt detection chất lượng cao (high-conf) và thấp (low-conf); high-conf được match trước, sau đó low-conf được match với các track chưa được gán
- **Kalman filter** [27] dự đoán vị trí của track ở frame kế tiếp giúp xử lý occlusion ngắn

ByteTrack được cấu hình qua tham số `tracker="bytetrack.yaml"` trong Ultralytics [28].

### 3.7.2 Thuật toán đếm qua vạch ảo

Module đếm được thiết kế dựa trên phát hiện giao cắt đoạn thẳng:

**Bước 1**: Người dùng xác định một hoặc nhiều vạch đếm (counting line) trên khung hình bằng hai điểm đầu–cuối `p₁, p₂`, đồng thời chọn hướng đếm: cả hai chiều, chỉ từ trái sang phải, hoặc chỉ từ phải sang trái.

**Bước 2**: Với mỗi phương tiện (xác định bằng `track_id`), hệ thống lưu lại vị trí tâm bounding box ở khung hình trước `(c_x^{t-1}, c_y^{t-1})`.

**Bước 3**: Ở khung hình hiện tại, xác định vị trí tâm mới `(c_x^t, c_y^t)`. Nếu đoạn thẳng nối tâm cũ với tâm mới cắt qua vạch đếm, hệ thống ghi nhận một sự kiện đếm gồm:
- `frame_idx` và `timestamp`
- `line_name` (tên vạch)
- `direction` (hướng di chuyển): xác định theo dấu của tích có hướng giữa vector vạch đếm và vector di chuyển
- `track_id`, `class_id`, `class_name`
- `confidence` của detection cuối

**Bước 4**: Sau khi được đếm, mỗi cặp `(line_name, track_id)` được đánh dấu trong set `counted`, đảm bảo mỗi phương tiện chỉ được đếm một lần trên mỗi vạch kể cả khi quỹ đạo cắt qua vạch nhiều lần (ví dụ xe tấp vào lề rồi lại chạy ra).

### 3.7.3 Output của hệ thống đếm

Hệ thống xuất hai loại kết quả:
- **CSV sự kiện**: danh sách sự kiện đếm theo thời gian với đầy đủ metadata, dùng cho phân tích offline (theo giờ, theo ngày, theo direction)
- **Video chú thích**: vẽ bounding box màu theo class, track ID, vạch đếm và bảng thống kê overlay realtime

## 3.8 Tiêu chí đánh giá

### 3.8.1 Chỉ số phát hiện

Mô hình phát hiện được đánh giá bằng các chỉ số chuẩn COCO [9], [16]:
- **mAP@0.5**: trung bình AP các lớp ở ngưỡng IoU = 0.5
- **mAP@0.5:0.95**: trung bình AP ở 10 ngưỡng IoU từ 0.5 đến 0.95 (bước 0.05), chỉ số strict hơn
- **AP per class**: AP riêng cho từng lớp (motorcycle, car, bus, truck)
- **Precision, Recall**: ở ngưỡng confidence tối ưu được chọn trên validation

Per-class AP được báo cáo để thấy ảnh hưởng của mất cân bằng lớp thay vì chỉ dùng số trung bình.

### 3.8.2 Chỉ số đếm

Đánh giá đếm bằng sai số giữa số sự kiện dự đoán và ground truth đếm tay trên các đoạn video kiểm thử:
- **MAE** (công thức 2.10): sai số tuyệt đối trung bình
- **MAPE** (công thức 2.11): sai số phần trăm tuyệt đối trung bình

Ground truth đếm tay được thu thập bằng cách chia video test thành các đoạn 1 phút, hai thành viên nhóm độc lập đếm số phương tiện từng lớp đi qua vạch; trường hợp chênh lệch được tham vấn thành viên thứ ba để thống nhất.

### 3.8.3 Chỉ số phân tích lỗi

Phân tích lỗi được thực hiện ở cấp bounding box:
- **TP** (True Positive): detection khớp GT (IoU ≥ 0.5) và đúng class
- **FP** (False Positive): detection không khớp GT hoặc sai class
- **FN** (False Negative): GT không có detection nào match
- **Class confusion**: match vị trí nhưng sai class, thể hiện qua confusion matrix 4×4

TP/FP/FN được tổng hợp theo lớp, theo nguồn dữ liệu và theo kích thước object (small/medium/large theo chuẩn COCO) để xác định điểm yếu cụ thể.

### 3.8.4 Chỉ số tốc độ

FPS (frames per second) được đo trên 100 khung hình đầu sau warm-up với:
- **Precision**: FP16 (half) cho YOLOv8s, FP32 cho Faster R-CNN
- **imgsz**: 960 cho YOLOv8s, 800×1333 (FPN standard) cho Faster R-CNN
- **GPU**: NVIDIA RTX 3500 Ada Laptop (12 GB VRAM)
- **Điều kiện**: bao gồm tiền xử lý, inference, hậu xử lý (NMS)

## 3.9 Tiểu kết chương

Chương này đã trình bày phương pháp đề xuất một cách hệ thống:
- **Dữ liệu**: V5+ gồm 10.203 ảnh hợp nhất từ 5 nguồn Roboflow (SOICT 2024, SOICT Night, Cần Thơ, VNv3, Vietnam Vehicles v10), schema 4 lớp thống nhất.
- **Chia tập theo video**: 7.237 train / 1.466 val / 1.500 test, kiểm tra tự động xác nhận 0% leak.
- **Tiền xử lý và augmentation**: letterbox 960×960 + augmentation mặc định Ultralytics + bổ sung mixup, cos_lr, scale=0.7.
- **Mô hình chính**: YOLOv8s fine-tune ba giai đoạn (V5 honest → V5+ 960 → p4_scale).
- **Mô hình đối chứng**: Faster R-CNN ResNet50-FPN với cấu hình chuẩn TorchVision, cùng dữ liệu V5+.
- **Theo vết và đếm**: ByteTrack + thuật toán giao cắt vạch với anti-duplicate counting.
- **Đánh giá**: mAP@0.5, mAP@0.5:0.95, AP per class cho detection; MAE/MAPE cho counting; phân tích lỗi TP/FP/FN theo nhiều chiều.

Chương 4 trình bày môi trường thực nghiệm, kết quả huấn luyện, so sánh giữa các cấu hình YOLOv8s (ablation), so sánh với Faster R-CNN và phân tích lỗi chi tiết.
