# CHƯƠNG 5  XÂY DỰNG HỆ THỐNG ỨNG DỤNG

Chương này trình bày việc triển khai hệ thống phát hiện và đếm phương tiện thành một ứng dụng demo có giao diện người dùng, sử dụng framework Streamlit. Mục tiêu là đóng gói toàn bộ pipeline (detection + tracking + counting + xuất kết quả) thành sản phẩm mà người dùng không có kiến thức kỹ thuật vẫn có thể sử dụng được.

## 5.1 Kiến trúc tổng thể của ứng dụng

### 5.1.1 Lựa chọn framework

Nhóm đã chọn **Streamlit** [31] làm framework xây dựng giao diện với các lý do sau:
- **Nhanh phát triển**: Streamlit cho phép xây dựng web app chỉ bằng Python thuần, không cần HTML/CSS/JavaScript.
- **Tương thích cao với PyTorch**: các thành phần mô hình đã viết bằng PyTorch và Ultralytics có thể import trực tiếp vào Streamlit app mà không cần API/REST riêng.
- **Có sẵn các widget đồ họa**: slider, file uploader, video player, data table — phù hợp với nhu cầu của ứng dụng.
- **Chạy local**: không cần deploy lên server, phù hợp với mục đích demo cho tiểu luận.

### 5.1.2 Sơ đồ luồng xử lý

**Hình 5.1** Sơ đồ luồng xử lý của ứng dụng Streamlit gồm 5 bước: (1) Upload video, (2) Cấu hình (chọn model, vạch đếm, direction), (3) Chạy pipeline detection + tracking + counting, (4) Hiển thị video chú thích inline + bảng đếm, (5) Tải xuống CSV sự kiện và video MP4.

### 5.1.3 Tổ chức mã nguồn

Ứng dụng được triển khai tại `ui/streamlit_app.py` (~830 dòng) với các module hỗ trợ:
- `src/pipeline/run.py` — hàm `run_pipeline()` thực thi detection + tracking + counting
- `src/tracking/counter.py` — class `Line` định nghĩa vạch đếm và logic crossing detection
- `src/stats/aggregator.py` — tổng hợp events thành bảng counts, tính flow rate, peak period
- `src/evaluation/metrics.py` — tính MAE, MAPE khi có ground truth đếm tay

## 5.2 Các chức năng chính

### 5.2.1 Tải video đầu vào

Người dùng có hai cách cung cấp video:
- **Upload file**: hỗ trợ MP4, MOV, AVI với kích thước tối đa 500 MB
- **Chọn từ thư mục test**: liệt kê các video mẫu đã có sẵn trong `data/test_videos/` để dùng demo nhanh

Khi upload, file được lưu tạm vào `/tmp/` và được hiển thị dạng preview ở panel chính.

### 5.2.2 Lựa chọn mô hình

Panel sidebar cho phép chọn model từ danh sách các checkpoint đã huấn luyện. Ứng dụng tự động quét `weights/` và `runs/detect/*/weights/best.pt` để lập danh sách. Các mô hình được khuyến nghị (featured) được đưa lên đầu với ghi chú mô tả:

**Bảng 5.1** Danh mục model có sẵn trong ứng dụng.

| Model | mAP@0.5 test | Mô tả |
|---|---|---|
| ⭐ V5+ (mặc định) | 0,782 | Best all-around, imgsz=960 |
| V6 pseudo | 0,761 | Motor cải thiện (dù mAP tổng thấp hơn) |
| V5 honest | 0,665 | Baseline honest split |
| p4_scale | 0,787 | Mô hình cuối (khuyến nghị cho đánh giá) |

Người dùng cũng có thể chọn model V3 VN (mAP val 0,951) với cảnh báo rằng số liệu này **có thể leak** và không đáng tin cậy — thể hiện tính minh bạch của ứng dụng.

### 5.2.3 Cấu hình vạch đếm

Ứng dụng hỗ trợ hai chế độ cấu hình vạch:

**Chế độ Baseline (đơn giản)**:
- Một vạch đếm ngang tại tọa độ Y tuỳ chọn (slider)
- Chỉ đếm theo một chiều xuống (phương tiện đi từ phía trên vạch xuống phía dưới)
- Dùng cho test nhanh

**Chế độ Advanced (đa vạch, đa chiều)**:
- Nhiều vạch đếm với tên tùy chọn (`line_1`, `line_2`, ...)
- Mỗi vạch định nghĩa bằng hai điểm (x₁, y₁, x₂, y₂)
- Chọn hướng đếm cho từng vạch: `both`, `left_to_right`, `right_to_left`
- Phù hợp với các ngã tư đa luồng hoặc đo lưu lượng theo direction

### 5.2.4 Hiển thị trực quan

Trong quá trình xử lý, ứng dụng hiển thị:
- **Progress bar** theo khung hình đã xử lý
- **Preview frame đầu tiên** có vẽ vạch đếm để người dùng xác nhận trước khi chạy full
- **Bảng thống kê realtime** cập nhật mỗi 100 khung hình: tổng số lượt đếm, chia theo lớp

Sau khi hoàn tất, panel chính hiển thị:
- **Video chú thích inline**: vẽ bounding box màu theo class (motorcycle: xanh lá, car: xanh dương, bus: vàng, truck: đỏ), track ID bền vững, vạch đếm, bảng counts overlay realtime trên khung hình
- **Bảng tổng hợp counts** theo lớp × line × direction

### 5.2.5 Xuất kết quả

Ứng dụng cho phép tải xuống:
- **Video MP4** đã chú thích (H.264 codec để browser play được, đã transcode qua ffmpeg)
- **CSV sự kiện**: mỗi dòng là một sự kiện đếm với các cột `frame, time_sec, line, direction, track_id, class_name, confidence`
- **JSON summary**: tổng hợp counts theo nhiều chiều (theo lớp, theo line, theo direction, theo time bucket)

### 5.2.6 Phân tích chuyên sâu (optional)

Nếu người dùng upload ground truth CSV đếm tay, ứng dụng tự động tính:
- **MAE** và **MAPE** tổng thể và theo lớp
- **Confusion matrix** của detection (nếu có bounding box ground truth)
- **Biểu đồ flow rate** theo thời gian (số lượt đếm mỗi phút)
- **Peak period detection** — xác định giờ cao điểm

## 5.3 Giao diện người dùng

**Hình 5.2** Ảnh chụp màn hình trang chính của ứng dụng Streamlit:
- Panel sidebar bên trái: chọn pipeline (Baseline/Advanced), chọn model, chọn video, cấu hình vạch đếm
- Panel chính bên phải: hiển thị video preview, bảng tổng counts, nút tải xuống kết quả

**Hình 5.3** Panel cấu hình vạch đếm Advanced mode với sliders cho (x₁, y₁, x₂, y₂) và dropdown chọn direction.

**Hình 5.4** Panel hiển thị video chú thích với bounding box màu theo class, track ID bền vững qua các khung hình, vạch đếm và bảng counts overlay realtime.

**Hình 5.5** Bảng CSV sự kiện đếm xuất ra, mỗi dòng là một phương tiện đi qua vạch với đầy đủ metadata.

## 5.4 Kiểm thử và đánh giá hệ thống

### 5.4.1 Kiểm thử chức năng

**Bảng 5.2** Kết quả kiểm thử các chức năng chính của ứng dụng trên các trường hợp đại diện.

| Chức năng | Trường hợp test | Kết quả | Ghi chú |
|---|---|---|---|
| Upload video | MP4 500 MB | ✓ Pass | Lưu vào /tmp/ thành công |
| Upload video | AVI 100 MB | ✓ Pass | Tự động transcode sang MP4 khi cần |
| Chọn model | V5+ best.pt (22 MB) | ✓ Pass | Load trong < 2 giây |
| Chọn model | FRCNN best.pt (158 MB) | ✓ Pass | Load trong ~5 giây |
| Cấu hình vạch | 1 vạch ngang | ✓ Pass | Preview hiển thị đúng vị trí |
| Cấu hình vạch | 3 vạch đa chiều | ✓ Pass | Mỗi vạch đếm độc lập |
| Xử lý video | Video 1 phút (1500 frame) | ✓ Pass | Hoàn tất trong ~45 giây |
| Xuất CSV | 400 sự kiện | ✓ Pass | Encoding UTF-8-BOM, đọc được trong Excel |
| Xuất MP4 | Video 1 phút | ✓ Pass | Play inline trên Chrome/Edge |
| Resume | Ngắt kết nối giữa chừng | ✗ Fail | Streamlit mất state, phải chạy lại — hạn chế |

### 5.4.2 Đánh giá tốc độ xử lý

**Bảng 5.3** Tốc độ xử lý end-to-end của ứng dụng trên GPU RTX 3500 Ada 12GB.

| Video | Số frame | FPS source | Thời gian xử lý | Tốc độ xử lý |
|---|---|---|---|---|
| soict_test_1min.mp4 | 1.500 | 25 | 42 giây | **35,7 FPS** |
| cantho_test_30s.mp4 | 900 | 30 | 28 giây | 32,1 FPS |
| hcm_rush_hour.mp4 (đông) | 3.000 | 25 | 2 phút 10 giây | 23,1 FPS |
| danang_night.mp4 | 4.500 | 30 | 2 phút 35 giây | 29,0 FPS |

Tốc độ xử lý trung bình ~30 FPS, đủ xử lý realtime cho video 25 FPS tiêu chuẩn. Trường hợp dense scene (giờ cao điểm) chậm xuống còn 23 FPS do số detection nhiều hơn làm NMS và tracking chậm hơn.

### 5.4.3 Hạn chế của hệ thống

Trong quá trình kiểm thử, nhóm ghi nhận các hạn chế sau:
- **Không hỗ trợ realtime streaming**: ứng dụng chỉ xử lý video file đã có, chưa kết nối trực tiếp với camera IP RTSP
- **Streamlit mất state khi reload**: cần chạy lại pipeline nếu trình duyệt mất kết nối
- **Giới hạn file 500 MB**: do cấu hình Streamlit mặc định, có thể tăng nhưng tốn RAM
- **Chưa có authentication**: ứng dụng demo chạy local, không phù hợp deploy public
- **Không có database**: mỗi lần chạy phải upload lại video, kết quả chỉ lưu tạm

## 5.5 Hướng triển khai thực tế

Để đưa hệ thống vào triển khai thực tế, cần bổ sung:
- **Kết nối camera IP**: thay thế input video file bằng RTSP stream reader
- **Database và session management**: lưu trữ kết quả, cho phép query theo thời gian
- **REST API**: tách backend ra khỏi UI Streamlit để nhiều client có thể truy cập đồng thời
- **Containerization**: đóng gói bằng Docker để triển khai trên server
- **Monitoring**: logging với Prometheus/Grafana để theo dõi hiệu năng

Các hướng này nằm ngoài phạm vi đề tài nhưng được đề xuất cho giai đoạn phát triển tiếp theo.

## 5.6 Tiểu kết chương

Chương này đã trình bày việc đóng gói pipeline phát hiện và đếm phương tiện thành ứng dụng Streamlit có giao diện trực quan. Ứng dụng hỗ trợ đầy đủ các chức năng từ upload video, cấu hình vạch đếm (đa vạch, đa chiều), chạy pipeline (detection + ByteTrack + counting), đến xuất kết quả dạng CSV và video chú thích. Tốc độ xử lý trung bình ~30 FPS trên GPU RTX 3500 Ada đủ cho ứng dụng thời gian thực. Các kiểm thử chức năng và hiệu năng xác nhận tính ổn định của hệ thống trong phạm vi demo.
