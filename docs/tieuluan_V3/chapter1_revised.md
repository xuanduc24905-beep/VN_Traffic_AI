# CHƯƠNG 1  TỔNG QUAN ĐỀ TÀI

## 1.1 Bối cảnh và lý do chọn đề tài

Hệ thống camera giám sát giao thông đang được triển khai rộng rãi tại các đô thị lớn của Việt Nam, cung cấp một lượng lớn dữ liệu hình ảnh và video theo thời gian thực. Các dữ liệu này nếu được xử lý tự động có thể hỗ trợ quy hoạch giao thông, điều phối tín hiệu, phân tích cơ cấu phương tiện và dự báo ùn tắc. Tuy nhiên, phần lớn hệ thống hiện tại mới chỉ dừng ở mức ghi hình và hiển thị thụ động cho người vận hành, trong khi khả năng trích xuất thông tin tự động vẫn còn hạn chế.

Phân loại toàn bộ một ảnh thành một nhãn bằng mạng nơ-ron tích chập (CNN) truyền thống không phù hợp với cảnh giao thông chứa nhiều phương tiện. Trong cùng một khung hình thường xuất hiện đồng thời xe máy, ô tô con, xe buýt và xe tải ở nhiều vị trí khác nhau. Vì vậy, bài toán cần giải quyết là *phát hiện đối tượng đa lớp* (multi-class object detection): xác định đồng thời vị trí bounding box và nhãn lớp cho từng phương tiện trong ảnh. Các bộ phát hiện dựa trên CNN như YOLO [17] và Faster R-CNN [1] đã được chứng minh có khả năng thực hiện đồng thời hai yêu cầu này với độ chính xác cao.

Dữ liệu giao thông Việt Nam đặt ra nhiều thách thức đặc thù. Mật độ xe máy trên đường thường chiếm trên 70% tổng số phương tiện, dẫn đến hiện tượng các xe che khuất lẫn nhau. Khoảng cách từ camera đến phương tiện biến thiên lớn khiến nhiều đối tượng có kích thước nhỏ trong khung hình (dưới 32×32 pixel). Điều kiện ánh sáng cũng rất đa dạng do kết hợp camera ngày và đêm. Ngoài ra, phân bố giữa các lớp xe lệch mạnh: xe buýt và xe tải thường chiếm dưới 5% tổng số đối tượng, khiến các chỉ số trung bình không phản ánh đúng chất lượng của các lớp ít mẫu.

Nhóm lựa chọn YOLOv8 làm mô hình chính vì có tốc độ xử lý cao phù hợp với video thời gian thực và cộng đồng nghiên cứu lớn với mã nguồn mở Ultralytics được bảo trì tích cực. Faster R-CNN với backbone ResNet50-FPN được chọn làm mô hình đối chứng do đại diện cho hướng tiếp cận hai giai đoạn kinh điển, cho phép đánh giá sự chênh lệch giữa hai triết lý thiết kế detector trên cùng một tập dữ liệu. Hai mô hình được huấn luyện và đánh giá trong điều kiện phần cứng và tập dữ liệu đồng nhất để đảm bảo so sánh công bằng.

## 1.2 Vấn đề cần giải quyết

Đề tài xác định ba vấn đề cốt lõi cần giải quyết, phản ánh ba thành phần chính của một hệ thống phát hiện phương tiện thực dụng.

**Vấn đề thứ nhất — xây dựng tập dữ liệu nhất quán và không rò rỉ.** Dữ liệu giao thông Việt Nam công khai hiện có trên các nền tảng như Roboflow và SOICT đến từ nhiều camera với các quy ước gán nhãn khác nhau. Cần xác định tiêu chí phân biệt rõ ràng giữa từng loại phương tiện, quy đổi nhãn về một schema thống nhất, loại bỏ (thay vì gộp) các lớp không chắc chắn, đồng thời kiểm soát hiện tượng rò rỉ dữ liệu (data leakage) do nhiều khung hình liên tiếp thuộc cùng một video bị tách vào các tập train và test khác nhau.

**Vấn đề thứ hai — lựa chọn và tinh chỉnh mô hình cho cảnh giao thông Việt Nam.** Mô hình cần hạn chế cả dự đoán sai (false positive) và bỏ sót (false negative), đặc biệt với xe máy nhỏ và các xe bị che khuất một phần. Việc tinh chỉnh cần cân nhắc giữa kích thước ảnh đầu vào (ảnh hưởng tốc độ và độ chính xác), chính sách tăng cường dữ liệu phù hợp với cảnh dày đặc, và chiến lược huấn luyện nhiều giai đoạn.

**Vấn đề thứ ba — đánh giá đúng và báo cáo trung thực.** Trong các báo cáo trước đó của nhóm (các phiên bản V1 đến V4), do chia dữ liệu theo khung hình thay vì theo video đã dẫn đến hiện tượng rò rỉ: các khung hình liên tiếp từ cùng một video xuất hiện cả trong train và test. Hệ quả là mAP@0.5 đạt mức ảo khoảng 0,95 nhưng không phản ánh năng lực thực của mô hình. Đề tài ở phiên bản V5 trở đi áp dụng chia tập theo video kèm kiểm tra rò rỉ tự động để các kết quả thực nghiệm phản ánh đúng khả năng tổng quát hóa.

## 1.3 Mục tiêu nghiên cứu

### 1.3.1 Mục tiêu tổng quát

Xây dựng và đánh giá một hệ thống phát hiện, phân loại và đếm phương tiện giao thông từ video bằng mô hình CNN, trong đó YOLOv8s tinh chỉnh đóng vai trò mô hình chính và Faster R-CNN ResNet50-FPN đóng vai trò mô hình đối chứng. Hệ thống hướng tới bốn loại phương tiện phổ biến (xe máy, ô tô con, xe buýt, xe tải) và cung cấp giao diện trực quan phục vụ thống kê cơ cấu phương tiện theo thời gian.

### 1.3.2 Mục tiêu cụ thể

- Xây dựng bộ dữ liệu V5+ gồm 10.203 ảnh (train 7.237, validation 1.466, test 1.500) hợp nhất từ 5 nguồn Roboflow công khai; gán nhãn 4 lớp theo chuẩn YOLO.
- Chuẩn hóa dữ liệu và chia train/validation/test theo video_id, áp dụng kiểm tra rò rỉ tự động xác nhận 0% leak giữa các tập.
- Tinh chỉnh YOLOv8s từ trọng số pretrained COCO qua ba giai đoạn (V5 honest 640px, V5+ 960px, p4_scale finetune) và huấn luyện Faster R-CNN ResNet50-FPN trên cùng tập dữ liệu để so sánh.
- Đánh giá Precision, Recall, mAP@0.5, mAP@0.5:0.95 tổng thể và theo từng lớp trên tập test độc lập.
- Đo FPS và độ trễ inference trên GPU RTX 3500 Ada Laptop cho cả hai mô hình trong điều kiện thống nhất.
- Xây dựng hệ thống đếm phương tiện dựa trên tracker ByteTrack và vạch đếm ảo; đánh giá bằng MAE và MAPE trên các đoạn video có nhãn đếm tay.
- Triển khai giao diện Streamlit cho phép tải video, cấu hình vạch đếm, hiển thị bounding box và xuất kết quả đếm theo lớp dưới dạng CSV.

## 1.4 Câu hỏi nghiên cứu

**Câu hỏi 1.** Trên cùng tập test V5+ gồm 1.500 ảnh với 4 lớp phương tiện, YOLOv8s tinh chỉnh và Faster R-CNN ResNet50-FPN khác nhau như thế nào về mAP@0.5, mAP@0.5:0.95, kết quả theo từng lớp và tốc độ xử lý trên cùng phần cứng?

**Câu hỏi 2.** Khi giữ nguyên kiến trúc YOLOv8s và cách chia dữ liệu, việc tăng độ phân giải ảnh từ 640 lên 960 pixel, áp dụng cosine learning rate schedule và augmentation không gian mạnh hơn (scale=0,7; translate=0,15) ảnh hưởng như thế nào đến mAP@0.5 và khoảng cách val–test?

**Câu hỏi 3.** Các lỗi nhận dạng nào xuất hiện phổ biến đối với phương tiện nhỏ, bị che khuất hoặc trong điều kiện ánh sáng yếu? Các lỗi thuộc nhóm định vị sai, nhầm lớp hay bỏ sót?

**Câu hỏi 4.** Mô hình cuối (p4_scale) có đạt mục tiêu mAP@0.5 ≥ 0,75 trên tập test hay không, và mức độ cải thiện so với baseline COCO zero-shot và V5 honest là bao nhiêu điểm phần trăm?

**Câu hỏi 5.** Hệ thống đếm phương tiện xây dựng trên nền ByteTrack đạt sai số MAE và MAPE bao nhiêu trên các đoạn video kiểm thử có nhãn đếm tay? Lỗi đếm tập trung ở lớp xe nào và trong điều kiện nào?

## 1.5 Đối tượng nghiên cứu

Đối tượng lý thuyết gồm mạng nơ-ron tích chập (CNN), học chuyển giao, phát hiện đối tượng một giai đoạn (one-stage detector), phát hiện đối tượng hai giai đoạn (two-stage detector) và các tiêu chí đánh giá định vị–phân loại. YOLOv8s và Faster R-CNN ResNet50-FPN là hai kiến trúc đại diện được khảo sát, không phải hai tầng chạy nối tiếp trong cùng pipeline.

Đối tượng thực nghiệm là ảnh và khung hình video chứa phương tiện giao thông đường bộ thuộc bốn lớp: xe máy (motorcycle), ô tô con (car), xe buýt (bus) và xe tải (truck). Đơn vị gán nhãn là từng phương tiện với một bounding box dạng (x_center, y_center, width, height) chuẩn hóa về khoảng [0, 1] và một nhãn lớp duy nhất.

Hệ thống đếm sử dụng thuật toán tracker ByteTrack [19] để liên kết bounding box qua các khung hình liên tiếp, kết hợp với kỹ thuật vạch đếm ảo (virtual counting line) để đếm phương tiện đi qua một mặt cắt cho trước của khung hình.

## 1.6 Phạm vi nghiên cứu

**Về dữ liệu.** Đề tài sử dụng bộ V5+ hợp nhất từ 5 nguồn Roboflow công khai: SOICT 2024 (camera ngày TP.HCM và Hà Nội), SOICT Night (camera đêm), Vehicle Vietnam-CanTho (Cần Thơ), Vietnamese Vehicle v3 (đa nguồn) và Vietnam Vehicles v10 (visionvn). Toàn bộ dữ liệu được quy đổi về schema 4 lớp theo định dạng YOLO chuẩn.

**Về mô hình.** Đề tài tập trung vào hai kiến trúc đại diện: YOLOv8s (biến thể small) và Faster R-CNN với backbone ResNet50-FPN. Các biến thể YOLOv8m, YOLOv8l và các kiến trúc khác (RT-DETR, YOLOv11) không nằm trong phạm vi do giới hạn thời gian và tài nguyên GPU.

**Về phần cứng.** Thực nghiệm được thực hiện trên một GPU NVIDIA RTX 3500 Ada Generation Laptop (12 GB VRAM). Các so sánh về tốc độ FPS và độ trễ chỉ có giá trị trong phạm vi phần cứng này.

**Về nhiệm vụ.** Đề tài giải quyết bốn nhiệm vụ: (1) phát hiện phương tiện, (2) phân loại phương tiện theo 4 lớp, (3) theo vết (tracking) phương tiện qua nhiều khung hình, và (4) đếm số lượng phương tiện đi qua một vạch ảo. Các nhiệm vụ nhận dạng biển số, đo tốc độ phương tiện và phát hiện vi phạm không nằm trong phạm vi.

## 1.7 Phương pháp nghiên cứu

Đề tài áp dụng phương pháp nghiên cứu thực nghiệm kết hợp đối chiếu với tài liệu. Các bước chính gồm:

1. **Khảo sát lý thuyết và công trình liên quan** về các mô hình phát hiện đối tượng dựa trên CNN, các bộ dữ liệu giao thông và các tiêu chí đánh giá, trình bày trong Chương 2.
2. **Xây dựng và kiểm soát chất lượng dữ liệu**: hợp nhất 5 nguồn, quy đổi nhãn về schema 4 lớp, chia tập theo video_id và kiểm tra rò rỉ tự động (Chương 3).
3. **Thiết kế thực nghiệm có đối chứng** theo nguyên tắc ablation: mỗi cấu hình chỉ thay đổi một nhóm tham số so với baseline để có thể quy gán sự thay đổi kết quả cho yếu tố cụ thể.
4. **Huấn luyện và đánh giá** hai mô hình YOLOv8s và Faster R-CNN trên cùng tập dữ liệu, cùng phần cứng (Chương 4).
5. **Phân tích lỗi** bằng confusion matrix, phân loại các trường hợp thất bại và đối chiếu với giả thuyết ban đầu.
6. **Triển khai hệ thống đếm** với ByteTrack và đánh giá bằng MAE/MAPE so với nhãn đếm tay (Chương 5).

## 1.8 Sản phẩm đã thực hiện

Đề tài đã hoàn thành bốn sản phẩm chính:

- **Bộ dữ liệu V5+** gồm 10.203 ảnh với nhãn chuẩn YOLO 4 lớp, được kiểm tra rò rỉ tự động xác nhận 0% leak và lưu trữ công khai tại thư mục `data/merged_v5plus/`.
- **Pipeline huấn luyện có đối chứng** gồm 8 recipe trong `scripts/pipelines/train.py`, cho phép chạy ablation đồng nhất; đã huấn luyện và đánh giá 5 mô hình YOLOv8s (V5 honest, V5+, p4_scale, p4_imgsz1280, V6 pseudo) và 1 mô hình Faster R-CNN ResNet50-FPN.
- **Mô hình phát hiện tốt nhất p4_scale** đạt mAP@0.5 = 0,787 và mAP@0.5:0.95 = 0,423 trên tập test V5+ 1.500 ảnh, cao hơn mô hình đối chứng Faster R-CNN tương ứng 0,146 và 0,062 điểm.
- **Ứng dụng demo Streamlit** (`ui/streamlit_app.py`) cho phép tải video, cấu hình vạch đếm, chạy phát hiện kết hợp ByteTrack và xuất bảng đếm theo lớp dạng CSV.

## 1.9 Ý nghĩa và đóng góp đã đạt được

**Về mặt phương pháp.** Đề tài đã xác định và khắc phục được hiện tượng rò rỉ dữ liệu ở mức video — một lỗi phổ biến nhưng ít được báo cáo trong các công trình phát hiện phương tiện trước đây. Kết quả cho thấy việc chuyển từ chia tập theo khung hình sang chia theo video_id làm giảm mAP@0.5 từ mức ảo 0,95 (V1–V4) xuống 0,665 (V5 honest) nhưng phản ánh đúng khả năng tổng quát hóa của mô hình.

**Về mặt thực nghiệm.** Đề tài đã cung cấp bộ số liệu đối chiếu giữa YOLOv8s và Faster R-CNN ResNet50-FPN trên cùng tập test V5+ với 1.500 ảnh, thể hiện rõ sự khác biệt giữa hai triết lý thiết kế detector trong bối cảnh dữ liệu giao thông Việt Nam dày đặc xe máy.

**Về mặt ứng dụng.** Hệ thống đếm phương tiện được xây dựng có thể triển khai lên các camera giám sát hiện có để tổng hợp cơ cấu phương tiện theo thời gian, hỗ trợ các bài toán quy hoạch và điều phối giao thông.

## 1.10 Cấu trúc báo cáo

Báo cáo gồm năm chương, phần tài liệu tham khảo và phụ lục:

- **Chương 1** trình bày tổng quan đề tài, mục tiêu, câu hỏi nghiên cứu và sản phẩm đã đạt được.
- **Chương 2** trình bày các công trình liên quan và cơ sở lý thuyết về CNN, học chuyển giao, YOLOv8, Faster R-CNN và các tiêu chí đánh giá.
- **Chương 3** mô tả phương pháp đề xuất gồm pipeline xử lý video, thiết kế dữ liệu, chiến lược huấn luyện YOLOv8s và cấu hình Faster R-CNN đối chứng.
- **Chương 4** trình bày môi trường thực nghiệm, kết quả huấn luyện, so sánh giữa các cấu hình YOLOv8s, so sánh với Faster R-CNN và phân tích lỗi chi tiết.
- **Chương 5** mô tả hệ thống đếm phương tiện xây dựng trên Streamlit, kiểm thử giao diện và đánh giá sai số đếm bằng MAE/MAPE.
- **Kết luận** tổng kết các đóng góp, chỉ ra hạn chế và đề xuất hướng phát triển tiếp theo.
