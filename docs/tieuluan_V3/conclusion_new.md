# KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 1. Tổng kết đóng góp

Đồ án đã hoàn thành việc xây dựng một hệ thống phát hiện, phân loại và đếm phương tiện giao thông Việt Nam từ video sử dụng mô hình CNN. Qua quá trình thực hiện, nhóm đã đạt được bốn đóng góp chính:

**Đóng góp 1 — Phát hiện và khắc phục hiện tượng rò rỉ dữ liệu cấp video**. Nhóm đã nhận diện được một lỗi phổ biến nhưng ít được báo cáo trong các công trình phát hiện phương tiện: chia tập ngẫu nhiên theo khung hình thay vì theo video_id. Trong các phiên bản đầu của đề tài (V1–V4), hiện tượng này đã gây ra mAP ảo khoảng 0,95 trên test. Sau khi rebuild dataset V5 theo nguyên tắc chia cấp video và áp dụng kiểm tra tự động, mAP honest giảm xuống 0,665 nhưng phản ánh đúng khả năng tổng quát hóa. Đây là bài học methodology quan trọng cho các nghiên cứu phát hiện phương tiện sử dụng dữ liệu video.

**Đóng góp 2 — Xây dựng và công bố bộ dữ liệu V5+ sạch**. Bộ dữ liệu V5+ gồm 10.203 ảnh hợp nhất từ năm nguồn Roboflow công khai với schema 4 lớp thống nhất, chia tập theo video_id và kiểm tra rò rỉ tự động xác nhận 0% leak. Bộ dữ liệu này có thể làm baseline cho các nghiên cứu tiếp theo về phát hiện phương tiện giao thông Việt Nam.

**Đóng góp 3 — So sánh định lượng YOLOv8s và Faster R-CNN trên dữ liệu VN**. Trên cùng tập test V5+ với 1.500 ảnh và cùng phần cứng GPU RTX 3500 Ada, mô hình YOLOv8s (p4_scale) đạt mAP@0.5 = 0,787, vượt Faster R-CNN ResNet50-FPN 14,6 điểm (0,641) và nhanh hơn ~10× (55,6 FPS vs 5,4 FPS). Kết quả này cung cấp bằng chứng cụ thể cho câu hỏi "chọn YOLO hay Faster R-CNN cho bài toán phát hiện phương tiện VN", thay vì chỉ dựa vào benchmark quốc tế.

**Đóng góp 4 — Ứng dụng demo có thể sử dụng**. Hệ thống đã được đóng gói thành ứng dụng Streamlit hỗ trợ đầy đủ pipeline từ detection, tracking (ByteTrack), đếm phương tiện qua vạch ảo đa chiều, đến xuất CSV sự kiện và video chú thích. Tốc độ xử lý ~30 FPS đủ cho ứng dụng thời gian thực. Sai số đếm MAPE = 5,1% trung bình trên các đoạn video kiểm thử có ground truth đếm tay (4,2% ban ngày và 8,0% ban đêm).

## 2. Hạn chế của đề tài

Mặc dù đã đạt các mục tiêu chính, đề tài vẫn có các hạn chế sau cần ghi nhận:

**Hạn chế 1 — Phạm vi địa lý hạn chế**. Dữ liệu V5+ chủ yếu từ TP.HCM, Cần Thơ, cùng một phần snapshot CCTV của Vietnam Vehicles v10 phân bố trên nhiều camera ở miền Nam. Các khu vực khác như Hà Nội, Đà Nẵng, miền Trung và miền núi phía Bắc chưa được đại diện đầy đủ, nên mô hình có thể bias về domain đô thị miền Nam.

**Hạn chế 2 — Đánh giá ngoài phân phối (OOD) chỉ định tính**. Video CCTV Đà Nẵng (29.195 khung hình) chưa được gán nhãn bounding box chi tiết, do đó chỉ thực hiện đánh giá định tính và counting (5 đoạn 1 phút). Không thể báo cáo chính xác mức suy giảm mAP cross-domain.

**Hạn chế 3 — Chưa có nhãn tracking per-frame**. Do không có nhãn tracking (IDs theo thời gian) chuẩn, không thể đánh giá tracker ByteTrack bằng các chỉ số chính thức như IDF1, MOTA, HOTA. Chất lượng tracker chỉ được đánh giá gián tiếp qua chất lượng counting.

**Hạn chế 4 — Số lớp giới hạn**. Schema 4 lớp (motorcycle, car, bus, truck) bỏ qua các loại phương tiện quan trọng khác như xe đạp, xe ba gác, xe lam, container. Mở rộng schema sẽ cần thu thập nhãn mới và không thể tái sử dụng hoàn toàn các checkpoint hiện tại.

**Hạn chế 5 — Thí nghiệm self-training thất bại**. Mô hình V6 pseudo (bổ sung 4.396 ảnh pseudo-label) không cải thiện so với V5+ (0,761 < 0,782) do khuếch đại class imbalance. Chưa có giải pháp được thực nghiệm để khắc phục.

**Hạn chế 6 — Phần cứng hạn chế**. Toàn bộ thực nghiệm chạy trên một GPU laptop 12 GB VRAM, không thử nghiệm các biến thể lớn hơn (YOLOv8m/l/x, Faster R-CNN với ResNet101/152). Kết quả có thể khác nếu có GPU mạnh hơn.

## 3. Hướng phát triển

Dựa trên các hạn chế đã nêu, nhóm đề xuất các hướng phát triển sau:

**Hướng 1 — Mở rộng dữ liệu đa vùng**. Thu thập bổ sung dữ liệu từ Hà Nội, Đà Nẵng, Hải Phòng và các tỉnh miền núi để tăng tính đại diện. Đặc biệt cần tăng số lượng mẫu cho các lớp hiếm (bus, truck) hiện chỉ chiếm 3–4% tập dữ liệu.

**Hướng 2 — Gán nhãn OOD Đà Nẵng và đánh giá định lượng cross-domain**. Gán nhãn ít nhất 500 khung hình mẫu từ video Đà Nẵng, sau đó đánh giá mô hình trên tập này để có mức suy giảm chính xác và xác định các điểm yếu cross-domain cụ thể.

**Hướng 3 — Kiến trúc YOLOv8s-P2 cho vật nhỏ**. Phân tích lỗi cho thấy recall ở đối tượng small (<32² pixel) chỉ đạt 74,1% so với 94,7% ở large. Kiến trúc YOLOv8s-P2 (thêm detection head stride=4) được thiết kế đặc biệt cho vật nhỏ, có cơ sở lý thuyết tốt để thử nghiệm.

**Hướng 4 — Cân bằng class-weighted loss**. Mất cân bằng lớp (motorcycle 69%, truck 3%) là vấn đề cốt lõi. Áp dụng `cls_pw > 1` cho các lớp hiếm hoặc Focal Loss có thể cải thiện AP của bus/truck.

**Hướng 5 — Pseudo-labeling có cân bằng**. Thí nghiệm V6 pseudo thất bại do CCTV HCM chủ yếu là xe máy. Phương pháp pseudo-labeling có **cân bằng** (balanced pseudo-labeling): chỉ lấy pseudo của các lớp thiếu để bổ sung, giữ nguyên distribution V5+ gốc, có thể khắc phục vấn đề này.

**Hướng 6 — Tích hợp camera IP realtime**. Thay thế input file video bằng RTSP stream reader để kết nối trực tiếp với camera giám sát hiện có. Kết hợp với REST API backend sẽ cho phép nhiều client query kết quả đồng thời.

**Hướng 7 — Theo dõi đa camera**. Mở rộng tracking từ single-camera sang multi-camera để theo dõi một phương tiện qua nhiều góc camera khác nhau. Có thể áp dụng các kỹ thuật ReID (Re-Identification) đã có sẵn.

**Hướng 8 — Đo lưu lượng theo làn**. Hiện tại hệ thống đếm tổng theo vạch mà không phân biệt làn đường. Mở rộng cấu hình để định nghĩa từng làn riêng biệt sẽ cung cấp thông tin chi tiết hơn cho các bài toán quy hoạch giao thông.

## 4. Lời kết

Đồ án đã hoàn thành đầy đủ các mục tiêu đã đặt ra: xây dựng bộ dữ liệu V5+ sạch, huấn luyện và so sánh YOLOv8s với Faster R-CNN, đánh giá chính xác không rò rỉ trên tập test, xây dựng ứng dụng demo có thể sử dụng. Kết quả thực nghiệm cho thấy YOLOv8s tinh chỉnh với cấu hình p4_scale đạt mAP@0.5 = 0,787 trên test honest V5+, vượt mô hình đối chứng Faster R-CNN cả về độ chính xác lẫn tốc độ, và đáp ứng mục tiêu nghiên cứu đã đề ra.

Bài học quan trọng nhất của đề tài không nằm ở con số mAP cuối cùng, mà ở phương pháp xây dựng dữ liệu đúng và đánh giá honest. Thay vì công bố mAP ảo 0,95 từ phiên bản V1–V4, nhóm chọn báo cáo mAP honest 0,787 sau khi khắc phục rò rỉ dữ liệu — một quyết định thể hiện tính trung thực khoa học cần có trong nghiên cứu machine learning thực hành.

Các hướng phát triển đã đề xuất mở ra nhiều khả năng cải tiến tiếp theo, đặc biệt là mở rộng dữ liệu đa vùng, kiến trúc chuyên biệt cho vật nhỏ, và triển khai realtime trên camera IP. Nhóm hy vọng bộ dữ liệu, mã nguồn và kết quả thực nghiệm của đề tài sẽ là tài liệu tham khảo hữu ích cho các nghiên cứu tiếp theo về phát hiện phương tiện giao thông tại Việt Nam.
