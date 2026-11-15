# KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 1. Tổng kết đóng góp

Đồ án đã hoàn thành việc xây dựng một hệ thống phát hiện, phân loại và đếm phương tiện giao thông Việt Nam từ video sử dụng mô hình CNN. Qua quá trình thực hiện, nhóm đã đạt được bốn đóng góp chính:

**Đóng góp 1 — Phát hiện và khắc phục hiện tượng rò rỉ dữ liệu cấp video**. Nhóm đã nhận diện được một lỗi phổ biến nhưng ít được báo cáo trong các công trình phát hiện phương tiện: chia tập ngẫu nhiên theo khung hình thay vì theo video_id. Trong các phiên bản đầu của đề tài (V1–V4), hiện tượng này đã gây ra mAP ảo khoảng 0,95 trên test. Sau khi rebuild dataset V5 theo nguyên tắc chia cấp video và áp dụng kiểm tra tự động, mAP honest giảm xuống 0,665 nhưng phản ánh đúng khả năng tổng quát hóa. Đây là bài học methodology quan trọng cho các nghiên cứu phát hiện phương tiện sử dụng dữ liệu video.

**Đóng góp 2 — Xây dựng và công bố bộ dữ liệu V5+ sạch**. Bộ dữ liệu V5+ gồm 10.203 ảnh hợp nhất từ năm nguồn Roboflow công khai với schema 4 lớp thống nhất, chia tập theo video_id và kiểm tra rò rỉ tự động xác nhận 0% leak. Bộ dữ liệu này có thể làm baseline cho các nghiên cứu tiếp theo về phát hiện phương tiện giao thông Việt Nam.

**Đóng góp 3 — So sánh định lượng YOLOv8s và Faster R-CNN trên dữ liệu VN**. Trên cùng tập test V5+ với 1.500 ảnh, mô hình YOLOv8s `r_clspw` đạt mAP@0.5 = 0,7927, vượt Faster R-CNN ResNet50-FPN 15,19 điểm tuyệt đối (0,6408). `r_clspw` cũng cao hơn `p4_scale` 0,58 điểm tuyệt đối (0,74% tương đối), qua đó xác nhận đóng góp của recipe class-weighted loss cho bài toán mất cân bằng lớp.

**Đóng góp 4 — Ứng dụng demo có thể sử dụng**. Hệ thống đã được đóng gói thành ứng dụng Streamlit hỗ trợ đầy đủ pipeline từ detection, tracking (ByteTrack), đếm phương tiện qua vạch ảo đa chiều, đến xuất CSV sự kiện và video chú thích. Tốc độ xử lý ~30 FPS đủ cho ứng dụng thời gian thực. Sai số đếm MAPE tổng là 5,1% trên các đoạn video kiểm thử có ground truth đếm tay.

## 2. Hạn chế của đề tài

Mặc dù đã đạt các mục tiêu chính, đề tài vẫn có các hạn chế sau cần ghi nhận:

**Hạn chế 1 — Phạm vi địa lý hạn chế**. Dữ liệu V5+ chủ yếu từ TP.HCM, Cần Thơ, cùng một phần snapshot CCTV của Vietnam Vehicles v10 phân bố trên nhiều camera ở miền Nam. Các khu vực khác như Hà Nội, Đà Nẵng, miền Trung và miền núi phía Bắc chưa được đại diện đầy đủ, nên mô hình có thể bias về domain đô thị miền Nam.

**Hạn chế 2 — Khoảng cách cross-domain còn lớn**. Đánh giá định lượng trên VN Aerial cho thấy `r_clspw` đạt 0,5676 mAP@0.5 trên test aerial, thấp hơn 22,51 điểm so với V5+ test. Bộ cross-domain mới chỉ có 635 ảnh và một kiểu góc nhìn trên cao, nên chưa đại diện đầy đủ cho mọi camera triển khai thực tế.

**Hạn chế 3 — Chưa có nhãn tracking per-frame**. Do không có nhãn tracking (IDs theo thời gian) chuẩn, không thể đánh giá tracker ByteTrack bằng các chỉ số chính thức như IDF1, MOTA, HOTA. Chất lượng tracker chỉ được đánh giá gián tiếp qua chất lượng counting.

**Hạn chế 4 — Số lớp giới hạn**. Schema 4 lớp (motorcycle, car, bus, truck) bỏ qua các loại phương tiện quan trọng khác như xe đạp, xe ba gác, xe lam, container. Mở rộng schema sẽ cần thu thập nhãn mới và không thể tái sử dụng hoàn toàn các checkpoint hiện tại.

**Hạn chế 5 — Thí nghiệm self-training thất bại**. Mô hình V6 pseudo (bổ sung 4.396 ảnh pseudo-label) không cải thiện so với V5+ (0,761 < 0,782) do khuếch đại class imbalance. Chưa có giải pháp được thực nghiệm để khắc phục.

**Hạn chế 6 — Phần cứng hạn chế**. Toàn bộ thực nghiệm chạy trên một GPU laptop 12 GB VRAM, không thử nghiệm các biến thể lớn hơn (YOLOv8m/l/x, Faster R-CNN với ResNet101/152). Kết quả có thể khác nếu có GPU mạnh hơn.

## 3. Hướng phát triển

Dựa trên các hạn chế đã nêu, nhóm đề xuất các hướng phát triển sau:

**Hướng 1 — Mở rộng dữ liệu đa vùng**. Thu thập bổ sung dữ liệu từ Hà Nội, Đà Nẵng, Hải Phòng và các tỉnh miền núi để tăng tính đại diện. Đặc biệt cần tăng số lượng mẫu cho các lớp hiếm (bus, truck) hiện chỉ chiếm 3–4% tập dữ liệu.

**Hướng 2 — Mở rộng benchmark cross-domain**. Bổ sung các bộ có nhãn từ camera ngang, camera đêm và nhiều địa phương, sau đó đánh giá bằng cùng pipeline để tách ảnh hưởng của góc nhìn, ánh sáng và vùng địa lý.

**Hướng 3 — Tối ưu phát hiện vật nhỏ sau negative result P2**. `r_p2` chỉ đạt 0,7530 mAP@0.5, thấp hơn baseline. Nghiên cứu tiếp theo nên thử tile inference, sampling theo kích thước hoặc thời gian huấn luyện dài hơn, đồng thời đo AP theo small/medium/large trước khi kết luận.

**Hướng 4 — Mở rộng class-weighted loss**. Recipe `cls_pw=0,5` đã cải thiện AP bus và truck. Có thể tiếp tục khảo sát trọng số theo từng lớp, Focal Loss hoặc re-sampling cân bằng, nhưng cần giữ test độc lập và báo cáo đồng thời ảnh hưởng đến motorcycle/car.

**Hướng 5 — Pseudo-labeling có cân bằng**. Thí nghiệm V6 pseudo thất bại do CCTV HCM chủ yếu là xe máy. Phương pháp pseudo-labeling có **cân bằng** (balanced pseudo-labeling): chỉ lấy pseudo của các lớp thiếu để bổ sung, giữ nguyên distribution V5+ gốc, có thể khắc phục vấn đề này.

**Hướng 6 — Tích hợp camera IP realtime**. Thay thế input file video bằng RTSP stream reader để kết nối trực tiếp với camera giám sát hiện có. Kết hợp với REST API backend sẽ cho phép nhiều client query kết quả đồng thời.

**Hướng 7 — Theo dõi đa camera**. Mở rộng tracking từ single-camera sang multi-camera để theo dõi một phương tiện qua nhiều góc camera khác nhau. Có thể áp dụng các kỹ thuật ReID (Re-Identification) đã có sẵn.

**Hướng 8 — Đo lưu lượng theo làn**. Hiện tại hệ thống đếm tổng theo vạch mà không phân biệt làn đường. Mở rộng cấu hình để định nghĩa từng làn riêng biệt sẽ cung cấp thông tin chi tiết hơn cho các bài toán quy hoạch giao thông.

## 4. Lời kết

Đồ án đã hoàn thành các mục tiêu chính: xây dựng bộ dữ liệu V5+ sạch, huấn luyện và so sánh YOLOv8s với Faster R-CNN, đánh giá không rò rỉ trên test và xây dựng ứng dụng demo. Mô hình cuối `r_clspw` đạt mAP@0.5 = 0,7927 trên test V5+, vượt Faster R-CNN 15,19 điểm tuyệt đối và đáp ứng mục tiêu nghiên cứu.

Bài học quan trọng nhất của đề tài không nằm ở con số mAP cuối cùng, mà ở phương pháp xây dựng dữ liệu đúng và đánh giá honest. Thay vì công bố mAP ảo 0,95 từ phiên bản V1–V4, nhóm chọn báo cáo mAP honest 0,7927 sau khi khắc phục rò rỉ dữ liệu, đồng thời công khai negative results của P2, progressive resolution và pseudo-labeling.

Các hướng phát triển đã đề xuất mở ra nhiều khả năng cải tiến tiếp theo, đặc biệt là mở rộng dữ liệu đa vùng, kiến trúc chuyên biệt cho vật nhỏ, và triển khai realtime trên camera IP. Nhóm hy vọng bộ dữ liệu, mã nguồn và kết quả thực nghiệm của đề tài sẽ là tài liệu tham khảo hữu ích cho các nghiên cứu tiếp theo về phát hiện phương tiện giao thông tại Việt Nam.
