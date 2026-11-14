

[COMMENT_GV] Về trang bìa: đổi "BÁO CÁO CUỐI KỲ" thành "BÁO CÁO ĐỒ ÁN MÔN HỌC THỊ GIÁC MÁY TÍNH VÀ ỨNG DỤNG"; ghi "Giảng viên hướng dẫn: TS. Lê Thị Vĩnh Thanh"; bổ sung ngành Khoa học máy tính, mã ngành 7480101, lớp và mã lớp học phần. Tên đề tài là "phân loại phương tiện qua ảnh bằng CNN" nhưng nội dung là phát hiện, theo vết và đếm phương tiện trên video; nhóm nên trao đổi để thống nhất, ví dụ đổi thành "Phát hiện, phân loại và đếm phương tiện giao thông Việt Nam từ video bằng CNN".
MỤC LỤC


# DANH MỤC HÌNH

Hình 2.1  Minh họa phép tích chập với kernel 3×3 tạo feature map (Nguồn: [18])	13
Hình 2.2  Khối học phần dư với kết nối tắt đồng nhất (Nguồn: [3])	14
Hình 2.3  So sánh ba nhiệm vụ phân loại ảnh, phát hiện đối tượng và phân vùng (Nguồn: AIML.com)	14
Hình 2.4  Luồng xử lý khái quát của detector dựa trên CNN	15
Hình 2.5  Kiến trúc tổng quát của YOLOv8 gồm Backbone, Neck và Head (Nguồn: [17])	16
Hình 2.6  Kiến trúc Faster R-CNN với mạng đề xuất vùng RPN (Nguồn: [1])	17
Hình 2.7  Minh họa bounding box chuẩn và bounding box dự đoán khi tính IoU	21
Hình 3.1  Pipeline xử lý video: Video đầu vào → YOLOv8s detect → ByteTrack gán track ID → Counter phát hiện giao cắt vạch → CSV sự kiện và video chú thích	25
Hình 3.2  Ví dụ ảnh từ bộ dữ liệu SOICT Night (camera đêm Hà Nội)	27
Hình 3.3  Ví dụ ảnh từ bộ dữ liệu SOICT 2024 (camera ngày TP.HCM)	27
Hình 3.4  Ví dụ ảnh từ bộ dữ liệu Cần Thơ	28
Hình 3.5  Ví dụ ảnh từ bộ dữ liệu VNv3	29
Hình 3.6  Ví dụ ảnh từ video CCTV Đà Nẵng (tập OOD)	29
Hình 3.7  Minh hoạ hệ toạ độ YOLO trên ảnh thực từ bộ SOICT 2024 (camera ngày): origin ở góc trên trái, trục +X sang phải, +Y xuống dưới; tâm bbox (cx, cy) và kích thước (w, h) đều chuẩn hoá trong khoảng [0, 1] theo kích thước ảnh gốc	31


# DANH MỤC BẢNG

Bảng 2.1  Tổng hợp công trình và ý nghĩa đối với đề tài	12
Bảng 2.2  Phân biệt ba nhiệm vụ liên quan	15
Bảng 2.3  Quy ước nhãn của đề tài	19
Bảng 3.1  Thống kê các nguồn dữ liệu	26
Bảng 3.2  Schema nhãn 4 lớp	30
Bảng 3.3  Các ràng buộc kiểm tra toàn vẹn nhãn	32
Bảng 3.4  Cấu hình chia tập V5 theo video	33
Bảng 3.5  Bước lấy mẫu (stride) theo loại video	34
Bảng 3.6  Tham số tăng cường dữ liệu mặc định	36
Bảng 3.7  Thiết lập huấn luyện bổ sung cho dữ liệu Việt Nam	37
Bảng 3.8  Cấu hình huấn luyện YOLOv8s	38
Bảng 3.9  Các chỉ số phát hiện	40

[COMMENT_GV] NHẬN XÉT TỔNG QUÁT CỦA GIẢNG VIÊN HƯỚNG DẪN (MỐC 2)
[COMMENT_GV] Cô đánh giá cao định hướng của nhóm: dùng dữ liệu giao thông Việt Nam thật (SOICT 2024 ngày, SOICT đêm, Cần Thơ, VNv3) và tách riêng video CCTV Đà Nẵng làm tập ngoài phân phối; quy đổi nhãn về một schema thống nhất và loại bỏ thay vì gộp các lớp không chắc chắn; chia tập theo video kèm kiểm tra rò rỉ tự động; lấy mẫu khung hình theo bước để giảm dư thừa; đề xuất đánh giá đếm bằng MAE, MAPE. Chương 2 có tổng quan các nhóm công trình kèm bảng tổng hợp và lý thuyết YOLOv8, Faster R-CNN đầy đủ. Tài liệu tham khảo khoảng 30 nguồn theo chuẩn IEEE: tốt.
[COMMENT_GV] Tuy nhiên, báo cáo khoảng 45 trang mới có 3 chương, chưa có Chương 4 Thực nghiệm (đoạn cuối Chương 3 nói "được triển khai trong Chương 4" nhưng chương này chưa có), chưa có chương Hệ thống và Kết luận, chưa có số liệu kết quả nào. Ngoài ra có một số điểm không thống nhất giữa các chương cần sửa (số lớp, mô hình đối chứng, tên đề tài so với nội dung), xem góp ý chi tiết.
[COMMENT_GV] Lưu ý quan trọng: số trang tăng lên phải đến từ nội dung khoa học thực sự (lý thuyết gắn với mô hình đã dùng, thực nghiệm, phân tích có số liệu, hình và bảng có thông tin), không phải từ chép lý thuyết chung chung, giãn dòng hay chèn hình trang trí. Các chương cần cân đối: cơ sở lý thuyết và tổng quan khoảng 25 đến 30%, phương pháp khoảng 20 đến 25%, thực nghiệm và thảo luận khoảng 30 đến 35% tổng số trang.
[COMMENT_GV] Kế hoạch trang đề nghị (tổng khoảng 70 trang)
Phần đầu khoảng 7 trang (bổ sung Lời cảm ơn, Nhận xét giảng viên, Tóm tắt, Danh mục từ viết tắt). Chương 1 khoảng 5 trang (giữ). Chương 2 khoảng 13 trang (giữ). Chương 3 khoảng 15 trang (rút gọn từ 19 trang: mục 3.8 môi trường chuyển sang Chương 4). Chương 4. Thực nghiệm và đánh giá khoảng 18 trang (chưa có). Chương 5. Hệ thống đếm phương tiện khoảng 5 trang (chưa có). Kết luận khoảng 3 trang. Tài liệu tham khảo khoảng 3 trang. Phụ lục khoảng 3 trang.
[COMMENT_GV] VỀ ĐỊNH DẠNG BÁO CÁO
[COMMENT_GV] Kết quả cô kiểm tra định dạng bản hiện tại: khổ A4, lề trái 3 cm, phải 2 cm: đúng; lề trên, dưới 2,2 cm, chỉnh về 2 cm; thân bài đã canh đều, cỡ 13; giãn dòng đang 1,2, cần chỉnh về 1,5; khoảng 31 đoạn cỡ 11 (chủ yếu ghi chú, nội dung bảng), đoạn văn thân bài phải cỡ 13; font chưa gán rõ ở kiểu Normal, cần đặt Times New Roman; mục lục tự động đã có; danh mục hình và bảng hiện dùng kiểu chú thích riêng, chưa phải Insert Caption, cần chèn lại bằng Insert Caption để danh mục tự cập nhật; tài liệu tham khảo thiếu khoảng trắng sau số thứ tự ("[1]S. Ren"), cần dùng tab hoặc khoảng trắng.
[COMMENT_GV] Yêu cầu định dạng thống nhất cho toàn bộ báo cáo, các em chỉnh trong Word bằng cách sửa kiểu (Modify Style) chứ không định dạng thủ công từng đoạn:
Khổ giấy A4 (21 × 29,7 cm); lề trái 3 cm, lề phải 2 cm, lề trên 2 cm, lề dưới 2 cm. Kiểu Normal: Times New Roman, cỡ 13, canh đều hai bên (Justify), giãn dòng 1,5 (Line spacing 1.5 lines), khoảng cách trước 0 pt và sau 6 pt, thụt đầu dòng 1 cm. Heading 1 (tên chương): cỡ 14, in đậm, in hoa, canh giữa, bắt đầu ở trang mới. Heading 2: cỡ 13, in đậm. Heading 3: cỡ 13, đậm nghiêng. Chú thích bảng, hình: cỡ 12, canh giữa, dùng References, Insert Caption có số chương (Bảng 2.1, Hình 3.4); chú thích bảng đặt trên bảng, chú thích hình đặt dưới hình. Nội dung trong bảng: cỡ 12, giãn dòng 1,0. Số trang ở giữa chân trang; phần đầu đánh số La Mã (i, ii, iii), từ chương đầu tiên đánh số 1, 2, 3 (chèn Section Break, Next Page và đặt lại Start at 1).
Mục lục, danh mục bảng, danh mục hình đều phải tạo tự động (References, Table of Contents và References, Insert Table of Figures) và cập nhật lại bằng Ctrl + A rồi F9 trước khi nộp. Công thức soạn bằng Equation và đánh số (2.1), (2.2) canh phải. Tài liệu tham khảo trình bày thống nhất theo chuẩn IEEE, đánh số theo thứ tự trích dẫn.
[COMMENT_GV] Đánh giá khả năng hoàn thiện
[COMMENT_GV] Phần dữ liệu và phương pháp cô ước hoàn thành khoảng 80%; tuy nhiên chưa có kết quả nào nên mức hoàn thành chung khoảng 45%. Nhóm có máy GPU RTX 3500 và pipeline Ultralytics, nên huấn luyện YOLOv8s và Faster R-CNN trên khoảng 10.000 khung hình là khả thi trong 1 tuần. Mốc đề nghị: (1) đến 08/11/2026: hoàn thành huấn luyện YOLOv8s và Faster R-CNN, đánh giá trên test và OOD; (2) đến 15/11/2026: ablation (dữ liệu, augmentation, imgsz), đếm phương tiện có ground truth; (3) đến 23/11/2026: viết Chương 4, 5 và Kết luận; (4) hạn hoàn thiện chậm nhất là ngày 30/11/2026.
Quy ước màu: chữ màu xanh là phần cô viết lại mẫu hoặc dàn ý gợi ý để các em tham khảo; chữ màu đỏ là góp ý chỉnh sửa, hướng dẫn và các việc cần làm tiếp. Các em xóa toàn bộ các khung nhận xét này trước khi nộp bản cuối.

# CHƯƠNG 1  TỔNG QUAN ĐỀ TÀI


## 1.1 Bối cảnh và lý do chọn đề tài

Ảnh và video giao thông cung cấp thông tin về số lượng, vị trí và cơ cấu phương tiện tại một khu vực quan sát. Để khai thác nguồn dữ liệu này, hệ thống cần chuyển nội dung ảnh thành các đối tượng có nhãn thay vì chỉ lưu trữ hoặc phát lại video. Việc xác định các nhóm xe phổ biến có thể hỗ trợ tổng hợp cơ cấu phương tiện và chuẩn bị dữ liệu cho các nghiên cứu giao thông đô thị.
Phân loại toàn bộ một ảnh thành một nhãn bằng mạng CNN truyền thống chưa phù hợp với cảnh giao thông chứa nhiều xe. Trong cùng một khung hình có thể xuất hiện nhiều loại phương tiện khác nhau (xe máy, ô tô con, xe buýt, xe tải và xe đạp) ở nhiều vị trí khác nhau. Do đó, đề tài được xác định là bài toán phát hiện đối tượng đa lớp: hệ thống phải xác định được vị trí (bounding box) và nhãn lớp cho từng phương tiện. Một bộ phát hiện đối tượng dựa trên CNN có thể thực hiện đồng thời hai yêu cầu này [1], [2].
Dữ liệu thực tế đặt ra những khó khăn như phương tiện ở xa có kích thước nhỏ, các xe che khuất nhau, khác biệt góc nhìn và thay đổi ánh sáng. Ngoài ra, số mẫu giữa các loại xe có thể chênh lệch, khiến kết quả trung bình không phản ánh đầy đủ chất lượng của các lớp ít mẫu. Những yếu tố này đòi hỏi nghiên cứu quan tâm cả chất lượng dữ liệu, cách chia tập và kết quả theo từng lớp.
Nhóm lựa chọn YOLOv8 làm mô hình chính và áp dụng học chuyển giao từ trọng số đã huấn luyện sẵn. Faster R-CNN được dùng làm mô hình đối chứng để có cơ sở so sánh. Hai mô hình cùng giải quyết nhiệm vụ phát hiện và phân loại trên một bộ dữ liệu năm lớp; kết quả so sánh sẽ được diễn giải trong điều kiện phần cứng và cách đánh giá được công bố rõ ràng. Việc lựa chọn mô hình không đồng nghĩa với việc đã đạt trước mục tiêu độ chính xác.

## 1.2 Vấn đề cần giải quyết

Vấn đề thứ nhất là xây dựng một tập dữ liệu có nhãn nhất quán. Dữ liệu từ nhiều nguồn có thể sử dụng các tên lớp và quy tắc gán nhãn khác nhau. Do đó, cần xác định tiêu chí phân biệt từng loại phương tiện, kiểm tra bounding box và xử lý các trường hợp không đủ thông tin để gán nhãn. Bên cạnh đó, ảnh giao thông Việt Nam được thu thập bổ sung để dữ liệu sát với bối cảnh ứng dụng thực tế.
Vấn đề thứ hai là lựa chọn và tinh chỉnh mô hình để nhận dạng nhiều phương tiện trong một ảnh. Hệ thống cần hạn chế cả dự đoán sai và bỏ sót, đặc biệt đối với xe đạp, xe máy nhỏ hoặc xe bị che khuất. Thay vì khẳng định một kiến trúc luôn tốt hơn, đề tài cần thực nghiệm trên cùng dữ liệu để xác định sự khác biệt giữa YOLOv8 và Faster R-CNN.
Vấn đề thứ ba là đánh giá đúng các thành phần của hệ thống. Điểm tin cậy của một dự đoán, Accuracy phân loại, mAP phát hiện và sai số thống kê số lượng là các đại lượng khác nhau. Báo cáo phải nêu rõ đối tượng được đánh giá, quy tắc ghép bounding box, ngưỡng sử dụng và số mẫu của từng lớp. Tốc độ xử lý cũng cần gắn với cấu hình phần cứng, kích thước ảnh và phạm vi thời gian được đo.

## 1.3 Mục tiêu nghiên cứu


### 1.3.1 Mục tiêu tổng quát

Xây dựng và đánh giá hệ thống phát hiện, phân loại phương tiện giao thông từ ảnh hoặc video bằng mô hình CNN, sử dụng YOLOv8 tinh chỉnh làm phương pháp chính và Faster R-CNN làm đối chứng. Hệ thống hướng tới việc nhận dạng năm loại xe phổ biến và cung cấp kết quả trực quan phục vụ thống kê cơ cấu phương tiện.

### 1.3.2 Mục tiêu cụ thể

• Xây dựng bộ dữ liệu kết hợp nguồn công khai với ảnh giao thông Việt Nam do nhóm thu thập; gán nhãn năm lớp gồm xe máy, ô tô con, xe buýt, xe tải và xe đạp.
• Chuẩn hóa dữ liệu và chia train, validation, test theo nguồn video hoặc camera; kiểm soát ảnh trùng và chỉ áp dụng tăng cường dữ liệu trong tập huấn luyện.
• Tinh chỉnh YOLOv8 từ trọng số pretrained và triển khai Faster R-CNN trên cùng danh sách ảnh của từng tập dữ liệu.
• Đánh giá mAP@0.5, Precision, Recall và F1 theo lớp; xây dựng phép đo Accuracy phân loại với điều kiện ghép bounding box được định nghĩa rõ, hướng tới mục tiêu Accuracy ≥ 0,90 trên tập test.
• Đo FPS và độ trễ trong điều kiện thống nhất; xây dựng demo nhận ảnh hoặc video, hiển thị bounding box, nhãn lớp, điểm tin cậy và số phương tiện theo loại.
• Đóng gói mã nguồn, trọng số, hướng dẫn sử dụng và báo cáo; phân tích các trường hợp thất bại thay vì chỉ trình bày ví dụ thuận lợi.
Ngưỡng Accuracy ≥ 0,90 là mục tiêu cần kiểm chứng bằng thực nghiệm. Tỉ lệ đối tượng được định vị thành công cũng được báo cáo kèm theo, để Accuracy cao trên các cặp bounding box đã ghép không che lấp hiện tượng bỏ sót. Định nghĩa phép đo được trình bày ở mục 2.8.3.

## 1.4 Câu hỏi nghiên cứu

Câu hỏi 1: Trên cùng tập test năm lớp, YOLOv8 và Faster R-CNN khác nhau như thế nào về mAP@0.5, khả năng nhận dạng từng lớp và tốc độ xử lý?
Câu hỏi 2: Khi giữ nguyên kiến trúc và cách chia dữ liệu, việc bổ sung ảnh Việt Nam hoặc thay đổi chính sách tăng cường dữ liệu ảnh hưởng như thế nào đến kết quả?
Câu hỏi 3: Những lỗi nào xuất hiện đối với phương tiện nhỏ, bị che khuất hoặc trong điều kiện ánh sáng khó; lỗi thuộc định vị, nhầm loại hay bỏ sót?
Câu hỏi 4: Hệ thống có đạt mục tiêu Accuracy phân loại đã xác định trên test hay không, và mức độ bao phủ đối tượng thật đi kèm kết quả đó là bao nhiêu?
[COMMENT_GV] Câu hỏi 1 so sánh YOLOv8 và Faster R-CNN "trên cùng tập test năm lớp", trong khi Chương 3 dùng schema 4 lớp và chỉ cấu hình YOLOv8s. Các em thống nhất toàn báo cáo: số lớp (4 hay 5) và có hay không huấn luyện Faster R-CNN. Nếu giữ Câu hỏi 1, Chương 3 cần bổ sung mục cấu hình Faster R-CNN ResNet50-FPN (torchvision hoặc Detectron2) trên cùng dữ liệu và độ phân giải. Nên bổ sung Câu hỏi 5 về độ chính xác đếm phương tiện, vì đây là sản phẩm chính của hệ thống.

## 1.5 Đối tượng nghiên cứu

Đối tượng lý thuyết gồm mạng nơ-ron tích chập, học chuyển giao, phát hiện đối tượng một giai đoạn và hai giai đoạn, cùng các tiêu chí đánh giá định vị và phân loại. YOLOv8 và Faster R-CNN là hai kiến trúc được khảo sát để thiết lập thực nghiệm, không phải hai tầng bắt buộc chạy nối tiếp trong một pipeline.
Đối tượng thực nghiệm là ảnh và khung hình video chứa phương tiện giao thông đường bộ thuộc năm lớp đã xác định. Đơn vị gán nhãn là từng phương tiện với một bounding box và một nhãn lớp. Kết quả quan sát gồm dự đoán của mô hình, nhãn chuẩn và thời gian xử lý.

## 1.6 Phạm vi nghiên cứu

Về đầu vào, đề tài xử lý ảnh màu hai chiều và video giao thông đã lưu. Phạm vi demo ưu tiên cảnh từ camera quan sát tương đối ổn định. Nhóm không đặt mục tiêu suy ra vận tốc, đọc biển số, nhận diện người lái, điều khiển đèn tín hiệu hoặc nhận dạng phương tiện ưu tiên.
Về đầu ra, mô hình tạo bounding box, nhãn và điểm tin cậy cho từng phương tiện. Số lượng theo lớp trong một ảnh hoặc một khung hình được tính từ các dự đoán còn lại sau hậu xử lý. Nếu mở rộng sang đếm lượt xe đi qua một vị trí trong video, cần bổ sung theo dõi định danh và quy tắc đếm; cộng trực tiếp số xe qua các khung hình sẽ tạo trùng lặp và không được xem là lưu lượng thực tế.
Về mô hình, YOLOv8 được lựa chọn làm detector chính; Faster R-CNN với backbone ResNet50-FPN là lựa chọn triển khai dự kiến cho đối chứng. Nghiên cứu tập trung tinh chỉnh các kiến trúc có sẵn với schema và dữ liệu được chuẩn hóa.
Về tài nguyên, Python, PyTorch, Ultralytics và OpenCV được sử dụng làm nền tảng. Google Colab và máy cá nhân có GPU là môi trường dự kiến; cấu hình thực tế sẽ được ghi nhận khi triển khai.

## 1.7 Phương pháp nghiên cứu

Nhóm sử dụng phương pháp nghiên cứu tài liệu để phân nhóm công trình về phân loại ảnh, phát hiện đối tượng và bộ dữ liệu giao thông. Mỗi công trình được xem xét theo nhiệm vụ, ý tưởng, bằng chứng thực nghiệm và giới hạn khi áp dụng cho bài toán năm lớp. Kết quả khảo sát là cơ sở lựa chọn hai mô hình và xác định nội dung cần kiểm chứng tại bối cảnh Việt Nam.
Phương pháp thực nghiệm được tổ chức theo trình tự: thống nhất quy tắc gán nhãn; thu thập và kiểm tra dữ liệu; cố định các tập dữ liệu; tinh chỉnh hai mô hình; chọn cấu hình bằng validation; đánh giá cấu hình đã chọn trên test; phân tích lỗi và xây dựng demo. Mọi thay đổi siêu tham số phải dựa trên train hoặc validation, không dựa trên kết quả test.
Đối với so sánh, nhóm giữ thống nhất tập test, schema năm lớp, quy tắc đánh giá và phần cứng đo tốc độ. Những khác biệt bắt buộc về resize, batch hoặc cấu hình đặc thù của từng thư viện được ghi rõ. Các thí nghiệm về tăng cường dữ liệu được tách khỏi thí nghiệm so sánh kiến trúc để hạn chế kết luận vượt quá bằng chứng.

## 1.8 Sản phẩm dự kiến

[COMMENT_GV] Mục 1.8 "Sản phẩm dự kiến", 1.9 "đóng góp dự kiến" và mục 2.9 "công cụ dự kiến" cần viết lại ở thì đã thực hiện trong báo cáo cuối.
• Bộ dữ liệu có nhãn năm lớp, danh sách train/validation/test, mô tả nguồn và hướng dẫn sử dụng phù hợp với điều kiện của từng nguồn dữ liệu.
• Mã nguồn huấn luyện và đánh giá YOLOv8, Faster R-CNN; tệp cấu hình và trọng số được chọn theo validation.
• Demo ảnh và video có kết quả trực quan; thống kê số đối tượng theo lớp trong ảnh hoặc khung hình.
• Bảng kết quả theo lớp, confusion matrix, biểu đồ Precision–Recall, log huấn luyện, kết quả đo tốc độ và tập ví dụ phân tích lỗi.
• README tái lập thí nghiệm, báo cáo cuối kỳ và slide thuyết trình.

## 1.9 Ý nghĩa và đóng góp dự kiến

Về học thuật, đề tài xây dựng một quy trình thực nghiệm có thể kiểm tra từ dữ liệu đến đánh giá cho bài toán phát hiện và phân loại năm loại phương tiện. Phần đối chứng giúp làm rõ sự đánh đổi giữa chất lượng dự đoán và chi phí xử lý trong điều kiện khảo sát. Đóng góp dự kiến nằm ở việc tổ chức dữ liệu, tích hợp hệ thống và đánh giá có kiểm soát; đề tài không tuyên bố đề xuất một kiến trúc mạng mới.
Về thực tiễn, sản phẩm có thể hỗ trợ quan sát cơ cấu loại xe trong phạm vi dữ liệu và cảnh quay được kiểm thử. Khả năng áp dụng cho camera, thời điểm hoặc điều kiện khác cần được xác nhận bằng dữ liệu mới.

## 1.10 Cấu trúc báo cáo

Báo cáo được tổ chức thành bảy chương. Chương 1 giới thiệu bài toán, mục tiêu và phạm vi. Chương 2 khảo sát công trình liên quan và trình bày cơ sở lý thuyết. Chương 3 phân tích yêu cầu và thiết kế phương pháp. Chương 4 mô tả dữ liệu, cài đặt và huấn luyện. Chương 5 trình bày thực nghiệm và đánh giá. Chương 6 giới thiệu demo và kiểm thử hệ thống. Chương 7 kết luận, nêu hạn chế và hướng phát triển.


# CHƯƠNG 2  CÔNG TRÌNH LIÊN QUAN VÀ CƠ SỞ LÝ THUYẾT

Chương này khảo sát các hướng tiếp cận liên quan, sau đó trình bày những khái niệm cần dùng để xây dựng và đánh giá hệ thống. Nội dung khảo sát được phân tích theo mức độ phù hợp với bài toán nhiều phương tiện trong một ảnh. Các lựa chọn kỹ thuật của đề tài được xem là phương án cần kiểm chứng ở các chương thực nghiệm.

## 2.1 Tổng quan các công trình nghiên cứu liên quan


### 2.1.1 Nhóm công trình phân loại ảnh và nhận dạng phương tiện

CNN học đặc trưng từ ảnh thay cho việc chỉ sử dụng bộ mô tả được thiết kế thủ công. Trong nhóm kiến trúc sâu, He và cộng sự [3] đề xuất học phần dư với kết nối tắt để hỗ trợ tối ưu mạng nhiều lớp. ResNet được đánh giá trên các bài toán nhận dạng và có thể đóng vai trò backbone cho detector. Tuy nhiên, một backbone hoặc một mô hình phân loại ảnh độc lập chưa tự cung cấp danh sách bounding box cho mọi phương tiện trong cảnh.
Yang và cộng sự [4] giới thiệu về CompCars cho các nhiệm vụ như phân loại dòng ô tô, xác minh và dự đoán thuộc tính. Công trình cho thấy vai trò của dữ liệu có cấu trúc và sự khác biệt giữa ảnh trên web với ảnh giám sát. Đối với đề tài này, CompCars là tài liệu tham khảo về nhận dạng phương tiện; nhiệm vụ phân biệt các dòng ô tô không trùng với nhiệm vụ phát hiện năm nhóm xe trong cùng một ảnh. Vì vậy, kết quả của hai nhiệm vụ không thể so sánh trực tiếp bằng một con số Accuracy.

### 2.1.2 Nhóm mô hình phát hiện hai giai đoạn

Ren và cộng sự [1] đề xuất Faster R-CNN với Region Proposal Network dùng chung đặc trưng tích chập với mạng phát hiện. Quy trình tạo vùng ứng viên rồi phân loại và hiệu chỉnh bounding box giúp đưa khâu đề xuất vùng vào mô hình học. Đây là cơ sở lựa chọn Faster R-CNN làm đối chứng cho đề tài.
Công trình đã kiểm chứng kiến trúc trên các benchmark của bài báo, nhưng kết quả đó không bảo đảm hiệu năng trên dữ liệu Việt Nam hoặc phần cứng khác. Chi phí xử lý phụ thuộc backbone, số vùng ứng viên, kích thước đầu vào và cách triển khai. Do đó, nhận định về tốc độ và chất lượng của đối chứng phải được xác nhận trong thí nghiệm hiện tại, thay vì mặc định detector hai giai đoạn luôn chính xác hơn.

### 2.1.3 Nhóm mô hình phát hiện một giai đoạn

Bài báo của Redmon và cộng sự [2] xây dựng YOLO theo hướng dự đoán bounding box và xác suất lớp từ toàn ảnh trong một mạng thống nhất. Công trình đặt trọng tâm vào tốc độ và khả năng tối ưu toàn bộ quy trình. Liu và cộng sự [5] đề xuất SSD, sử dụng các default bounding box và đặc trưng nhiều độ phân giải để dự đoán đối tượng ở các kích thước khác nhau. Hai công trình cung cấp nền tảng cho hướng phát hiện một giai đoạn.
YOLOv8 là phiên bản thuộc hệ sinh thái Ultralytics, được cung cấp với nhiều kích thước mô hình và trọng số cho từng nhiệm vụ [6]. Đề tài sử dụng bản detection để tạo bounding box và nhãn cho từng xe. Những đặc điểm như head tách nhánh và thiết kế anchor-free là cơ sở kỹ thuật để khảo sát; chúng không tự chứng minh YOLOv8 sẽ vượt Faster R-CNN trong mọi điều kiện.

### 2.1.4 Nhóm bộ dữ liệu và đánh giá giao thông

Tờ báo của Wen và cộng sự [7] xây dựng UA-DETRAC cho phát hiện và theo dõi nhiều đối tượng trong video giao thông. Giá trị của benchmark là đặt việc đánh giá trong các cảnh và điều kiện quan sát cụ thể. Khi cân nhắc sử dụng dữ liệu này, cần kiểm tra schema nhãn và mức độ bao phủ năm lớp của đề tài; không được tự động xem mọi tập dữ liệu phương tiện là tương đương nhau.
Yu và cộng sự [8] giới thiệu BDD100K với dữ liệu lái xe đa dạng và nhiều nhiệm vụ thị giác. Công trình nhấn mạnh sự khác nhau về môi trường, thời tiết và nhiệm vụ. Nhóm rút ra yêu cầu đánh giá theo nguồn và điều kiện quan sát. Dữ liệu từ camera trên xe cũng không thể được xem là thay thế hoàn toàn cho ảnh camera giao thông cố định.
COCO [9] là bộ dữ liệu quan trọng đối với các trọng số pretrained được khảo sát. Vai trò của COCO trong đề tài chủ yếu là nguồn tiền huấn luyện và tài liệu tham chiếu về đánh giá. Một bộ dữ liệu tổng quát không loại bỏ nhu cầu bổ sung ảnh Việt Nam hoặc kiểm tra chất lượng các lớp ít gặp trong dữ liệu đích.

### 2.1.5 Tổng hợp và đánh giá các hướng tiếp cận

Bảng 2.1  Tổng hợp công trình và ý nghĩa đối với đề tài

Tổng hợp trên cho thấy ba yêu cầu đối với nghiên cứu: lựa chọn nhiệm vụ phù hợp, xây dựng dữ liệu đúng năm lớp và đánh giá trên một quy trình thống nhất. Khoảng trống mà đề tài hướng tới là chưa có bằng chứng so sánh hai detector này trên tập dữ liệu giao thông Việt Nam được chuẩn hóa theo năm lớp. Đây là nhu cầu kiểm chứng trong phạm vi đề tài; nhóm không khẳng định chưa từng có nghiên cứu nào giải quyết bài toán tương tự.

## 2.2 Cơ sở lý thuyết về mạng nơ-ron tích chập


### 2.2.1 Minh họa phép tích chập với kernel 3×3

Ảnh màu có thể biểu diễn dưới dạng tensor gồm chiều cao, chiều rộng và các kênh màu. Lớp tích chập áp dụng bộ lọc tại nhiều vị trí để tạo bản đồ đặc trưng. Việc sử dụng chung bộ lọc giúp mô hình học một mẫu cục bộ có thể xuất hiện ở những vị trí khác nhau. Các tầng đầu thường phản ứng với cạnh hoặc kết cấu; nhiều tầng kết hợp tạo ra biểu diễn phức tạp hơn.
Trong (2.1), k là kích thước kernel theo chiều đang xét, p là padding, s là stride và d là dilation. Công thức tương tự được áp dụng cho chiều rộng. Thay đổi stride hoặc padding ảnh hưởng độ phân giải đặc trưng và khả năng giữ thông tin của phương tiện nhỏ. Tính chất dịch chuyển của tích chập không có nghĩa toàn bộ CNN luôn bất biến tuyệt đối với mọi phép tịnh tiến.

Hình 2.1  Minh họa phép tích chập với kernel 3×3 tạo feature map (Nguồn: [18])

### 2.2.2 Hàm kích hoạt và giảm kích thước đặc trưng

Hàm kích hoạt đưa tính phi tuyến vào mạng. ReLU đặt giá trị âm về 0; SiLU nhân đầu vào với giá trị sigmoid của chính nó, tức SiLU(x) = x·σ(x). Việc giảm kích thước có thể thực hiện bằng pooling hoặc tích chập có stride lớn hơn 1. Cách tổ chức cụ thể phụ thuộc kiến trúc; không phải mọi CNN đều bắt buộc chứa chuỗi convolution, max pooling và fully connected theo cùng một mẫu.
Giảm độ phân giải giúp tiết kiệm tính toán nhưng có thể làm mất tín hiệu của đối tượng nhỏ. Vì vậy, detector thường khai thác đặc trưng nhiều mức. Với xe máy hoặc xe đạp ở xa, việc giữ lại đủ thông tin không gian và lựa chọn kích thước ảnh đầu vào là các yếu tố cần được khảo sát cùng chi phí xử lý.

### 2.2.3 Học phần dư và backbone ResNet

y = F(x) + x	(2.2)
Biểu thức (2.2) mô tả trường hợp kết nối tắt đồng nhất trong một khối residual [3]. Khi kích thước không tương thích, nhánh tắt cần phép chiếu hoặc điều chỉnh phù hợp. Trong phương án Faster R-CNN của đề tài, ResNet50 đóng vai trò trích xuất đặc trưng; FPN kết hợp các mức đặc trưng. Đây là backbone của detector, khác với việc huấn luyện một ResNet50 phân loại ảnh độc lập.

Hình 2.2  Khối học phần dư với kết nối tắt đồng nhất (Nguồn: [3])

## 2.3 Bài toán phát hiện và phân loại phương tiện

Với ảnh I, detector tạo tập dự đoán D gồm các phần tử (b, c, s), trong đó b là bounding box, c là nhãn lớp và s là điểm tin cậy. Nhãn chuẩn G chứa bounding box và lớp do người gán nhãn xác định. Trong cùng một ảnh, số phần tử của D có thể khác số phần tử của G vì mô hình có thể bỏ sót hoặc tạo dự đoán dư.

Hình 2.3  So sánh ba nhiệm vụ phân loại ảnh, phát hiện đối tượng và phân vùng (Nguồn: AIML.com)
Bảng 2.2  Phân biệt ba nhiệm vụ liên quan

Đề tài tập trung vào hàng thứ hai của Bảng 2.2. Hình 2.4 mô tả luồng suy luận khái quát của detector dựa trên CNN. Việc thống kê số đối tượng trong một khung hình được thực hiện sau hậu xử lý; nếu mở rộng thành đếm lượt theo thời gian thì cần thành phần bổ sung.

Hình 2.4  Luồng xử lý khái quát của detector dựa trên CNN

## 2.4 Kiến trúc và cơ chế hoạt động của YOLOv8


### 2.4.1 Backbone và tổng hợp đặc trưng đa tỉ lệ

Cấu hình YOLOv8 detection của Ultralytics sử dụng các tầng Conv, khối C2f và SPPF để tạo biểu diễn ảnh, đồng thời có các phép upsample và nối đặc trưng trong phần tổng hợp đa tỉ lệ [10]. Các mức đầu ra có độ phân giải khác nhau hỗ trợ dự đoán xe ở những kích thước khác nhau. Việc mô tả này phải gắn với cấu hình và phiên bản thư viện được sử dụng khi triển khai.
Trong đề tài, có thể bắt đầu khảo sát từ biến thể nhỏ để kiểm tra toàn bộ quy trình trước khi tăng kích thước mô hình. Việc chọn biến thể cuối cùng dựa vào validation và ngân sách tính toán. Tăng số tham số không được xem là bảo đảm tăng chất lượng đối với mọi lớp hoặc mọi cảnh.

Hình 2.5  Kiến trúc tổng quát của YOLOv8 gồm Backbone, Neck và Head (Nguồn: [17])

### 2.4.2 Head dự đoán và cơ chế anchor-free

YOLOv8 dùng head tách phần dự đoán lớp và phần hồi quy bounding box, với thiết kế anchor-free [6]. Mô hình vẫn dự đoán trên các vị trí của bản đồ đặc trưng, nhưng không sử dụng bộ kích thước anchor box định trước theo cách của các detector anchor-based. Cơ chế này có những khác biệt so với mô tả ô lưới của YOLO phiên bản đầu.
Đầu ra detection cho biết vị trí, lớp và điểm tin cậy của mỗi dự đoán [11]. Điểm 0,95 trên một bounding box là điểm do mô hình gán cho dự đoán đó; nó không phải bằng chứng mô hình đạt Accuracy 95% trên tập dữ liệu. Nhãn lớp của đề tài được cấu hình lại cho năm loại xe, trong khi các trọng số tương thích được kế thừa để fine-tune.

### 2.4.3 Hàm mất mát và hậu xử lý

L = λbox Lbox + λcls Lcls + λdfl Ldfl	(2.3)
Theo phần triển khai loss của detector YOLOv8 trong Ultralytics [12], các thành phần chính gồm box loss, classification loss và Distribution Focal Loss. Box loss sử dụng thông tin IoU, trong đó có CIoU ở cách triển khai được khảo sát; classification loss sử dụng BCE. YOLOv8 detection không sử dụng một nhánh objectness loss độc lập giống một số phiên bản YOLO trước.
Sau dự đoán, hệ thống lọc theo điểm tin cậy và áp dụng Non-Maximum Suppression để giảm các bounding box trùng. Ngưỡng IoU của NMS phục vụ hậu xử lý, khác với ngưỡng IoU dùng để đánh giá bounding box có đúng hay không. Các ngưỡng suy luận phải được lựa chọn trên validation và công bố khi đánh giá test.

## 2.5 Faster R-CNN làm mô hình đối chứng


### 2.5.1 Nguyên lý hai giai đoạn

Faster R-CNN dùng đặc trưng từ backbone để tạo vùng ứng viên bằng RPN; các vùng sau đó được đưa tới bộ phận phân loại và hồi quy bounding box [1]. Đây là khác biệt về tổ chức xử lý so với detector dự đoán trực tiếp tại nhiều vị trí của đặc trưng. Kết quả cuối vẫn là danh sách bounding box và nhãn, nên hai mô hình có thể được đánh giá trên cùng nhiệm vụ.

Hình 2.6  Kiến trúc Faster R-CNN với mạng đề xuất vùng RPN (Nguồn: [1])

### 2.5.2 Lựa chọn ResNet50-FPN và cấu trúc nhãn

Torchvision cung cấp Faster R-CNN ResNet50-FPN với trọng số COCO và sử dụng MultiScaleRoIAlign trong phần xử lý vùng [13]. Vì vậy, khi mô tả phiên bản này, thuật ngữ RoI Align phù hợp hơn việc khẳng định mọi triển khai đều dùng RoI Pooling. Lớp dự đoán được thay để phù hợp với các lớp của bài toán theo hướng dẫn fine-tune [14].
Trong quy ước triển khai dự kiến, YOLO dùng ID từ 0 đến 4 cho năm loại xe; Faster R-CNN của Torchvision dùng nhãn 1 đến 5 và dành 0 cho background. Background không phải lớp phương tiện thứ sáu. Bộ chuyển đổi dữ liệu cần giữ nguyên bounding box và ý nghĩa nhãn khi thay đổi cách đánh số.

### 2.5.3 Nguyên tắc so sánh công bằng

Cả hai mô hình phải được đánh giá trên cùng ảnh test và cùng bộ nhãn chuẩn. Thông số resize, chính sách augmentation, tiêu chí chọn checkpoint và ngân sách huấn luyện cần được ghi nhận. Không nhất thiết ép mọi siêu tham số giống nhau nếu điều đó không phù hợp với thư viện, nhưng phải tránh so sánh hai kết quả có điều kiện khác nhau mà không giải thích.
Đối chứng này trả lời câu hỏi khác biệt giữa hai hệ thống được fine-tune. Nếu muốn kết luận riêng về lợi ích của transfer learning, cần thêm thí nghiệm cùng một kiến trúc được khởi tạo theo hai cách. Đề tài hiện lựa chọn đối chứng theo kiến trúc và không sử dụng Faster R-CNN làm bằng chứng duy nhất cho mọi lợi ích của học chuyển giao.

## 2.6 Học chuyển giao và tăng cường dữ liệu


### 2.6.1 Học chuyển giao và tinh chỉnh

Học chuyển giao sử dụng trọng số đã được học ở nhiệm vụ hoặc dữ liệu nguồn để khởi tạo mô hình cho dữ liệu đích. Fine-tune tiếp tục tối ưu một phần hoặc toàn bộ tham số bằng nhãn của bài toán mới. Với detector có sẵn, phần dự đoán lớp được điều chỉnh cho schema năm lớp; việc đóng băng tầng và lịch mở khóa là các lựa chọn cần ghi trong cấu hình thực nghiệm [14].
Quy trình dự kiến theo dõi chất lượng validation để chọn checkpoint và dừng khi không còn cải thiện theo tiêu chí đã đặt. Nhóm cần lưu phiên bản phần mềm, seed, cấu hình dữ liệu và siêu tham số. Mức độ tái lập của GPU cũng cần được cân nhắc khi diễn giải những chênh lệch nhỏ giữa các lần chạy.

### 2.6.2 Tăng cường dữ liệu có bảo toàn nhãn

Những phép biến đổi được cân nhắc gồm lật ngang, thay đổi tỉ lệ và điều chỉnh màu sắc trong phạm vi hợp lý. Với detection, phép biến đổi hình học phải được áp dụng đồng thời lên ảnh và bounding box. Các bounding box ra ngoài vùng ảnh hoặc còn diện tích quá nhỏ cần được xử lý bằng quy tắc thống nhất.
YOLOv4 [15] là một tài liệu tham khảo cho Mosaic và các kỹ thuật tăng cường trong huấn luyện detector. Mosaic ghép nhiều ảnh để thay đổi ngữ cảnh và tỉ lệ đối tượng, nhưng mức tác động đối với bộ dữ liệu hiện tại phải được đánh giá riêng. Các phép biến đổi quá mạnh có thể tạo mẫu khác xa điều kiện cần nhận dạng.
Augmentation thông thường không tự cân bằng số lượng đối tượng giữa các lớp. Nhóm cần thống kê cả số ảnh và số instance, sau đó cân nhắc thu thập bổ sung hoặc lấy mẫu có chủ đích cho lớp thiếu. Ảnh validation và test không được bổ sung các bản biến đổi từ ảnh train để làm tăng kết quả đánh giá.

## 2.7 Cơ sở xây dựng và quản lý bộ dữ liệu


### 2.7.1 Nguồn dữ liệu và schema năm lớp

[COMMENT_GV] Mục 2.7.1 ghi schema năm lớp, mục 3.2.2 ghi schema 4 lớp. Mục 2.7 (nguồn dữ liệu, định dạng nhãn, chia tập) cũng trùng với mục 3.2, 3.3. Đề nghị bỏ mục 2.7 ở Chương 2 (chỉ giữ lý thuyết về rò rỉ dữ liệu khi chia theo khung hình nếu cần) và để toàn bộ thiết kế dữ liệu ở Chương 3.
Dữ liệu dự kiến kết hợp nguồn công khai với ảnh Việt Nam tự thu thập. Kaggle, Roboflow hoặc Hugging Face là nền tảng phân phối; khi chọn dữ liệu cần ghi tên bộ dữ liệu cụ thể, tác giả, phiên bản, đường dẫn nguồn, điều kiện sử dụng và nội dung nhãn. Không mặc định một liên kết chứa ảnh phân loại đã có bounding box phù hợp cho detection.
Bảng 2.3  Quy ước nhãn của đề tài

Bảng 2.3 là quy ước tổ chức dữ liệu của đề tài. Trước khi gán nhãn hàng loạt, nhóm phải thống nhất các trường hợp biên bằng hướng dẫn và ví dụ. Không tự động gộp van vào truck chỉ để khớp số lớp; nếu nhãn nguồn không đủ rõ, cần kiểm tra lại hoặc loại trường hợp không thể ánh xạ tin cậy.

### 2.7.2 Định dạng bounding box và kiểm tra nhãn

Định dạng YOLO lưu một dòng cho mỗi đối tượng gồm class_id, x_c, y_c, w và h đã chuẩn hóa. W và H là kích thước ảnh gốc; bounding box hợp lệ phải có chiều rộng và chiều cao dương. Khi chuyển sang dữ liệu Faster R-CNN, tọa độ được đưa về dạng pixel và nhãn được chuyển theo quy ước tương ứng [11], [14].
Kiểm tra dữ liệu cần có cả kiểm tra tự động và xem ảnh phủ nhãn. Các lỗi cần rà soát gồm thiếu bounding box, bounding box lệch, nhầm lớp, ID vượt phạm vi và ảnh hỏng. Nhóm nên kiểm tra chéo một phần nhãn giữa các thành viên, ghi nguyên nhân chỉnh sửa và giữ lại phiên bản dữ liệu dùng cho mỗi thí nghiệm.

### 2.7.3 Chia tập và kiểm soát rò rỉ dữ liệu

Tập train phục vụ tối ưu tham số; validation phục vụ chọn cấu hình; test dùng đánh giá cuối cùng. Với ảnh trích từ video, đơn vị chia tập nên là video, camera hoặc nhóm cảnh có liên quan, thay vì chia ngẫu nhiên từng khung hình. Những khung hình gần nhau hoặc cùng một phương tiện xuất hiện trong cả train và test có thể làm phép đánh giá kém độc lập.
Tỉ lệ chia cụ thể được xác định sau khi kiểm kê nguồn dữ liệu, đồng thời phải kiểm tra đủ mẫu của từng lớp ở mỗi tập. Nếu thiếu camera độc lập, nhóm cần ghi giới hạn này thay vì gọi test là kiểm chứng trên mọi bối cảnh. Danh sách file của các tập và quy tắc loại ảnh trùng phải được lưu để có thể tái lập.

## 2.8 Các tiêu chí đánh giá


### 2.8.1 IoU và quy tắc ghép bounding box

IoU đo mức chồng lấp giữa bounding box dự đoán và bounding box chuẩn. Trong đánh giá detection theo lớp, một dự đoán chỉ là TP khi được ghép một lần với một bounding box chuẩn cùng lớp và IoU đạt ngưỡng. Dự đoán dư hoặc không ghép được tạo FP; đối tượng chuẩn không được nhận dạng đúng tạo FN. Dự đoán sai lớp gây FP cho lớp dự đoán và FN cho lớp thật. Mỗi nhãn chuẩn chỉ được ghép tối đa một lần [16].

Hình 2.7  Minh họa bounding box chuẩn và bounding box dự đoán khi tính IoU

### 2.8.2 Precision, Recall, F1 và mAP

Precision phản ánh tỉ lệ dự đoán đúng trong các dự đoán; Recall phản ánh tỉ lệ đối tượng thật được tìm thấy đúng. AP tổng hợp đường Precision–Recall khi thay đổi ngưỡng confidence; mAP là trung bình AP của các lớp. Ngoài mAP@0.5, nhóm có thể báo cáo mAP@0.5:0.95 để xem mức độ chính xác định vị ở các ngưỡng khắt khe hơn [16].
Cần công bố kết quả từng lớp và số instance tương ứng. Khi một lớp không có mẫu chuẩn trong test, không nên trình bày một con số như thể lớp đó đã được đánh giá đầy đủ. Confusion matrix cần nêu quy ước chuẩn hóa và cách xử lý background; một ô đường chéo cao không tự đại diện cho Accuracy tổng thể.

### 2.8.3 Định nghĩa Accuracy phân loại trong đề tài

Vì detector trả về nhiều bounding box, báo cáo phải tách độ đúng nhãn khỏi khả năng tìm thấy xe. Đề tài dự kiến đo Accuracy phân loại có điều kiện trên các cặp bounding box được ghép theo vị trí. Cụ thể, tại một ngưỡng confidence cố định chọn trên validation, các dự đoán được xét theo confidence giảm dần; mỗi dự đoán ghép với bounding box chuẩn chưa dùng có IoU lớn nhất và đạt 0,5, không yêu cầu cùng lớp ở bước ghép.
M_match là số cặp ghép theo vị trí; M_correct là số cặp có nhãn dự đoán đúng. Accuracy theo loại xe được tính trên các cặp có lớp chuẩn tương ứng. Nếu mẫu số bằng 0, kết quả được ghi là không đủ dữ liệu, không tự gán 0% hoặc 100%. Đây là phép đo có điều kiện do đề tài định nghĩa; nó không phải mAP hoặc phép đo đầy đủ của toàn detector.
Coverage_loc cho biết tỉ lệ bounding box chuẩn có dự đoán được ghép theo vị trí. Đề tài báo cáo (2.10) cùng (2.11), Precision, Recall và mAP để Accuracy cao trên một số ít xe không che khuất bỏ sót. Ví dụ minh họa cách tính, không phải kết quả thực nghiệm: với 100 xe chuẩn, ghép được 80 và đúng lớp 76, Accuracy_cls|match là 95%, Coverage_loc là 80%, còn tỉ lệ xe chuẩn vừa được định vị vừa đúng lớp là 76%.
Mục tiêu Accuracy ≥ 0,90 trong đề tài được gắn với phép đo (2.10) và cần nêu rõ điều kiện đo khi báo cáo. Kết quả trên các cặp bounding box đã ghép được phân biệt với kết quả classification trên ảnh cắt chuẩn của từng xe. Tăng ngưỡng confidence để chỉ giữ ít dự đoán dễ không được xem là bằng chứng duy nhất hoàn thành hệ thống.

### 2.8.4 Tốc độ xử lý và khả năng tái lập

Tốc độ model-only và tốc độ toàn pipeline cần được phân biệt. Phép đo của demo dự kiến bao gồm tiền xử lý, suy luận và hậu xử lý; thời gian đọc, hiển thị hoặc ghi video được ghi riêng hoặc được nêu rõ nếu tính vào tổng. Nhóm cần warm-up, đồng bộ GPU khi đo thời gian bất đồng bộ, chạy lặp và công bố phần cứng, batch size, precision và kích thước đầu vào.
So sánh FPS chỉ có ý nghĩa khi điều kiện đo tương thích. Nếu các mô hình dùng cấu hình resize khác nhau theo thư viện, báo cáo phải ghi rõ. Chỉ số tốc độ từ tài liệu hoặc benchmark công khai được xem là thông tin tham khảo, không thay cho phép đo trên hệ thống của nhóm.

## 2.9 Công nghệ và công cụ dự kiến

Python và PyTorch là nền tảng lập trình và huấn luyện. Ultralytics được dùng cho YOLOv8; Torchvision được dùng cho Faster R-CNN. OpenCV và Pillow hỗ trợ đọc, xử lý và hiển thị ảnh/video. Công cụ gán nhãn được lựa chọn theo khả năng xuất bounding box và quản lý dữ liệu; nhóm lưu hướng dẫn nhãn cùng với các tệp dữ liệu.
Môi trường chạy cần có tệp mô tả phiên bản thư viện và cấu hình thực nghiệm. Mỗi lần huấn luyện lưu nguồn trọng số, manifest dữ liệu, seed, log, checkpoint và tiêu chí lựa chọn. Công cụ theo dõi có thể là log của thư viện hoặc TensorBoard; việc lựa chọn giao diện không thay thế yêu cầu lưu minh chứng có thể kiểm tra.

## 2.10 Tổng kết chương và định hướng phương pháp

YOLOv8 và Faster R-CNN ResNet50-FPN được lựa chọn để so sánh trên cùng dữ liệu năm lớp. Chất lượng theo lớp, khả năng định vị và tốc độ là các nội dung cần kiểm chứng. Cơ sở lý thuyết của chương này dẫn tới thiết kế thí nghiệm ở Chương 3, triển khai ở Chương 4 và đánh giá kết quả ở Chương 5.


# CHƯƠNG 3  PHƯƠNG PHÁP ĐỀ XUẤT

Chương này trình bày phương pháp hệ thống phát hiện và đếm phương tiện giao thông Việt Nam được đề xuất trong đồ án. Nội dung tập trung mô tả thiết kế cuối cùng — bao gồm dữ liệu, mô hình, thuật toán theo vết và tiêu chí đánh giá — mà không đi sâu vào các cấu hình thăm dò hay thí nghiệm phụ (các nội dung này được trình bày trong Chương 4).

## 3.1 Tổng quan hệ thống


### 3.1.1 Kiến trúc hai luồng

Hệ thống gồm hai quy trình xử lý tách biệt nhau. Quy trình thứ nhất dùng để huấn luyện mô hình, quy trình thứ hai dùng mô hình đó để xử lý video giao thông. Hai quy trình chạy vào những thời điểm khác nhau và dùng loại dữ liệu khác nhau, nhưng cùng dựa trên một mô hình phát hiện phương tiện là YOLOv8s:
Luồng huấn luyện (offline): Đây là giai đoạn chuẩn bị, được thực hiện trước và không yêu cầu xử lý theo thời gian thực. Mục đích là giúp mô hình học cách nhận biết các loại phương tiện. Luồng này gồm bốn bước:
• Chuẩn hóa ảnh và nhãn: dữ liệu được thu thập từ nhiều nguồn khác nhau, và mỗi nguồn có thể đặt tên lớp hoặc định dạng nhãn theo cách riêng. Vì vậy, toàn bộ dữ liệu được đưa về cùng một quy ước trước khi huấn luyện.
• Chia tập theo video: các khung hình (frame) lấy từ cùng một video thường rất giống nhau. Nếu chia ngẫu nhiên từng frame vào các tập, một frame ở tập huấn luyện có thể có frame gần giống ở tập kiểm tra, khiến kết quả đánh giá cao hơn thực tế. Do đó, tất cả frame của cùng một video được xếp vào cùng một tập (huấn luyện, kiểm định hoặc kiểm tra).
• Fine-tune từ trọng số pretrained COCO: thay vì huấn luyện từ đầu, nhóm bắt đầu từ mô hình YOLOv8 đã được huấn luyện sẵn trên bộ dữ liệu COCO, vốn đã biết nhận diện nhiều đối tượng thông dụng [9]. Sau đó mô hình được huấn luyện tiếp trên dữ liệu giao thông của đề tài để chuyên biệt hóa cho các loại phương tiện cần nhận dạng.
• Chọn checkpoint tốt nhất: trong quá trình huấn luyện, mô hình được lưu lại nhiều lần, mỗi bản lưu gọi là một checkpoint. Nhóm chọn checkpoint có kết quả tốt nhất trên tập kiểm định (validation) để sử dụng cho các bước sau.
Luồng suy luận (online): Đây là giai đoạn ứng dụng, sử dụng mô hình đã huấn luyện để xử lý video thực tế. Luồng này gồm năm bước:
• Đọc video theo frame: video được tách thành chuỗi các khung hình liên tiếp.
• Phát hiện phương tiện trên từng frame: mô hình xác định vị trí (bounding box) và loại của từng phương tiện xuất hiện trong mỗi khung hình.
• Theo vết bằng ByteTrack: mô hình phát hiện chỉ xử lý từng frame độc lập, nên không biết xe ở frame này có phải là xe ở frame trước hay không. Bộ theo vết ByteTrack [19] liên kết các kết quả phát hiện giữa các frame liên tiếp và gán cho mỗi phương tiện một mã định danh (track ID). Nhờ đó, một xe xuất hiện qua nhiều frame vẫn chỉ được tính là một xe.
• Phát hiện giao cắt vạch đếm: một vạch ảo được đặt trên khung hình. Khi quỹ đạo của một phương tiện (xác định theo track ID) cắt qua vạch này, hệ thống ghi nhận một lượt xe đi qua.
• Xuất kết quả: các sự kiện đếm được lưu vào tệp CSV, đồng thời hệ thống xuất video đã được chú thích để người dùng kiểm tra trực quan.

### 3.1.2 Pipeline xử lý video

Pipeline suy luận có thể biểu diễn như sau:

Hình 3.1  Pipeline xử lý video: Video đầu vào → YOLOv8s detect → ByteTrack gán track ID → Counter phát hiện giao cắt vạch → CSV sự kiện và video chú thích
Ở đây YOLOv8s là mạng CNN chính thực hiện phát hiện; ByteTrack là thuật toán theo vết [19] (không phải mạng CNN thứ hai) chỉ liên kết các detection liên tiếp dựa trên IoU và điểm tin cậy.


## 3.2 Dữ liệu và quy ước nhãn


### 3.2.1 Nguồn dữ liệu

Phương pháp đề xuất sử dụng bốn nguồn dữ liệu giao thông Việt Nam cho tập huấn luyện và đánh giá:
Bảng 3.1  Thống kê các nguồn dữ liệu
Ngoài bốn nguồn trên, video CCTV Đà Nẵng (29.195 frame, 1 video) được tách riêng làm tập dữ liệu ngoài phân phối (out-of-distribution — OOD) để đánh giá khả năng tổng quát hoá của mô hình sang domain CCTV khác địa điểm.
[COMMENT_GV] Bảng 3.1 cần thêm cột số khung hình sau lấy mẫu, số đối tượng mỗi lớp, và thống kê kích thước hộp (nhỏ, vừa, lớn theo chuẩn COCO). Chú thích Hình 3.2 ghi "camera đêm Hà Nội" trong khi mục 3.8.3 ghi dữ liệu "chủ yếu ở TP.HCM và Cần Thơ", các em kiểm tra lại nguồn gốc địa lý của SOICT Night và ghi thống nhất. Cần ghi rõ giấy phép sử dụng và trích dẫn của từng bộ dữ liệu.

Hình 3.2  Ví dụ ảnh từ bộ dữ liệu SOICT Night (camera đêm Hà Nội)

Hình 3.3  Ví dụ ảnh từ bộ dữ liệu SOICT 2024 (camera ngày TP.HCM)

Hình 3.4  Ví dụ ảnh từ bộ dữ liệu Cần Thơ

Hình 3.5  Ví dụ ảnh từ bộ dữ liệu VNv3

Hình 3.6  Ví dụ ảnh từ video CCTV Đà Nẵng (tập OOD)

### 3.2.2 Schema nhãn 4 lớp

Tất cả nguồn dữ liệu được quy đổi về cùng một schema 4 lớp phương tiện phù hợp với đặc thù giao thông Việt Nam:
Bảng 3.2  Schema nhãn 4 lớp
Các lớp khác xuất hiện trong dữ liệu nguồn (bicycle, auto-rickshaw, van, person, …) được loại bỏ hoàn toàn chứ không gộp sang lớp gần nhất về ngữ nghĩa. Nguyên tắc này nhằm tránh gây nhầm lớp — ví dụ ánh xạ van sang truck có thể làm mô hình học nhầm hình dáng (van gần SUV/minivan hơn xe tải thùng).

### 3.2.3 Định dạng nhãn YOLO

Mỗi ảnh đi kèm một tệp nhãn .txt cùng tên trong thư mục nhãn song song. Mỗi dòng trong tệp nhãn biểu diễn một bounding box theo định dạng [11]:
class_id  cx  cy  w  h
Trong đó (cx, cy) là toạ độ tâm hộp, (w, h) là chiều rộng và chiều cao hộp, đã được chuẩn hoá về khoảng [0, 1] theo kích thước ảnh gốc. Chuẩn hoá cho phép tái sử dụng cùng tệp nhãn với mọi độ phân giải resize.

Hình 3.7  Minh hoạ hệ toạ độ YOLO trên ảnh thực từ bộ SOICT 2024 (camera ngày): origin ở góc trên trái, trục +X sang phải, +Y xuống dưới; tâm bbox (cx, cy) và kích thước (w, h) đều chuẩn hoá trong khoảng [0, 1] theo kích thước ảnh gốc

### 3.2.4 Chiến lược kiểm tra chất lượng dữ liệu

Phương pháp đề xuất áp dụng chiến lược kiểm tra hai tầng trước mỗi lần huấn luyện, cả hai tầng đều phải đạt chuẩn kiểm tra mới cho phép training bắt đầu (fail-hard):
Tầng 1: Kiểm tra toàn vẹn nhãn (validate_labels.py): kiểm tra 7 ràng buộc chất lượng ở cấp độ từng bounding box:
Bảng 3.3  Các ràng buộc kiểm tra toàn vẹn nhãn
Kết quả validation lưu vào tệp JSON (vd label_validation_v5.json) kèm phân bố lớp để làm bằng chứng tái lập.
Tầng 2: Kiểm tra rò rỉ cấp video (audit_leak.py, trình bày chi tiết trong §3.3): đảm bảo không có video nào xuất hiện trong nhiều split cùng lúc.
Căn cứ thiết kế hai tầng độc lập:
• Tầng 1 (bounding box) và tầng 2 (video-level) đo hai loại lỗi hoàn toàn khác nhau. Vi phạm tầng này không bao hàm tầng kia.
• Mỗi tầng là một script độc lập, chạy nhanh (vài giây đến vài chục giây cho dataset V5), có thể tích hợp vào pipeline CI/CD của dự án.
• Ultralytics tự kiểm tra một phần ràng buộc toàn vẹn khi quét dataset lần đầu [20] (cảnh báo trên console + lưu vào .cache), nhưng chỉ warn-and-skip chứ không fail-hard. Script validate_labels.py bổ sung cơ chế fail-hard chủ động để kiểm soát chất lượng trước khi tốn tài nguyên GPU.

## 3.3 Chia tập theo video và kiểm soát rò rỉ


### 3.3.1 Vấn đề của việc chia ngẫu nhiên theo frame

Camera CCTV cố định quay 25–30 frame mỗi giây, giữa hai frame liên tiếp:
• Background giống nhau gần như hoàn toàn (camera không di chuyển)
• Lane marking, biển báo, cây cối giữ nguyên
• Ánh sáng và thời tiết chỉ thay đổi rất chậm
• Phương tiện di chuyển vài pixel
Hai frame cách nhau 0,04 giây về bản chất là hai bản sao gần như y hệt. Nếu chia ngẫu nhiên theo frame, một frame vào train và frame kế tiếp vào test, mô hình sẽ "ghi nhớ cảnh" thay vì học khái niệm phương tiện — dẫn đến chỉ số đánh giá cao giả tạo (data leakage).

### 3.3.2 Nguyên tắc chia theo video

Phương pháp đề xuất lấy video ID làm đơn vị chia — mỗi video chỉ xuất hiện trong một split duy nhất (train HOẶC val HOẶC test). Nguyên tắc này đảm bảo mô hình được đánh giá trên các cảnh (scene) hoàn toàn chưa từng gặp trong huấn luyện.

### 3.3.3 Cấu hình chia dataset merged_V5

[COMMENT_GV] Chia 10 video train, 3 validation, 2 test là đúng nguyên tắc, nhưng tập test chỉ 2 video nên kết quả phụ thuộc mạnh vào 2 camera đó. Các em cần đảm bảo test có cả cảnh ngày và đêm; nếu được, thực hiện thêm đánh giá chéo theo video (ví dụ 3 lượt chia khác nhau) và báo cáo trung bình ± độ lệch chuẩn của mAP.
Áp dụng nguyên tắc trên, 15 video được chia như sau:
Bảng 3.4  Cấu hình chia tập V5 theo video

### 3.3.4 Chiến lược lấy mẫu để giảm dư thừa

Vì các video dense (hàng nghìn frame liên tiếp) tạo ra sự dư thừa cao, phương pháp áp dụng lấy mẫu stride theo nguyên tắc:
Bảng 3.5  Bước lấy mẫu (stride) theo loại video

### 3.3.5 Xác minh không rò rỉ

Sau khi chia dữ liệu thành các tập huấn luyện (train), kiểm định (val) và kiểm tra (test), nhóm chạy script audit_leak.py để kiểm tra xem việc chia tập có thật sự tách bạch hay không. Script này xác nhận hai điều kiện:
• Không có video nào xuất hiện ở hơn một tập: số video giao nhau giữa các tập bằng 0, tức mỗi video chỉ thuộc đúng một tập.
• Không có frame nào bị rò rỉ: tỷ lệ frame của tập val và test thuộc về video đã có trong tập train bằng 0%. Điều này đảm bảo mô hình chưa từng “nhìn thấy” bất kỳ cảnh nào của tập kiểm tra trong lúc huấn luyện, nên kết quả đánh giá phản ánh đúng khả năng nhận dạng trên dữ liệu mới.
• Kết quả kiểm tra được lưu vào một tệp JSON (ví dụ leak_audit_v5.json). Cùng với đó, nhóm lưu tệp split_manifest.json, ghi lại chính xác từng video được xếp vào tập nào. Hai tệp này đóng vai trò là bằng chứng tái lập cho phương pháp thực nghiệm:
• Bất kỳ ai muốn kiểm chứng đều có thể mở tệp manifest để xem cách chia cụ thể, rồi chạy lại script audit để xác nhận kết quả mà không phải tin vào mô tả bằng lời trong báo cáo.

## 3.4 Tiền xử lý và tăng cường dữ liệu


### 3.4.1 Pipeline tiền xử lý

Mỗi ảnh training được xử lý qua bốn bước trước khi đưa vào mạng:
• Letterbox resize: ảnh được resize về 960 × 960 giữ nguyên tỉ lệ khung hình gốc; phần trống được đệm màu xám (114, 114, 114) để hình dạng phương tiện không bị méo.
• Chuyển thứ tự kênh màu: từ BGR (OpenCV mặc định) sang RGB (YOLOv8 yêu cầu).
• Chuẩn hoá pixel: giá trị [0, 255] được chia 255 để đưa về [0.0, 1.0] float32.
• Cache metadata: ở lần huấn luyện đầu, Ultralytics quét dataset và lưu metadata vào tệp .cache cạnh thư mục nhãn; các lần sau đọc trực tiếp từ cache.
Mọi phép biến đổi hình học (lật ngang, co giãn, tịnh tiến, mosaic [15]) đều cập nhật bounding box đồng bộ với ảnh. Augmentation chỉ áp dụng cho tập train; validation và test giữ nguyên để đo lường phản ánh đúng phân phối dữ liệu.

### 3.4.2 Lựa chọn độ phân giải đầu vào

Phương pháp đề xuất sử dụng imgsz = 960 cho cả huấn luyện và suy luận. Căn cứ chọn:
• Phân phối độ phân giải nguồn: 85% dữ liệu đến từ SOICT có kích thước gốc 1280 × 720. Kích thước 960 giữ được chi tiết của các đối tượng nhỏ (xe máy ở xa camera), đồng thời không phóng to quá mức các ảnh có kích thước nhỏ hơn (416×416 của bộ Cần Thơ, 640×640 của VNv3).
• Ràng buộc phần cứng: với GPU 12 GB VRAM, kích thước 960 vẫn đủ bộ nhớ để huấn luyện với batch 16, giúp ước lượng gradient ổn định hơn so với batch 8 (nếu dùng imgsz 1280 phải giảm batch xuống 8).
• Chi phí tính toán cân bằng: số điểm ảnh ở kích thước 960 gấp 2,25 lần so với 640, nhưng mức cải thiện khả năng phát hiện đối tượng nhỏ được xem là xứng đáng với chi phí tính toán tăng thêm.

### 3.4.3 Chiến lược tăng cường dữ liệu

Phương pháp kết hợp các phép tăng cường mặc định của Ultralytics YOLOv8 [20] với một số thiết lập bổ sung phù hợp với đặc thù dữ liệu giao thông Việt Nam (cảnh đông, nhiều phương tiện che khuất, mất cân bằng lớp).
(a) Tăng cường mặc định được giữ nguyên (Bảng 3.6):
Bảng 3.6  Tham số tăng cường dữ liệu mặc định
(b) Thiết lập bổ sung cho dữ liệu VN, gồm mixup [21] và lịch learning rate cosine [22] (Bảng 3.7):
Bảng 3.7  Thiết lập huấn luyện bổ sung cho dữ liệu Việt Nam
(c) Validation và test: chỉ áp dụng letterbox resize, chuyển kênh và chuẩn hoá; không áp dụng mixup, mosaic, flip hay bất kỳ augmentation nào — để mAP đo được phản ánh đúng phân phối gốc.

## 3.5 Mô hình phát hiện YOLOv8s


### 3.5.1 Lý do chọn YOLOv8s

YOLOv8s là phiên bản small (nhỏ) trong họ YOLOv8 của Ultralytics [6], có đặc điểm:
• Số tham số: 11,1 triệu (nhỏ gọn, phù hợp triển khai trên thiết bị có cấu hình hạn chế như máy tính nhúng hoặc thiết bị di động)
• GFLOPs: 28,6
• Kiến trúc anchor-free, backbone CSPDarknet [15], neck PAN-FPN [23], [24], head Decoupled Head [6]
• Cân bằng tốc độ/độ chính xác: nhanh hơn YOLOv8m/l và đủ chính xác cho bài toán 4 lớp

### 3.5.2 Cấu hình huấn luyện

Bảng 3.8  Cấu hình huấn luyện YOLOv8s

### 3.5.3 Hàm mất mát

YOLOv8 sử dụng ba thành phần loss kết hợp [12]:
• Box loss (regression): CIoU loss cho bounding box [25]
• Classification loss: Binary Cross-Entropy có class weights
• DFL (Distribution Focal Loss): giúp ước lượng vị trí các cạnh của bounding box chính xác hơn [26]
Trọng số mặc định box=7.5, cls=0.5, dfl=1.5 được giữ nguyên.

### 3.5.4 Tiêu chí lựa chọn checkpoint

Checkpoint tốt nhất được chọn dựa trên mAP@0.5 trên tập validation, đồng thời xem xét AP của từng lớp để tránh chọn mô hình có kết quả trung bình cao nhưng yếu ở các lớp quan trọng (bus, truck). Tập test không tham gia quá trình chọn checkpoint, nhằm tránh việc kết quả trên test bị đánh giá cao hơn thực tế (selection bias).

## 3.6 Theo vết và đếm phương tiện


### 3.6.1 Theo vết với ByteTrack

Sau khi mô hình YOLOv8s phát hiện phương tiện ở mỗi frame, thuật toán ByteTrack [19] liên kết các detection giữa các frame liên tiếp để gán ID bền vững cho mỗi phương tiện.
ByteTrack hoạt động dựa trên:
• IoU giữa detection ở frame hiện tại và track hiện có
• Confidence score để phân biệt detection chất lượng cao/thấp
• Kalman filter [27] dự đoán vị trí track ở frame kế tiếp
ByteTrack được cấu hình qua tham số tracker="bytetrack.yaml" trong hàm track_stream của Ultralytics [28].

### 3.6.2 Thuật toán đếm qua vạch ảo

Module đếm được thiết kế dựa trên phát hiện giao cắt đoạn thẳng:
Bước 1: Người dùng xác định một hoặc nhiều vạch đếm (counting line) trên khung hình bằng hai điểm đầu và cuối p₁, p₂, đồng thời chọn hướng đếm: đếm cả hai chiều, chỉ đếm từ trái sang phải hoặc chỉ đếm từ phải sang trái.
Bước 2: Với mỗi phương tiện (xác định bằng track ID), hệ thống lưu lại vị trí tâm bounding box của phương tiện đó ở frame trước.
Bước 3: Ở frame hiện tại, hệ thống xác định vị trí tâm mới của phương tiện. Nếu đoạn thẳng nối tâm cũ với tâm mới cắt qua vạch đếm, hệ thống ghi nhận một sự kiện đếm gồm:
• Frame index và timestamp
• Tên vạch
• Hướng di chuyển (trái sang phải hoặc phải sang trái), xác định theo việc phương tiện đi từ phía nào của vạch sang phía nào (dựa trên dấu của tích có hướng giữa vạch đếm và hướng di chuyển)
• Track ID và class ID của phương tiện
Bước 4: Sau khi được đếm, mỗi cặp (vạch đếm, track ID) được đánh dấu lại, nhờ đó mỗi phương tiện chỉ được đếm một lần trên mỗi vạch, kể cả khi quỹ đạo cắt qua vạch nhiều lần.

### 3.6.3 Output

Hệ thống xuất hai loại kết quả:
• CSV sự kiện: danh sách sự kiện đếm theo thời gian với đầy đủ metadata
• Video chú thích: vẽ bounding box màu theo class, track ID, vạch đếm và bảng thống kê overlay realtime

## 3.7 Tiêu chí đánh giá


### 3.7.1 Chỉ số phát hiện (detection metrics)

Mô hình phát hiện được đánh giá bằng các chỉ số chuẩn COCO [9], [16]:
Bảng 3.9  Các chỉ số phát hiện
Báo cáo per-class AP để thấy ảnh hưởng của mất cân bằng lớp, không chỉ dùng số trung bình.

### 3.7.2 Chỉ số đếm (counting metrics)

[COMMENT_GV] Để tính MAE, MAPE của bài toán đếm, các em cần số đếm thật: đếm tay số phương tiện theo từng lớp đi qua vạch ảo trên mỗi video test và video Đà Nẵng (có thể chia nhỏ thành các đoạn 1 phút để có nhiều mẫu đánh giá). Ghi rõ quy trình đếm tay, số người đếm, cách xử lý trường hợp không chắc chắn.
Đánh giá đếm bằng sai số giữa số sự kiện dự đoán và ground truth:
Trong đó, (3.1) là sai số tuyệt đối trung bình (MAE – Mean Absolute Error), (3.2) là sai số phần trăm tuyệt đối trung bình (MAPE – Mean Absolute Percentage Error) và (3.3) là sai số tổng trên toàn bộ tập đánh giá; N là số mẫu đánh giá, Npred,i và NGT,i lần lượt là số lượt đếm dự đoán và số lượt thực tế của mẫu thứ i.
Độ chính xác đếm (nếu dùng) được định nghĩa:
Lưu ý quan trọng: đây là chỉ số đếm trên pipeline tổng (detection + tracking + counting), không phải Accuracy phân loại của YOLO. Không được trình bày thành "Accuracy > X%" chung chung mà không nêu rõ context.

### 3.7.3 Chỉ số phân tích lỗi

Phân tích lỗi detection được thực hiện ở cấp độ bounding box:
• True Positive (TP): detection khớp GT (IoU ≥ 0.5) và đúng class
• False Positive (FP): detection không khớp GT hoặc sai class
• False Negative (FN): GT không có detection nào match
• Class confusion: match vị trí nhưng sai class
TP/FP/FN được tổng hợp theo lớp, theo nguồn dữ liệu (SOICT/cantho/vnv3) và theo kích thước object (small/medium/large) để xác định điểm yếu cụ thể của mô hình.

### 3.7.4 Chỉ số tốc độ

FPS (frame per second) được đo trên 100 frame đầu sau warm-up, với:
• Precision FP16 (half precision)
• Imgsz 960
• GPU NVIDIA RTX 3500 Ada Laptop (12 GB VRAM)
• Bao gồm cả thời gian tracking + đọc/ghi video
[COMMENT_GV] Mục 3.8 Môi trường và tái lập chuyển sang đầu Chương 4. Mục 3.8.3 Giới hạn chuyển vào phần Kết luận (hạn chế).

## 3.8 Môi trường và tái lập


### 3.8.1 Phần cứng và phần mềm

• GPU: NVIDIA RTX 3500 Ada Laptop (12 GB VRAM)
• Python: 3.10.20
• PyTorch [29]: 2.5.1 + CUDA 12.4
• Ultralytics [6]: 8.4.142
• OS: Linux (WSL2 trên Windows)

### 3.8.2 Tái lập kết quả

Mỗi kết quả báo cáo được gắn với:
• Phiên bản mã nguồn (git commit hash)
• Tệp YAML cấu hình dataset
• Tệp manifest split (split_manifest.json)
• Seed cố định (42)
• Checkpoint .pt và results.csv

### 3.8.3 Giới hạn của phương pháp

Phương pháp đề xuất có các giới hạn sau cần ghi nhận:
• OOD evaluation nhỏ: chỉ có 1 video Đà Nẵng làm OOD → kết luận về tổng quát hoá cross-domain cần diễn giải cẩn trọng.
• Nhãn chỉ có bounding box: không có polygon segmentation → một số kỹ thuật augmentation nâng cao (vd copy_paste của Ultralytics) không áp dụng được. Việc tự sinh mask bằng SAM [30] hoặc tương tự để kích hoạt lại copy_paste là hướng phát triển.
• Phạm vi địa lý hạn chế: 15 video source hiện tại chủ yếu ở TP.HCM và Cần Thơ → mô hình có thể bias về domain đô thị miền Nam. Mở rộng dataset VN đa vùng (Hà Nội, Đà Nẵng, miền núi) là hướng phát triển.
• Chưa có nhãn tracking per-frame: không đánh giá được IDF1, MOTA, HOTA chính thức. Tracker ByteTrack chỉ được đánh giá gián tiếp qua chất lượng counting.
• Dataset mở rộng được: pipeline hỗ trợ thêm nguồn dữ liệu mới trong tương lai bằng cách bổ sung mẫu nhận dạng video_id trong audit_leak.py và cập nhật SPLIT_PLAN trong script build dataset.

## 3.9 Tiểu kết chương

Chương này đã trình bày phương pháp đề xuất một cách hệ thống:
• Dữ liệu: 4 nguồn Việt Nam (15 video, 23.383 frame raw) + 1 OOD Đà Nẵng, schema 4 lớp thống nhất.
• Chia tập theo video: 10 train / 3 val / 2 test, không chia theo frame, verify bằng audit tự động.
• Tiền xử lý và augmentation: pipeline chuẩn Ultralytics + bổ sung mixup và cos_lr cho đặc thù VN.
• Mô hình: YOLOv8s fine-tune từ COCO với imgsz 960, 40 epoch, patience 15.
• Theo vết và đếm: ByteTrack + thuật toán giao cắt vạch với anti-duplicate counting.
• Đánh giá: mAP cho detection, MAE/MAPE cho counting, phân tích lỗi TP/FP/FN theo nhiều chiều.
Phương pháp này được triển khai và thực nghiệm trong Chương 4, bao gồm cả quá trình phát hiện và khắc phục rò rỉ dữ liệu, các thí nghiệm khảo sát (ablation study) và kết quả so sánh với baseline.
[COMMENT_GV] Báo cáo đang kết thúc ở đây. Các em bổ sung CHƯƠNG 4. THỰC NGHIỆM VÀ ĐÁNH GIÁ (khoảng 18 trang) với dàn ý:
4.1. Môi trường và cấu hình thực nghiệm. 4.2. Kết quả huấn luyện: đường cong loss, mAP theo epoch. 4.3. So sánh YOLOv8s và Faster R-CNN trên test: bảng Precision, Recall, mAP@0.5, mAP@0.5:0.95, AP từng lớp, AP theo kích thước, FPS. 4.4. Kết quả trên tập ngoài phân phối Đà Nẵng và mức suy giảm. 4.5. Ablation: không có và có dữ liệu bổ sung, các mức augmentation (mixup, cos_lr), imgsz 640 và 960. 4.6. Kết quả đếm: bảng số đếm thật và số đếm hệ thống theo lớp, MAE, MAPE, theo ngày và đêm. 4.7. Phân tích lỗi: ma trận nhầm lẫn 4 lớp (ví dụ xe tải nhỏ và ô tô con), lỗi đối tượng nhỏ, che khuất, ban đêm; ảnh minh họa. 4.8. Thảo luận, trả lời từng câu hỏi nghiên cứu.
CHƯƠNG 5. XÂY DỰNG HỆ THỐNG (khoảng 5 trang): giao diện chọn video, vẽ vạch đếm, hiển thị hộp và ID, bảng đếm theo lớp, xuất CSV; ảnh chụp màn hình; kiểm thử và tốc độ. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (khoảng 3 trang, Heading 1 riêng).


# TÀI LIỆU THAM KHẢO

[1]	S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 28, 2015, pp. 91–99. [Online]. Available: https://proceedings.neurips.cc/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html
[2]	J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You Only Look Once: Unified, Real-Time Object Detection,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 779–788. [Online]. Available: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf
[3]	K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 770–778. [Online]. Available: https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html
[4]	L. Yang, P. Luo, C. C. Loy, and X. Tang, “A Large-Scale Car Dataset for Fine-Grained Categorization and Verification,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2015, pp. 3973–3981. [Online]. Available: https://openaccess.thecvf.com/content_cvpr_2015/html/Yang_A_Large-Scale_Car_2015_CVPR_paper.html
[5]	W. Liu et al., “SSD: Single Shot MultiBox Detector,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2016, pp. 21–37. [Online]. Available: https://arxiv.org/abs/1512.02325
[6]	Ultralytics, “Explore Ultralytics YOLOv8,” Ultralytics Docs. Accessed: Oct. 15, 2026. [Online]. Available: https://docs.ultralytics.com/models/yolov8/
[7]	L. Wen et al., “UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking,” Comput. Vis. Image Understand., vol. 193, Apr. 2020, Art. no. 102907. [Online]. Available: https://doi.org/10.1016/j.cviu.2020.102907
[8]	F. Yu et al., “BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 2636–2645. [Online]. Available: https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_BDD100K_A_Diverse_Driving_Dataset_for_Heterogeneous_Multitask_Learning_CVPR_2020_paper.html
[9]	T.-Y. Lin et al., “Microsoft COCO: Common Objects in Context,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2014, pp. 740–755. [Online]. Available: https://arxiv.org/abs/1405.0312
[10]	Ultralytics, “YOLOv8 Detection Model Configuration,” GitHub. Accessed: Oct. 15, 2026. [Online]. Available: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8.yaml
[11]	Ultralytics, “Object Detection,” Ultralytics Docs. Accessed: Oct. 15, 2026. [Online]. Available: https://docs.ultralytics.com/tasks/detect/
[12]	Ultralytics, “Detection Loss Implementation,” GitHub. Accessed: Oct. 15, 2026. [Online]. Available: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py
[13]	TorchVision, “fasterrcnn_resnet50_fpn,” TorchVision Documentation. Accessed: Oct. 15, 2026. [Online]. Available: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
[14]	PyTorch, “TorchVision Object Detection Finetuning Tutorial,” PyTorch Tutorials. Accessed: Oct. 15, 2026. [Online]. Available: https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html
[15]	A. Bochkovskiy, C.-Y. Wang, and H.-Y. M. Liao, “YOLOv4: Optimal Speed and Accuracy of Object Detection,” 2020, arXiv:2004.10934. [Online]. Available: https://arxiv.org/abs/2004.10934
[16]	Ultralytics, “YOLO Performance Metrics,” Ultralytics Docs. Accessed: Oct. 15, 2026. [Online]. Available: https://docs.ultralytics.com/guides/yolo-performance-metrics/
[17]	ResearchGate, “YOLOv8 network architecture. The figure illustrates the backbone (C2f and SPPF modules), neck and head,” ResearchGate Scientific Diagram. Accessed: Oct. 19, 2026. [Online]. Available: https://www.researchgate.net/
[18]	ResearchGate, “Kernel of the traditional U-Net. The input matrix represents the image,” ResearchGate Scientific Diagram. Accessed: Oct. 19, 2026. [Online]. Available: https://www.researchgate.net/
[19]	Y. Zhang et al., “ByteTrack: Multi-Object Tracking by Associating Every Detection Box,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2022, pp. 1–21.
[20]	Ultralytics, “Configuration,” Ultralytics Docs. Accessed: Nov. 15, 2026. [Online]. Available: https://docs.ultralytics.com/usage/cfg/
[21]	H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, “mixup: Beyond Empirical Risk Minimization,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2018.
[22]	I. Loshchilov and F. Hutter, “SGDR: Stochastic Gradient Descent with Warm Restarts,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2017.
[23]	T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie, “Feature Pyramid Networks for Object Detection,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 2117–2125.
[24]	S. Liu, L. Qi, H. Qin, J. Shi, and J. Jia, “Path Aggregation Network for Instance Segmentation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 8759–8768.
[25]	Z. Zheng, P. Wang, W. Liu, J. Li, R. Ye, and D. Ren, “Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression,” in Proc. AAAI Conf. Artif. Intell., vol. 34, no. 7, 2020, pp. 12993–13000.
[26]	X. Li et al., “Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 33, 2020, pp. 21002–21012.
[27]	R. E. Kalman, “A New Approach to Linear Filtering and Prediction Problems,” J. Basic Eng., vol. 82, no. 1, pp. 35–45, Mar. 1960.
[28]	Ultralytics, “Multi-Object Tracking with Ultralytics YOLO,” Ultralytics Docs. Accessed: Nov. 15, 2026. [Online]. Available: https://docs.ultralytics.com/modes/track/
[29]	A. Paszke et al., “PyTorch: An Imperative Style, High-Performance Deep Learning Library,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 32, 2019, pp. 8024–8035.
[30]	A. Kirillov et al., “Segment Anything,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 4015–4026.
