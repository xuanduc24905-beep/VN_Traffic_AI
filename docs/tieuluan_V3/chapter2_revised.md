# CHƯƠNG 2  CÔNG TRÌNH LIÊN QUAN VÀ CƠ SỞ LÝ THUYẾT

Chương này khảo sát các hướng tiếp cận liên quan đến bài toán phát hiện phương tiện giao thông và trình bày các khái niệm lý thuyết cần thiết để xây dựng, đánh giá hệ thống. Nội dung được tổ chức theo trình tự: (i) khảo sát công trình liên quan, (ii) cơ sở lý thuyết CNN và các khối xây dựng mạng sâu, (iii) hai mô hình được sử dụng trong đề tài (YOLOv8 và Faster R-CNN), (iv) các kỹ thuật phụ trợ (học chuyển giao, tăng cường dữ liệu, rò rỉ dữ liệu), và (v) các tiêu chí đánh giá áp dụng cho đồ án. Thiết kế dữ liệu cụ thể của đề tài được trình bày riêng ở Chương 3.

## 2.1 Tổng quan các công trình nghiên cứu liên quan

### 2.1.1 Nhóm công trình phân loại ảnh và nhận dạng phương tiện

Mạng nơ-ron tích chập (CNN) học đặc trưng từ ảnh thay cho các bộ mô tả thủ công như HOG hay SIFT. Trong nhóm các kiến trúc sâu, He và cộng sự [3] đề xuất học phần dư với kết nối tắt (skip connection) để hỗ trợ huấn luyện mạng nhiều lớp mà không bị suy biến gradient. ResNet đã được chứng minh hiệu quả trên bài toán phân loại ImageNet và thường được sử dụng làm backbone cho các detector, trong đó có Faster R-CNN ResNet50-FPN mà đề tài này sử dụng làm mô hình đối chứng.

Yang và cộng sự [4] giới thiệu CompCars cho các nhiệm vụ phân loại dòng ô tô, xác minh và dự đoán thuộc tính. Công trình này cho thấy vai trò của dữ liệu có cấu trúc và sự khác biệt giữa ảnh web với ảnh giám sát. Tuy nhiên, nhiệm vụ phân biệt dòng ô tô cụ thể khác với nhiệm vụ phát hiện bốn nhóm xe trong cùng một ảnh của đề tài, nên kết quả hai nhiệm vụ không thể so sánh trực tiếp bằng một chỉ số Accuracy chung.

### 2.1.2 Nhóm mô hình phát hiện hai giai đoạn

Ren và cộng sự [1] đề xuất Faster R-CNN với Region Proposal Network (RPN) dùng chung đặc trưng tích chập với mạng phát hiện. Quy trình gồm hai giai đoạn: (i) RPN tạo các vùng ứng viên (region proposal), (ii) bộ phận phân loại và hồi quy bounding box xử lý các vùng này. Faster R-CNN với backbone ResNet50-FPN đã trở thành một trong những đường cơ sở (baseline) phổ biến cho nghiên cứu phát hiện đối tượng. Đề tài này sử dụng Faster R-CNN ResNet50-FPN làm mô hình đối chứng để đánh giá sự khác biệt giữa triết lý hai giai đoạn và một giai đoạn trên cùng tập dữ liệu giao thông Việt Nam.

### 2.1.3 Nhóm mô hình phát hiện một giai đoạn

Redmon và cộng sự [2] đề xuất YOLO (You Only Look Once) với triết lý thiết kế: dự đoán bounding box và xác suất lớp trực tiếp từ toàn ảnh trong một mạng thống nhất, không có bước đề xuất vùng riêng. Liu và cộng sự [5] đề xuất SSD với các default bounding box ở nhiều độ phân giải feature map. Hai công trình này đặt nền móng cho hướng phát hiện một giai đoạn với tốc độ cao hơn detector hai giai đoạn.

YOLOv8 [6] là phiên bản thuộc hệ sinh thái Ultralytics, được phân phối với nhiều kích thước mô hình (n, s, m, l, x) và trọng số pretrained trên COCO. Đặc điểm nổi bật của YOLOv8 bao gồm: (i) head tách nhánh (decoupled head) phân loại và hồi quy, (ii) thiết kế anchor-free loại bỏ việc định nghĩa trước kích thước anchor box, (iii) hàm mất mát mới kết hợp CIoU và Distribution Focal Loss. Đề tài sử dụng biến thể YOLOv8s (small) làm mô hình chính vì cân bằng tốc độ và độ chính xác, đồng thời phù hợp với giới hạn VRAM 12 GB của GPU RTX 3500 Ada Laptop.

### 2.1.4 Nhóm bộ dữ liệu và đánh giá giao thông

Wen và cộng sự [7] xây dựng UA-DETRAC cho phát hiện và theo dõi đa đối tượng trong video giao thông. Giá trị của benchmark là đặt việc đánh giá trong các cảnh và điều kiện quan sát cụ thể. Yu và cộng sự [8] giới thiệu BDD100K với dữ liệu lái xe đa dạng, nhấn mạnh khác biệt về môi trường và điều kiện thời tiết. Tuy nhiên, hai bộ dữ liệu này đều được thu thập ở Mỹ hoặc Trung Quốc, với tỉ lệ xe máy thấp và tổ chức giao thông khác biệt so với Việt Nam. COCO [9] là bộ dữ liệu quan trọng đối với các trọng số pretrained; vai trò chính của COCO trong đề tài là nguồn khởi tạo cho học chuyển giao.

Khoảng trống nghiên cứu mà đề tài hướng tới là chưa có bằng chứng so sánh định lượng giữa YOLOv8 và Faster R-CNN ResNet50-FPN trên tập dữ liệu giao thông Việt Nam được chuẩn hóa theo 4 lớp và kiểm soát rò rỉ dữ liệu cấp video. Đây là nhu cầu kiểm chứng trong phạm vi đề tài.

### 2.1.5 Tổng hợp các hướng tiếp cận

**Bảng 2.1** tổng hợp các công trình liên quan và ý nghĩa đối với đề tài.

| Công trình | Ý tưởng chính | Ý nghĩa với đề tài |
|---|---|---|
| ResNet [3] | Kết nối tắt phần dư | Backbone cho Faster R-CNN đối chứng |
| Faster R-CNN [1] | RPN + hai giai đoạn | Mô hình đối chứng |
| YOLO [2], YOLOv8 [6] | Detector một giai đoạn, anchor-free | Mô hình chính |
| SSD [5] | Multi-scale default boxes | Nền tảng one-stage |
| UA-DETRAC [7], BDD100K [8] | Benchmark giao thông quốc tế | Tham khảo định nghĩa chỉ số, schema khác VN |
| COCO [9] | Benchmark đa lớp | Nguồn pretrained weights |

## 2.2 Cơ sở lý thuyết về mạng nơ-ron tích chập

### 2.2.1 Phép tích chập và feature map

Ảnh màu được biểu diễn dưới dạng tensor ba chiều gồm chiều cao H, chiều rộng W và số kênh C (với ảnh RGB, C = 3). Lớp tích chập áp dụng bộ lọc (kernel) kích thước k × k với d kênh đầu vào tại nhiều vị trí của ảnh để tạo bản đồ đặc trưng (feature map). Kích thước đầu ra theo một chiều được tính theo công thức:

H_out = ⌊(H_in + 2p − d(k−1) − 1) / s⌋ + 1                    (2.1)

trong đó p là padding, s là stride và d là dilation. Việc chia sẻ trọng số (weight sharing) của tích chập giúp mô hình học một mẫu cục bộ có thể xuất hiện ở nhiều vị trí khác nhau, đồng thời giảm đáng kể số tham số so với mạng fully-connected.

**Hình 2.1** minh họa phép tích chập với kernel 3×3 trên một vùng cục bộ của ảnh đầu vào tạo ra một giá trị trong feature map.

### 2.2.2 Hàm kích hoạt và giảm kích thước

Hàm kích hoạt đưa tính phi tuyến vào mạng, cho phép mô hình học được các hàm phức tạp. ReLU đặt giá trị âm về 0: ReLU(x) = max(0, x). SiLU (Sigmoid Linear Unit, còn gọi là Swish) được định nghĩa: SiLU(x) = x · σ(x), trong đó σ là hàm sigmoid. YOLOv8 sử dụng SiLU trong các khối C2f vì SiLU có đạo hàm liên tục và hiệu quả hơn ReLU trong thực nghiệm.

Giảm kích thước feature map có thể thực hiện bằng max pooling hoặc tích chập với stride > 1. YOLOv8 sử dụng cách thứ hai (strided convolution) để downsample, giúp mạng học được cách giảm kích thước thay vì cố định quy tắc pooling. Cách tổ chức cụ thể này giữ lại được nhiều thông tin không gian hơn, đặc biệt quan trọng với các đối tượng nhỏ như xe máy ở xa camera.

### 2.2.3 Học phần dư và backbone ResNet

Khối học phần dư của He và cộng sự [3] có dạng:

y = F(x, {W_i}) + x                                              (2.2)

trong đó F là hàm biến đổi học được (thường gồm vài lớp tích chập), x là đầu vào và y là đầu ra. Kết nối tắt x → y cho phép gradient truyền trực tiếp qua các khối mà không bị suy giảm, cho phép huấn luyện mạng rất sâu (34, 50, 101, 152 lớp). Khi kích thước của F(x) không tương thích với x (do stride > 1 hoặc thay đổi số kênh), nhánh tắt cần phép chiếu 1×1 để điều chỉnh.

Trong đề tài này, ResNet50 đóng vai trò backbone cho Faster R-CNN, trích xuất feature map từ ảnh đầu vào. Feature Pyramid Network (FPN) [23] kết hợp các mức đặc trưng từ ResNet để tạo ra pyramid gồm 5 mức P2–P6 với các độ phân giải khác nhau, cho phép phát hiện đối tượng ở nhiều kích thước.

**Hình 2.2** Khối học phần dư với kết nối tắt đồng nhất, cho phép gradient truyền trực tiếp qua các khối giúp huấn luyện mạng rất sâu (Nguồn: [3]).

## 2.3 Bài toán phát hiện và phân loại phương tiện

Với ảnh I ∈ R^(H×W×3), detector tạo ra tập dự đoán D = {(b_i, c_i, s_i)}, trong đó b_i là bounding box dạng (x, y, w, h), c_i ∈ {0, 1, 2, 3} là nhãn lớp và s_i ∈ [0, 1] là điểm tin cậy. Nhãn chuẩn G = {(b_j^*, c_j^*)} do người gán nhãn xác định. Số phần tử |D| và |G| thường khác nhau do mô hình có thể bỏ sót hoặc dự đoán dư.

**Bảng 2.2** phân biệt ba nhiệm vụ liên quan trong thị giác máy tính.

| Nhiệm vụ | Đầu vào | Đầu ra | Ví dụ áp dụng |
|---|---|---|---|
| Phân loại ảnh | Một ảnh | Một nhãn | Nhận dạng dòng xe trên ảnh cắt |
| Phát hiện đối tượng | Một ảnh | Nhiều (box + nhãn + score) | Phát hiện phương tiện trong camera giao thông |
| Phân vùng ảnh | Một ảnh | Một nhãn/pixel | Phân vùng làn đường |

Đề tài này tập trung vào nhiệm vụ phát hiện đối tượng (hàng thứ hai). Việc thống kê số phương tiện theo lớp được thực hiện sau khi hậu xử lý detection; đếm lượt phương tiện đi qua một mặt cắt trong video cần thêm thành phần theo vết (tracking) và quy tắc đếm, được trình bày ở Chương 3.

**Hình 2.3** So sánh ba nhiệm vụ thị giác máy tính: phân loại ảnh (một nhãn/ảnh), phát hiện đối tượng (nhiều bounding box + nhãn) và phân vùng ngữ nghĩa (một nhãn/pixel).

## 2.4 Kiến trúc và cơ chế hoạt động của YOLOv8

### 2.4.1 Backbone và tổng hợp đặc trưng đa tỉ lệ

YOLOv8 detection của Ultralytics [10] có kiến trúc gồm ba thành phần: backbone trích xuất đặc trưng, neck tổng hợp đặc trưng đa tỉ lệ, và head dự đoán.

**Backbone** gồm các khối Conv (tích chập + BatchNorm + SiLU), khối C2f (Cross-Stage Partial với 2 convolution, một dạng cải tiến của C3 trong YOLOv5) và khối SPPF (Spatial Pyramid Pooling Fast). Backbone có 5 mức downsampling liên tiếp, tạo ra 5 feature map với stride 2, 4, 8, 16, 32 so với ảnh đầu vào.

**Neck** sử dụng cấu trúc PAN-FPN (Path Aggregation Network + Feature Pyramid Network) [23], [24]: FPN truyền thông tin ngữ nghĩa cao từ các mức sâu xuống các mức nông, PAN truyền thông tin định vị từ các mức nông lên các mức sâu. Kết quả là 3 feature map ở stride 8, 16, 32 chứa cả thông tin ngữ nghĩa và định vị.

**Hình 2.5** minh họa kiến trúc tổng quát của YOLOv8 gồm Backbone, Neck và Head.

### 2.4.2 Head dự đoán và cơ chế anchor-free

YOLOv8 sử dụng head tách nhánh (decoupled head) với hai luồng riêng cho phân loại và hồi quy bounding box, khác với YOLOv5 dùng head gộp. Thiết kế anchor-free [6] loại bỏ việc định nghĩa trước các kích thước anchor box; mô hình trực tiếp dự đoán bốn giá trị khoảng cách (left, top, right, bottom) từ mỗi vị trí trên feature map đến bốn cạnh của bounding box.

Ở mỗi vị trí (i, j) trên feature map ở mức stride s, head dự đoán:
- **Phân loại**: vector C chiều (với C là số lớp, C = 4 trong đề tài này) qua sigmoid, cho biết xác suất của mỗi lớp
- **Hồi quy**: 4 × (reg_max + 1) giá trị cho Distribution Focal Loss [26], sau đó lấy expected value để có 4 khoảng cách l, t, r, b

Bounding box cuối cùng được tính: (x_1, y_1, x_2, y_2) = ((j - l)·s, (i - t)·s, (j + r)·s, (i + b)·s). Điểm tin cậy cuối = max(sigmoid(cls_score)).

### 2.4.3 Hàm mất mát và hậu xử lý

Hàm mất mát tổng của YOLOv8 kết hợp ba thành phần [12]:

L = λ_box · L_box + λ_cls · L_cls + λ_dfl · L_dfl                (2.3)

với các giá trị mặc định λ_box = 7.5, λ_cls = 0.5, λ_dfl = 1.5.

- **L_box** là CIoU loss [25], đo mức chồng lấp có cân nhắc khoảng cách tâm và tỉ lệ cạnh giữa bounding box dự đoán và ground truth.
- **L_cls** là Binary Cross-Entropy (BCE) tính riêng cho mỗi lớp.
- **L_dfl** là Distribution Focal Loss [26] khuyến khích phân phối xác suất rời rạc của giá trị hồi quy tập trung quanh giá trị thật.

Khác với YOLOv3/v4/v5, YOLOv8 không sử dụng nhánh objectness riêng; điểm tin cậy cuối được lấy từ xác suất lớp cao nhất. Việc gán nhãn positive–negative cho các vị trí trên feature map dùng TaskAlignedAssigner [6].

Sau khi mô hình dự đoán, hệ thống lọc theo ngưỡng confidence (conf_threshold = 0.25 mặc định) và áp dụng Non-Maximum Suppression (NMS) với ngưỡng IoU = 0.7 để loại các bounding box trùng lặp. Lưu ý rằng ngưỡng IoU của NMS khác với ngưỡng IoU dùng để đánh giá (thường là 0.5 hoặc chuỗi 0.5:0.05:0.95).

## 2.5 Faster R-CNN làm mô hình đối chứng

### 2.5.1 Nguyên lý hai giai đoạn

Faster R-CNN [1] tổ chức xử lý thành hai giai đoạn liên tiếp. **Giai đoạn 1 (Region Proposal Network)**: backbone trích xuất feature map, RPN sliding-window qua feature map và dự đoán, tại mỗi vị trí, k anchor box (thường k = 9 với 3 tỉ lệ × 3 kích thước) kèm điểm objectness. Các anchor có điểm objectness cao được giữ lại làm vùng ứng viên (region proposal), thường 2.000 proposals cho training và 1.000 cho inference.

**Giai đoạn 2 (Detection Head)**: với mỗi proposal, hệ thống dùng RoI Align [13] để trích xuất feature có kích thước cố định (7×7) từ feature map của backbone, sau đó đưa qua hai nhánh FC: một nhánh phân loại (C+1 lớp, trong đó +1 là background) và một nhánh hồi quy bounding box (4 × C giá trị). Kết quả cuối là danh sách bounding box + nhãn + điểm tin cậy.

**Hình 2.6** minh họa kiến trúc Faster R-CNN với mạng đề xuất vùng RPN.

### 2.5.2 Lựa chọn ResNet50-FPN

Đề tài sử dụng triển khai `fasterrcnn_resnet50_fpn` của TorchVision [13] với trọng số pretrained trên COCO. Backbone ResNet50 kết hợp FPN cho ra 5 mức pyramid P2–P6; RPN và detection head hoạt động trên tất cả các mức này, cho phép phát hiện đối tượng ở nhiều kích thước khác nhau.

Khi fine-tune, lớp `roi_heads.box_predictor` được thay bằng `FastRCNNPredictor(in_channels, num_classes=5)`, trong đó 5 = 4 lớp phương tiện + 1 lớp background. Trọng số backbone và các lớp khác được giữ pretrained.

### 2.5.3 So sánh YOLOv8 và Faster R-CNN

Hai mô hình đại diện cho hai triết lý thiết kế. YOLOv8 (one-stage) tối ưu tốc độ bằng cách dự đoán trực tiếp ở tất cả vị trí mà không có bước đề xuất vùng. Faster R-CNN (two-stage) chấp nhận chi phí cao hơn để tăng độ chính xác qua bước proposal lọc trước khi phân loại. **Bảng 2.3** so sánh các đặc điểm chính của hai kiến trúc.

| Đặc điểm | YOLOv8s | Faster R-CNN ResNet50-FPN |
|---|---|---|
| Số giai đoạn | 1 | 2 |
| Anchor | Anchor-free | Anchor-based (9 anchors/vị trí) |
| Số tham số | 11,1 triệu | ~41,8 triệu |
| GFLOPs @ imgsz 960 | 28,6 | ~134 |
| Backbone | CSPDarknet | ResNet50 |
| Neck | PAN-FPN | FPN |
| Loss hồi quy | CIoU + DFL | Smooth L1 |
| Hậu xử lý | NMS lớp | NMS lớp (sau 2 giai đoạn) |

Khi so sánh trên cùng tập dữ liệu, cần đảm bảo cùng ảnh test, cùng bộ nhãn, cùng cách đánh giá. Những khác biệt bắt buộc về resize, batch, chính sách augmentation giữa hai thư viện (Ultralytics và TorchVision) phải được ghi rõ.

## 2.6 Học chuyển giao và tăng cường dữ liệu

### 2.6.1 Học chuyển giao và fine-tuning

Học chuyển giao (transfer learning) sử dụng trọng số đã được học ở nhiệm vụ nguồn (thường là phân loại ImageNet hoặc phát hiện COCO) để khởi tạo cho nhiệm vụ đích. Fine-tuning là quá trình tiếp tục huấn luyện một phần hoặc toàn bộ tham số của mô hình pretrained với dữ liệu của nhiệm vụ mới.

Đối với detector có sẵn, lớp dự đoán cuối được thay cho phù hợp với số lớp của bài toán mới; phần còn lại kế thừa trọng số pretrained. Trong đề tài này, YOLOv8s khởi tạo từ `yolov8s.pt` (pretrained COCO 80 lớp), lớp head classification được thay cho 4 lớp. Faster R-CNN khởi tạo từ `fasterrcnn_resnet50_fpn` weights COCO, lớp `box_predictor` được thay cho 5 lớp (4 + background).

Quy trình fine-tuning theo dõi mAP trên validation để chọn checkpoint tốt nhất và dừng sớm (early stopping) khi không còn cải thiện trong một số epoch nhất định (patience = 15 trong đề tài).

### 2.6.2 Tăng cường dữ liệu có bảo toàn nhãn

Tăng cường dữ liệu (data augmentation) tạo thêm các biến thể từ ảnh gốc để mô hình học được tính bất biến với các phép biến đổi. Với bài toán phát hiện, các phép biến đổi hình học (lật ngang, co giãn, tịnh tiến) phải được áp dụng đồng thời lên ảnh và bounding box. Các phép biến đổi màu sắc (HSV, contrast, brightness) chỉ ảnh hưởng ảnh.

Đề tài sử dụng tập hợp augmentation sau:
- **Hình học**: lật ngang (fliplr=0.5), co giãn (scale=0.5→0.7), tịnh tiến (translate=0.1→0.15)
- **Màu sắc**: thay đổi HSV (hsv_h=0.015, hsv_s=0.7, hsv_v=0.4)
- **Mosaic** [15]: ghép 4 ảnh thành 1 với mosaic=1.0, tắt ở 10 epoch cuối (close_mosaic=10)
- **Mixup** [21]: trộn hai ảnh với trọng số alpha, mixup=0.1

Validation và test không áp dụng augmentation để các chỉ số đánh giá phản ánh đúng phân phối gốc.

### 2.6.3 Rò rỉ dữ liệu khi chia theo khung hình

Một dạng rò rỉ dữ liệu phổ biến trong các bài toán phát hiện phương tiện là chia tập ngẫu nhiên theo khung hình (frame-level random split) thay vì theo video (video-level split). Camera giám sát giao thông cố định thường quay 25–30 khung hình mỗi giây; hai khung hình liên tiếp chỉ khác nhau một vài pixel do phương tiện di chuyển, trong khi nền (background), biển báo, mặt đường giữ nguyên.

Nếu chia ngẫu nhiên theo khung hình, một khung hình vào train và khung hình kế tiếp vào test sẽ dẫn đến hiện tượng mô hình "ghi nhớ cảnh" (scene memorization) thay vì học khái niệm phương tiện. Hậu quả là các chỉ số đánh giá trên test cao giả tạo nhưng mô hình không tổng quát hóa được sang video mới. Trong các phiên bản đầu của đề tài (V1–V4), hiện tượng này đã gây ra mAP ảo khoảng 0,95; sau khi chuyển sang chia theo video (V5), mAP honest giảm xuống 0,665 nhưng phản ánh đúng khả năng tổng quát hóa.

Nguyên tắc chống rò rỉ cấp video được triển khai trong đề tài: (i) chọn đơn vị chia là video_id hoặc camera_id, (ii) đảm bảo mỗi video chỉ xuất hiện trong đúng một tập (train HOẶC val HOẶC test), (iii) kiểm tra tự động bằng script audit sau khi chia. Chi tiết triển khai được trình bày ở §3.3.

## 2.7 Các tiêu chí đánh giá

### 2.7.1 IoU và quy tắc ghép bounding box

Intersection over Union (IoU) đo mức chồng lấp giữa hai bounding box B₁ và B₂:

IoU(B₁, B₂) = |B₁ ∩ B₂| / |B₁ ∪ B₂|                              (2.4)

Trong đánh giá detection theo lớp, một dự đoán được xem là True Positive (TP) khi: (i) được ghép với một bounding box ground truth chưa dùng, (ii) IoU ≥ ngưỡng (thường 0.5), (iii) nhãn lớp khớp. Dự đoán dư hoặc không ghép được là False Positive (FP). Ground truth không có dự đoán nào ghép đúng là False Negative (FN). Mỗi ground truth chỉ được ghép tối đa một lần [16].

**Hình 2.7** minh họa bounding box dự đoán và bounding box chuẩn khi tính IoU.

### 2.7.2 Precision, Recall, F1 và mAP

Các chỉ số cơ bản được tính theo công thức:

Precision = TP / (TP + FP)                                       (2.5)

Recall = TP / (TP + FN)                                          (2.6)

F₁ = 2 · Precision · Recall / (Precision + Recall)               (2.7)

Average Precision (AP) tổng hợp đường cong Precision–Recall khi thay đổi ngưỡng confidence:

AP = ∫₀¹ P(r) dr                                                  (2.8)

Trong thực tế, AP được tính xấp xỉ bằng 101-point interpolation theo chuẩn COCO [9], [16]. Mean Average Precision (mAP) là trung bình AP của các lớp:

mAP = (1/C) · Σ_{c=1..C} AP_c                                    (2.9)

Đề tài báo cáo hai biến thể: **mAP@0.5** (AP trung bình ở ngưỡng IoU = 0.5) và **mAP@0.5:0.95** (AP trung bình ở 10 ngưỡng IoU từ 0.5 đến 0.95, bước 0.05 — chỉ số strict hơn). Mỗi lớp cũng báo cáo AP riêng để đánh giá ảnh hưởng của mất cân bằng lớp.

### 2.7.3 Chỉ số tốc độ và điều kiện đo

Tốc độ xử lý được đo bằng Frames Per Second (FPS) hoặc latency trên một khung hình. Phép đo cần phân biệt:
- **Model-only time**: chỉ bao gồm forward pass của mô hình trên GPU
- **End-to-end time**: bao gồm tiền xử lý (resize, normalize, HWC→CHW), suy luận, và hậu xử lý (NMS, chuyển định dạng output)

Để đo chính xác, cần: (i) warm-up với vài chục forward pass trước khi đo, (ii) đồng bộ GPU (`torch.cuda.synchronize()`) trước và sau khi đo, (iii) chạy ít nhất 100 khung hình và lấy trung bình, (iv) ghi rõ phần cứng, batch size, precision (FP32 hay FP16), kích thước ảnh đầu vào. So sánh FPS giữa các mô hình chỉ có ý nghĩa khi các điều kiện này tương thích.

### 2.7.4 Chỉ số đánh giá đếm

Với bài toán đếm phương tiện qua vạch ảo, các chỉ số được sử dụng:

MAE = (1/N) · Σ_{i=1..N} |N_pred,i − N_GT,i|                     (2.10)

MAPE = (1/N) · Σ_{i=1..N} |N_pred,i − N_GT,i| / N_GT,i × 100%    (2.11)

trong đó N là số đoạn video đánh giá, N_{pred,i} và N_{GT,i} lần lượt là số lượt phương tiện hệ thống đếm được và số lượt đếm tay (ground truth) ở đoạn thứ i. MAE phản ánh sai số tuyệt đối, MAPE phản ánh sai số tương đối. Đánh giá được thực hiện theo từng lớp phương tiện và theo điều kiện (ngày/đêm) nếu có đủ dữ liệu.

## 2.8 Công nghệ và công cụ đã sử dụng

Đề tài sử dụng các công nghệ và công cụ sau cho các giai đoạn khác nhau:

**Nền tảng lập trình**: Python 3.10.20 làm ngôn ngữ chính; PyTorch 2.5.1 với CUDA 12.4 cho tính toán GPU và huấn luyện mạng nơ-ron.

**Thư viện phát hiện đối tượng**: Ultralytics 8.4.142 cung cấp YOLOv8 kèm công cụ huấn luyện, đánh giá và theo vết (ByteTrack tích hợp sẵn). TorchVision 0.20.1 cung cấp Faster R-CNN ResNet50-FPN với trọng số COCO. torchmetrics 1.5 được dùng để tính mAP trên cùng tập test để so sánh công bằng hai mô hình.

**Xử lý ảnh và video**: OpenCV (cv2) 4.10 cho đọc/ghi video, xử lý khung hình, vẽ bounding box. Pillow 10.4 hỗ trợ đọc/ghi các định dạng ảnh. ffmpeg được dùng để transcode video sang H.264 cho hiển thị trên browser.

**Giao diện người dùng**: Streamlit 1.38 để xây dựng demo web cho phép người dùng tải video, cấu hình vạch đếm và xem kết quả trực tiếp.

**Theo dõi thí nghiệm và tái lập**: git kết quả mỗi commit được gắn với một cấu hình; split_manifest.json ghi lại chính xác video nào thuộc tập nào; seed cố định (42) cho tất cả các run. Kết quả training lưu vào `runs/detect/<run_name>/` gồm trọng số, results.csv và các biểu đồ chẩn đoán.

**Phần cứng**: GPU NVIDIA RTX 3500 Ada Generation Laptop (12 GB VRAM) cho huấn luyện và suy luận; CPU Intel Core i9-13900H; RAM 32 GB; OS Linux (WSL2 trên Windows 11).

## 2.9 Tổng kết chương

Chương này đã trình bày cơ sở lý thuyết cho đề tài. Hai mô hình được lựa chọn là YOLOv8s (one-stage, anchor-free, 11,1 triệu tham số) đóng vai trò mô hình chính và Faster R-CNN ResNet50-FPN (two-stage, anchor-based, ~41,8 triệu tham số) đóng vai trò mô hình đối chứng. Cả hai mô hình được fine-tune từ trọng số pretrained COCO với thay đổi lớp đầu ra cho 4 lớp phương tiện.

Về tiêu chí đánh giá, đề tài báo cáo mAP@0.5, mAP@0.5:0.95, AP từng lớp và tốc độ FPS cho phát hiện; MAE và MAPE cho đếm. Quy trình chống rò rỉ dữ liệu cấp video được xem là yêu cầu bắt buộc để các chỉ số đánh giá phản ánh đúng khả năng tổng quát hóa của mô hình.

Thiết kế dữ liệu cụ thể (các nguồn, cấu hình chia tập, chiến lược lấy mẫu) và triển khai huấn luyện được trình bày trong Chương 3.
