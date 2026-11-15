
### 4.3.5 Đánh giá cross-domain trên bộ Vietnam Vehicles aerial

Để đánh giá khả năng tổng quát hóa qua các góc nhìn camera khác nhau, nhóm thu thập thêm bộ **Vietnam Vehicles** (Ultralytics Platform, 635 ảnh test+val) chụp từ camera aerial (nhìn từ trên xuống), hoàn toàn khác góc CCTV ngang của V5+. Bộ này đã được kiểm tra MD5 hash xác nhận 0 ảnh trùng với V5+.

**Bảng 4.11** Đánh giá cross-domain: mAP@0.5 của các mô hình trên 2 bộ test độc lập.

| Mô hình | V5+ test (CCTV) | VN Aerial val | VN Aerial test | Gap CCTV→Aerial |
|---|---|---|---|---|
| YOLOv8s-COCO (zero-shot) | 0.092 | — | — | — |
| V5 honest (640) | 0.665 | — | — | — |
| V5+ 960px | 0.782 | — | — | — |
| p4_scale (cuối) | 0.787 | — | — | — |
| r_p2 (ablation) | 0.753 | — | — | — |
| r_clspw (ablation) | 0.793 | — | — | — |
| r_progressive (ablation) | 0.771 | — | — | — |

**Nhận xét**: Gap CCTV→Aerial đo mức suy giảm khi mô hình phải xử lý góc nhìn hoàn toàn mới. Mô hình nào có gap nhỏ nhất → generalize tốt nhất cross-view.

**Hình 4.11** Biểu đồ so sánh mAP@0.5 trên 2 bộ test.



### 4.4.5 Ablation về kiến trúc: YOLOv8s-P2 vs YOLOv8s mặc định

Thí nghiệm này tách riêng đóng góp của **kiến trúc** khỏi dataset và hyperparameter. Giữ nguyên tập dữ liệu V5+, giữ nguyên imgsz=960, chỉ thay đổi kiến trúc detection head:

| Mô hình | Kiến trúc | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| p4_scale | YOLOv8s mặc định (3 heads: stride 8/16/32) | 0.787 | 0.423 |
| **r_p2** | **YOLOv8s-P2 (4 heads: stride 4/8/16/32)** | 0.753 | 0.395 |
| Δ | | -0.034 | -0.029 |

**Kết luận**: YOLOv8s-P2 KÉM YOLOv8s mặc định về mAP@0.5 (-0.034 điểm). Không ủng hộ giả thuyết thêm detection head stride=4 giúp phát hiện vật nhỏ tốt hơn trong cảnh giao thông VN dày đặc xe máy. Mặc dù giả thuyết lý thuyết hợp lý, trong thực nghiệm P2 head không bù đắp được overhead tính toán trên tập V5+ hiện tại.
