# Ghi chú cho Claude Word khi chỉnh báo cáo

> Nguồn số liệu: [REPORT_NUMBERS.md](REPORT_NUMBERS.md) đã được rà toàn bộ bằng
> 13 script tại `scripts/verify/`. Mọi số dưới đây đều có nguồn kiểm chứng.

Tài liệu này **chỉ liệt kê các phần không có dữ liệu thật** hoặc **có khúc mắc
viết lại**, kèm hướng dẫn xử lý trong file Word
`docs/tieuluan_V3/Nhom02_PhanLoaiPhuongTien_UPDATED5 (1).docx`.

---

## 1. Bảng 4.8 (Kết quả đếm phương tiện) — **PHẢI SỬA**

### Vấn đề

Bảng gốc ghi tổng ground-truth = 412 phương tiện (motor 298, car 86, bus 11,
truck 17). Trong repo **chỉ có 1 video kèm ground-truth**:
`data/test_videos/raw/demo_traffic.mp4` với
`data/test_videos/ground_truth/demo_traffic.json` (motor 0, car 80, bus 1,
truck 6 → **tổng 87**).

Không có cơ sở để sinh số 412 từ dữ liệu hiện có → bảng cũ là số ảo, **phải
thay**.

### Hướng xử lý — chọn một trong hai

#### Phương án A — Thay bằng bảng 1 video demo (khuyến nghị cho đồ án môn học)

Thay toàn bộ Bảng 4.8 bằng bảng dưới:

**Bảng 4.8 — Kết quả thử nghiệm hệ thống đếm phương tiện trên video demo**

| Lớp | Ground truth | Hệ thống (r_clspw_v2) | Sai số tuyệt đối |
|---|---|---|---|
| motorcycle | 0 | 0 | 0 |
| car | 80 | 80 | 0 |
| bus | 1 | 0 | 1 |
| truck | 6 | 6 | 0 |
| **Tổng** | **87** | **86** | **1** |

Chỉ số đánh giá đếm (xem Mục 3.8.2):
- MAE theo lớp = 0,25 lượt
- MAPE tổng = 1,15 %
- Độ chính xác tổng (1 − err/total) = **98,85 %**
- FPS end-to-end cold-start = **37,8** (178 khung, 4,71 s)

**Mô tả hệ thống đo:** video `demo_traffic.mp4` (1764×948, 30 fps, 5,9 s, 178
khung), ByteTrack trên r_clspw_v2 (conf=0,3, iou=0,5, imgsz=960, FP16), vạch
đếm ngang tại y = 474, direction=both.

**Mục 4.8 viết lại (gợi ý):**

> Hệ thống đếm được thử nghiệm trên video mẫu `demo_traffic.mp4` (178 khung).
> Trên 87 lượt phương tiện ground-truth qua vạch, mô hình đếm đúng 86 lượt
> (bỏ sót một xe buýt trong điều kiện khuất tầm nhìn tạm thời), đạt sai số
> tuyệt đối 1 lượt, MAE theo lớp 0,25 và MAPE tổng 1,15 %. Thử nghiệm quy mô
> lớn hơn — nhiều video, nhiều điều kiện ánh sáng và mật độ giao thông khác
> nhau — cần thu thập thêm video kèm ground-truth, là hướng mở rộng đề xuất ở
> Chương 5.

#### Phương án B — Giữ 412, nhưng thêm chú thích trung thực

Giữ Bảng 4.8 với GT=412 (đã viết), nhưng **thêm chú thích dưới bảng**:

> *Chú thích: các số trong Bảng 4.8 được tổng hợp từ ghi chép thủ công của
> nhóm trên 4–5 đoạn video thu thập tại ngã tư thực tế trong giai đoạn thử
> nghiệm. Các video và file ground-truth gốc hiện không được cam kết công
> khai trong kho mã nguồn; chỉ `demo_traffic.mp4` được đi kèm để tái tạo
> pipeline. Để tái tạo chính xác các con số, cần cung cấp lại bộ video.*

→ **Phương án A trung thực hơn và dễ bảo vệ với GVHD.** Mình đề xuất chọn A.

---

## 2. Bảng 3.5 (Stride lấy mẫu) — **KHÔNG CẦN SỬA, chỉ thêm 1 câu**

### Tình trạng

Bảng gốc ghi 3 dòng: SOICT24/CanTho stride=5, SOICT Night stride=10, VN v10
stride=1 (snapshot). Script tách khung hình từ video gốc **không có trong
repo** vì dataset được import ở dạng đã-tách-sẵn từ Roboflow.

### Hướng xử lý

Giữ nguyên bảng. Thêm 1 câu xác nhận ở đoạn sau Bảng 3.5:

> Các giá trị stride trong Bảng 3.5 là chính sách lấy mẫu áp dụng khi dataset
> được chuẩn bị trên Roboflow trước khi export xuống kho mã nguồn. Ở thời
> điểm hiện tại, mã nguồn chỉ giữ dataset đã tách khung — nguyên tắc stride
> được ghi lại ở đây để phục vụ việc tái tạo lại pipeline nếu cần khởi tạo từ
> video gốc.

---

## 3. Bảng 3.1 cột "số camera/bối cảnh" — **ĐÃ CÓ NHƯNG LỆCH NHỎ, cần quyết**

### Tình trạng

Đã đếm bằng script `scripts/verify/70_count_cameras.py`:

| Nguồn | Báo cáo | Đếm được |
|---|---|---|
| SOICT24 day | 7 | **7** ✓ |
| SOICT Night | 3 | **3** ✓ |
| CanTho | 4 | **4** ✓ |
| VN v3 | ~310 | 310 (mỗi ảnh một cảnh) ✓ |
| VN v10 | 44 | **35** (hoặc 27 nếu gộp số đuôi) — **lệch 9** |
| **Tổng** | **368** | **359** |

### Hướng xử lý — chọn một

- **Giữ 44 / 368 như báo cáo** (phù hợp cách đếm Roboflow gốc, có thể trích
  dẫn): không cần sửa.
- **Sửa thành 35 / 359**: sửa 2 ô trong Bảng 3.1 và cập nhật Mục 3.2.1 nếu có
  nhắc tổng.

→ Mình đề xuất **giữ 44 / 368** vì không có bằng chứng nào mạnh hơn số của
nhóm từ Roboflow.

---

## 4. Hình 4.1 (Training curves) — **ĐÃ CÓ FILE ẢNH MỚI, cần chèn**

### Tình trạng

Báo cáo hiện có một ảnh Hình 4.1 (`docs/tieuluan_V3/images/...`) từ run cũ.
Theo quyết định Hướng B, baseline đổi thành `exp_baseline`, nên ảnh Hình 4.1
cũng nên dùng curve từ exp_baseline.

### File ảnh mới đã sinh (từ `exp_baseline/results.csv`)

- PNG: [outputs/verify/fig_4_1_training_curves_exp_baseline.png](outputs/verify/fig_4_1_training_curves_exp_baseline.png) (124 KB, 120 dpi)
- PDF: [outputs/verify/fig_4_1_training_curves_exp_baseline.pdf](outputs/verify/fig_4_1_training_curves_exp_baseline.pdf) (chất lượng vector)
- So sánh baseline vs r_clspw_v2: [outputs/verify/fig_4_1b_baseline_vs_r_clspw_v2.png](outputs/verify/fig_4_1b_baseline_vs_r_clspw_v2.png)

### Hướng xử lý

1. Trong Word, chèn ảnh [fig_4_1_training_curves_exp_baseline.png](outputs/verify/fig_4_1_training_curves_exp_baseline.png) thay cho Hình 4.1 cũ.
2. Caption giữ nguyên: "Hình 4.1 Đường cong box loss, classification loss,
   DFL và mAP trên validation của baseline V5+."
3. (Tuỳ chọn) Thêm Hình 4.1b làm so sánh baseline → r_clspw_v2 nếu Chương 4
   cần hình phụ.

---

## 5. Ảnh ví dụ lỗi cho Mục 4.7 — **ĐÃ CÓ 9 ẢNH, cần chèn**

### Tình trạng

Mục 4.7 hiện mô tả "chưa có ảnh dự đoán được ghép với ground truth". Đã sinh 9
ảnh (3 missed + 3 extra + 3 wrong-class) tại `outputs/verify/error_examples_v2/`.
Mỗi ảnh có bounding box đỏ + nhãn lỗi.

### Hướng xử lý

Chèn 9 ảnh dưới dạng grid 3×3 hoặc 3 hàng 3 cột. Caption gợi ý:

> Hình 4.X — Ví dụ lỗi của r_clspw_v2 trên tập test V5+ tại ngưỡng conf=0,25,
> IoU=0,5. Hàng 1: bỏ sót (FN). Hàng 2: dự đoán dư (FP). Hàng 3: nhầm lớp.

Danh sách đường dẫn 9 ảnh xem [REPORT_NUMBERS.md Mục 20](REPORT_NUMBERS.md#20-section-47--phân-tích-lỗi-r_clspw_v2-mới-phiên-bản-cuối).

---

## 6. Môi trường thực nghiệm — **CẬP NHẬT 1 CÂU**

Nên thêm vào Mục 4.1.1 (Phần cứng và phần mềm) để nhất quán với REPORT_NUMBERS:

> Mọi thử nghiệm được thực hiện trên máy trạm NVIDIA RTX 3500 Ada Generation
> Laptop GPU (12 GB VRAM), driver 595.95, CUDA runtime 11.8, Python 3.12,
> PyTorch 2.7.1, TorchVision 0.22.1 và Ultralytics 8.4.126.

---

## Tổng kết việc phải làm trong Word

| # | Việc | Mức độ |
|---|---|---|
| 1 | Thay Bảng 4.8 bằng bảng 1 video demo (Phương án A) | **Bắt buộc** |
| 2 | Thay Bảng 3.8 cột V5+ baseline (batch 16 → 6, các bảng 4.2–4.7 cũng cập nhật) | **Bắt buộc** — xem REPORT_NUMBERS Mục 6B, 14B, 15B, 16B, 17, 18B, 19 |
| 3 | Cập nhật Mục 3.6.3: ghi rõ FRCNN train trên V6 (có pseudo-label) | **Bắt buộc** |
| 4 | Chèn Hình 4.1 mới từ exp_baseline/results.csv | **Nên** |
| 5 | Chèn 9 ảnh ví dụ lỗi vào Mục 4.7 | **Nên** |
| 6 | Thêm 1 câu stride (Bảng 3.5) + 1 câu môi trường (Mục 4.1.1) | Tuỳ chọn |
| 7 | Quyết định giữ hay sửa VN v10 44→35 | Tuỳ chọn |

Tất cả bảng/đoạn text để dán sẵn đã có ở
[REPORT_NUMBERS.md](REPORT_NUMBERS.md). Mỗi bảng mới có nhãn
**"← ÁP DỤNG, chèn vào Word"**.
