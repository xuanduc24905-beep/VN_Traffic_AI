# Hướng dẫn chạy lại giao thức sạch và điền số vào bài

> **Trạng thái (2026-11-16):** đã train `full` seed 42 trên máy chính (117 phút) và điền đủ số vào `main.tex`.
> Tài liệu này chỉ cần nếu muốn chạy lại (ví dụ thêm seed) trên máy khác.

Bài `main.tex` đã viết theo **giao thức dữ liệu sạch**. Các ô `\TBD` (hiện ra `[?]` trong PDF) là
số liệu chưa có, vì chưa train trên bản sạch. **Không được tự bịa số** — chỉ điền từ kết quả chạy thật.

## 1. Yêu cầu máy
- GPU NVIDIA >= 12 GB VRAM (batch 16, ảnh 960), CUDA, đã cài `ultralytics` (repo từng chạy 8.4.126 và 8.4.142).
- Có sẵn `yolov8s.pt` ở thư mục gốc repo và dữ liệu `data/merged_v5plus_clean` (6.969 train / 1.000 val / 1.500 test)
  cùng `data/vn_vehicles_aerial`. Không copy kèm các tệp `*.npy` (bộ nhớ đệm, tự sinh lại).
- Thời gian ước tính: 2–5 giờ. Máy laptop nên đặt nơi thoáng mát, theo dõi nhiệt độ GPU.

## 2. Lệnh chạy (đúng thứ tự)
```bash
cd /đường/dẫn/vn_traffic_ai
python scripts/training/v5plus_clean_train.py --variant full --seed 42
python scripts/verify/63_eval_clean_full.py
python scripts/training/v5plus_clean_train.py --summarize
```
- Lệnh 1: train 60 epoch trên bản sạch, chọn checkpoint bằng val sạch (cam_07), rồi đánh giá **test V5 đúng một lần**.
  Chạy lại cùng lệnh sẽ tiếp tục (resume) nếu bị ngắt, hoặc bỏ qua nếu đã xong.
- Lệnh 2: đánh giá thêm VN Aerial val/test. Kết quả ở `outputs/verify/clean_full_eval.json`.
- Theo dõi tiến độ: `watch -n 30 "cut -d, -f1,8,9 runs/detect/train_v8s_v5pc_full_s42/results.csv"`
  (cột: epoch, mAP50, mAP50-95 trên validation).

## 3. Điền vào `main.tex`
Lấy số từ `outputs/verify/clean_full_eval.json` (và `reports/v5plus_clean/results/train_v8s_v5pc_full_s42.json`
cho test V5). Dùng dấu phẩy thập phân, 4 chữ số, ví dụ `0{,}7984`.

| Vị trí trong bài | Lấy từ |
|---|---|
| Bảng "YOLOv8s theo giao thức sạch" (4 dòng) | mAP50, mAP50_95, precision, recall của V5_test, VN_Aerial_val, VN_Aerial_test |
| Bảng "AP@0,5 theo lớp" (3 dòng) | `AP50_per_class` của từng bộ |
| Tóm tắt, Kết luận | mAP@0,5 của V5_test và VN_Aerial_test |
| Mục V-A: "hơn \TBD điểm" | (mAP50 sạch − 0,6408) × 100 |
| Mục V-A: "chênh lệch là \TBD điểm" so với 0,7930 | (mAP50 sạch − 0,7930) × 100, có thể âm |
| Thảo luận: "\TBD điểm với mô hình bản sạch" | cùng số với mục V-A (so với Faster R-CNN) |
| Mục V (đầu): máy huấn luyện | tên GPU, phiên bản Ultralytics/PyTorch của máy chạy |

Tìm các ô còn thiếu: `grep -n "TBD" docs/paper/main.tex`.

## 4. Sau khi điền số: đọc lại và sửa lời văn theo kết quả
- Nếu mô hình sạch **thấp hơn** 0,7930 hoặc 0,7984: giữ nguyên lời văn, đó là kết quả trung thực (bản sạch bỏ rò rỉ và giảm dữ liệu).
  Đừng xóa bảng cũ; chúng được gắn nhãn "bản chia cũ".
- Nếu mô hình sạch **cao hơn**: kiểm tra câu "gộp nhiều thay đổi cùng lúc" vẫn đúng, không viết là do dữ liệu sạch.
- Kiểm tra các câu có chữ "hơn", "tốt nhất", "vượt" còn khớp với số vừa điền.

## 5. Những điều chưa làm (đã ghi trong bài là hạn chế)
- Mới 1 hạt giống. Nếu còn thời gian: `--seed 42 0 1`, rồi báo trung bình ± độ lệch chuẩn (`--summarize`).
- Faster R-CNN chưa train lại trên bản sạch.
- Chưa có đối chứng `cls_pw=0` so với `0,5` trên bản sạch; trọng số lớp trong bài chỉ là khảo sát sơ bộ trên bản chia cũ.
- Kiểm tra camera dựa trên mã băm ảnh và ảnh nền, chưa là bằng chứng tuyệt đối.
