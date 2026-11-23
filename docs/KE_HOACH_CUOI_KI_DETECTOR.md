# Kế hoạch cải thiện detector — project cuối kì

Cập nhật: 23/11/2026. Nhánh làm việc: `xuanduc/v5fix-detector`. Bài NCKH (SSRC) giữ hướng cũ (V5+, `r_clspw_v2`); tài liệu này chỉ cho project cuối kì.

## 1. Vấn đề

Model fine-tune từ YOLOv8s COCO bắt xe **kém hơn chính YOLOv8s COCO gốc** khi gặp camera khác kiểu dữ liệu train. Mục tiêu: giữ YOLOv8s (để triển khai được trên camera) nhưng không "quên" kiến thức COCO, và bắt tốt hơn ở camera VN.

## 2. Đã làm

| Bước | Nội dung | Script / kết quả |
|---|---|---|
| Sửa dữ liệu | Bỏ Cần Thơ; bù nhãn ô tô/xe buýt (yolov8x, conf ≥ 0,5) và che xám xe thiếu nhãn ở SOICT, SOICT đêm, VNv3; lấy lại 598 ảnh xe tải v10 | `scripts/data_prep/build_v6final.py`, `reports/v6final/` |
| Thêm dữ liệu | Giữ 3.000 ảnh MIO-TCD cũ, thêm 4.500 ảnh mới (ô tô nhỏ, bán tải/van, xe tải, xe buýt) | `build_v6final_mio.py`, `reports/v6final_mio/` |
| Chia lại val/test | Chia theo camera/điểm quay. Val = SOICT cam_07 + 4 điểm v10; test A = SOICT cam_09/10; test phụ = 5 điểm v10. Video Đà Nẵng đã thử rồi bỏ (camera thấp, ngoài phạm vi, nhãn thiếu nhiều) | `build_v6split.py`, `reports/v6split/` |
| Train | S1 = freeze 10 lớp, S2 = không freeze; 30 epoch, imgsz 960, cùng dữ liệu V6split | `scripts/training/v6final_train.py --run s1/s2` |
| Đánh giá | SOICT test, test phụ v10, tập video ngoài (nhãn NHÁP yolov8x) | `scripts/evaluation/eval_v5fix.py`, `reports/v5fix/results.json` |
| UI | Thêm S1/S2 vào Streamlit; video test cho nhóm ở Release `test-videos-v1` | `ui/streamlit_app.py` |

## 3. Kết quả hiện tại

| Model | SOICT mAP50 | Xe tải SOICT | Test v10 | Ấn Độ recall ô tô | Camera VN recall | Ngã tư recall xe máy (conf 0,25 / 0,1) | Hộp khổng lồ |
|---|---|---|---|---|---|---|---|
| yolov8s COCO | – | – | – | **0,56** | 0,73 | **0,95 / 1,00** | 7 |
| r_clspw_v2 (NCKH) | **0,799** | 0,69 | 0,99 (rò rỉ) | 0,27 | 0,58 | 0,39 / 0,57 | 25 |
| S1 (freeze) | 0,767 | 0,55 | 0,69 | 0,42 | **0,79** | 0,57 / 0,70 | 9 |
| S2 (không freeze) | 0,790 | **0,71** | **0,77** | 0,36 | 0,76 | 0,46 / 0,66 | 17 |

Đọc kết quả:
- Model fine-tune vẫn **thua COCO xa** ở cảnh lạ (ngã tư đông, quay chéo gần) → model đang quên kiến thức gốc.
- S1 bắt được nhiều xe hơn ở cả ba nguồn ngoài; S2 vẽ hộp/phân lớp đúng hơn (mAP ngoài 0,431 vs 0,385, xe tải). Mới 1 hạt giống, chênh 0,03–0,06 chưa đủ kết luận.
- Hạ conf xuống 0,1 thu hẹp khoảng cách S1–S2 → một phần lợi thế của S1 là độ tự tin, không hẳn khả năng thấy xe.
- Dữ liệu ảnh hưởng nhiều hơn freeze: từ r_clspw_v2 sang S1/S2, recall ô tô Ấn Độ 0,27 → 0,36–0,42.

## 4. Các bước tiếp theo

Thứ tự theo hiệu quả trên công sức. Mỗi lần train ≈ 1,5 giờ (RTX 12 GB).

### Bước 1 — Trộn trọng số (WiSE-FT), không cần train
- Trộn backbone + neck: `w = α·w_finetune + (1 − α)·w_COCO`, α ∈ {0,3; 0,5; 0,7}; head 4 lớp giữ của bản fine-tune.
- Áp cho S1 và S2. Đo: recall xe máy video ngã tư, recall ô tô Ấn Độ, SOICT test (xem có tụt không).
- Công: ~1 giờ.

### Bước 2 — L2-SP ("freeze mềm")
- Loss = loss detection + λ·Σ(w − w_COCO)² trên backbone + neck.
- Viết lớp con `DetectionTrainer` (Ultralytics 8.4.142, ghim phiên bản); lưu bản sao trọng số COCO; cộng phạt trong hàm loss; xử lý đúng thang loss (nhân batch size), AMP, EMA, resume.
- Cấu hình như S2, dò λ ∈ {1e-4, 1e-3, 1e-2} → 3 lần train.
- Ý nghĩa: S2 = λ 0, S1 = λ ∞ trên 10 lớp đầu, L2-SP nằm giữa → trả lời trực tiếp "freeze hay không freeze tổng quát hơn".
- Công: 1–2 giờ code + ~4,5 giờ train.

### Bước 3 — Học lại dữ liệu cũ (rehearsal) + dữ liệu đúng kiểu camera
- Trộn ~3.000 ảnh COCO có xe cộ (đổi nhãn sang 4 lớp) để chống quên; `lr0` 0,002; giữ `freeze=10` hoặc dùng λ tốt nhất của bước 2.
- Thêm vài trăm khung ngã tư VN quay chéo từ trên (nhiều ngã tư khác nhau), gán nhãn bằng yolov8x rồi duyệt tay.
- Ghi rõ tỉ lệ trộn COCO trong báo cáo (phương pháp chuẩn: Li & Hoiem 2016; Kirkpatrick và cộng sự 2017; Wortsman và cộng sự 2022).

### Bước 4 — EWC (chỉ làm nếu L2-SP có tác dụng)
- Giống L2-SP nhưng mỗi trọng số có hệ số theo Fisher information tính trên COCO val2017 (~5.000 ảnh, ~1 GB).
- Công: 3–4 giờ code + tính Fisher, rồi train như bước 2.

### Bước 5 — Thước đo đáng tin (làm song song, cần người)
- Sửa tay nhãn tập ngoài: 160 khung Ấn Độ + 90 khung camera VN (`data/ext_benchmark`, hiện là nhãn nháp yolov8x → thiên về COCO).
- Đếm tay số xe qua vạch trên 2–3 video ngã tư → so số đếm của từng model (đúng mục đích dự án).
- Duyệt 4 bảng bù nhãn `reports/v6final/duyet_bu_nhan_1..4.jpg`.
- Train thêm 1–2 hạt giống cho cấu hình thắng để xác nhận chênh lệch.

## 5. Tiêu chí chọn model triển khai

- Ưu tiên: recall trên video ngã tư/camera VN ngoài (nhãn đã sửa tay) và sai số đếm xe.
- Ràng buộc: SOICT test không tụt quá ~2 điểm so với S2; xe tải AP50 ≥ 0,65.
- Giữ YOLOv8s (cùng tốc độ cho mọi phương án trên).

## 6. Lưu ý

- MIO-TCD là CC BY-NC-SA 4.0 → chỉ dùng phi thương mại.
- Video trong Release `test-videos-v1` chỉ để test nội bộ; hai clip iStock có bản quyền Getty Images, không dùng để train.
- v10 có khung/chữ vẽ sẵn trong 95% ảnh → test phụ v10 chỉ để tham khảo.
