## Nhận xét (viết tay, cập nhật sau mỗi run)

Mọi số đo cùng `scripts/evaluation/eval_chong_quen.py`. best.pt chọn theo val (SOICT cam_07 + 4 điểm v10), không theo test.
"Ngã tư", "Ấn Độ", "Camera VN" chấm theo mốc/nhãn nháp yolov8x → thiên về model giống COCO; SOICT test và test v10 là nhãn người gán.

### Chất lượng nhãn giả yolov8x (trước khi train, 28/11)

`scripts/data_prep/audit_pseudo_labels.py` so nhãn giả với nhãn người gán sẵn có của chính các khung đó (`reports/v6split_pseudo/audit.json`):
- SOICT ngày, ngưỡng 0,6: yolov8x chỉ bắt **3%** xe máy người gán (8% được tô xám, còn ~89% thành "nền"), ô tô 37%; SOICT đêm gần như không bắt gì (xe máy 0,2%, ô tô 8%).
- Hộp yolov8x gán "bus" ở SOICT ngày: ~29% là xe tải thật (62/217 cặp trùng) → bản nhãn giả thuần có thể kéo AP xe tải/xe buýt xuống.
- Camera Đà Nẵng khá hơn (ô tô 77%, xe máy 37%, precision ô tô 0,95).
- Vì vậy chạy 2 bản trên cùng khung: **nhãn giả thuần** (`t2_pseudo`, theo kế hoạch) và **ghép** (`t2_ghep`: nhãn người gán gốc + bù hộp yolov8x không trùng — 691 ô tô, 138 xe buýt, 149 xe máy).

### T2 — 3 hạt giống (42, 0, 1), 29/11
- Độ dao động nhỏ ở các thước đo nhãn người gán: SOICT mAP50 0,789 ± 0,004, test v10 0,756 ± 0,005, xe máy SOICT 0,883 ± 0,001.
- Dao động lớn ở xe tải SOICT (0,620 ± 0,022), Ấn Độ ô tô (0,692 ± 0,028) và ngã tư xe máy ở conf 0,25 (0,903 ± 0,017) → chênh lệch < ~2 SD giữa các cấu hình ở những cột này (vd. T1 0,940 vs T2 0,907 ngã tư; xe tải 0,60 vs 0,64) KHÔNG đủ kết luận từ 1 seed.
- So với L2-SP + COCO 5k (1 seed): SOICT 0,773 vs 0,789 ± 0,004 (≈ 4 SD) → nhân đôi ảnh xe tải VN giúp SOICT là có thật; xe tải 0,585 vs 0,620 ± 0,022 chỉ ≈ 1,6 SD.
- Seed 42 (model đang dùng trong UI) không phải seed tốt nhất cũng không tệ nhất; từ đây so cấu hình mới với **trung bình 3 seed** của T2.

### T2 + ghép (nhãn người gán + bù yolov8x trên 4.050 khung VN mới), 1 seed, 29/11
So với T2 trung bình 3 seed:
- Tăng rõ (> 2 SD): ngã tư xe máy 0,956 (0,903 ± 0,017), ngã tư ô tô 0,777 (0,761 ± 0,005), xe tải SOICT 0,697 (0,620 ± 0,022), camera VN 0,834 (0,803 ± 0,012), mAP tập ngoài 0,538 (0,489 ± 0,001). SOICT mAP50 0,794 (0,789 ± 0,004) ngang.
- **Giảm rõ: xe máy SOICT 0,829 (0,883 ± 0,001)**; test v10 0,744 (0,756 ± 0,005) hơi giảm.
- Chẩn đoán trên 300 ảnh SOICT test (conf 0,25): hộp xe máy dự đoán vẫn cùng quy ước (cao bằng ~1,0 lần hộp người gán, IoU trung vị 0,73 vs 0,75) nhưng **sót nhiều hơn** (2.403 vs 1.851 / 8.023 xe máy không có hộp IoU ≥ 0,3) → model kém tự tin với xe máy nhỏ SOICT.
- Nghi nguồn: 2.000 khung Đà Nẵng (camera thấp, xe máy to; preview thấy nhiều xe máy đang chạy không có nhãn). Đang chạy `t2_ghep_nocctv` (bỏ khung Đà Nẵng) để tách nguyên nhân.
