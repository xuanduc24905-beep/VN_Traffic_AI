## Nhận xét (viết tay, cập nhật sau mỗi run)

Mọi số đo cùng `scripts/evaluation/eval_chong_quen.py`. best.pt chọn theo val (SOICT cam_07 + 4 điểm v10), không theo test.
"Ngã tư", "Ấn Độ", "Camera VN" chấm theo mốc/nhãn nháp yolov8x → thiên về model giống COCO; SOICT test và test v10 là nhãn người gán.

### Chất lượng nhãn giả yolov8x (trước khi train, 28/11)

`scripts/data_prep/audit_pseudo_labels.py` so nhãn giả với nhãn người gán sẵn có của chính các khung đó (`reports/v6split_pseudo/audit.json`):
- SOICT ngày, ngưỡng 0,6: yolov8x chỉ bắt **3%** xe máy người gán (8% được tô xám, còn ~89% thành "nền"), ô tô 37%; SOICT đêm gần như không bắt gì (xe máy 0,2%, ô tô 8%).
- Hộp yolov8x gán "bus" ở SOICT ngày: ~29% là xe tải thật (62/217 cặp trùng) → bản nhãn giả thuần có thể kéo AP xe tải/xe buýt xuống.
- Camera Đà Nẵng khá hơn (ô tô 77%, xe máy 37%, precision ô tô 0,95).
- Vì vậy chạy 2 bản trên cùng khung: **nhãn giả thuần** (`t2_pseudo`, theo kế hoạch) và **ghép** (`t2_ghep`: nhãn người gán gốc + bù hộp yolov8x không trùng — 691 ô tô, 138 xe buýt, 149 xe máy).
