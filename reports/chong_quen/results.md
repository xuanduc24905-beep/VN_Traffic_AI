# So sánh các cách chống quên

Ngã tư: tỉ lệ bắt trúng xe yolov8x thấy (mốc thiên về COCO). SOICT, v10: nhãn người gán. Ấn Độ, camera VN: nhãn nháp yolov8x.

| Model | Ngã tư xe máy (0,25 / 0,1) | Ngã tư ô tô (0,25) | SOICT mAP50 | Xe máy / Xe tải SOICT | Test v10 | Ấn Độ ô tô | Camera VN | Hộp khổng lồ | mAP tập ngoài |
|---|---|---|---|---|---|---|---|---|---|
| coco_yolov8s | 0.940 / 0.995 | 0.770 | – | – / – | – | 0.557 | 0.727 | 7 | – |
| r_clspw_v2 | 0.443 / 0.568 | 0.519 | 0.799 | 0.862 / 0.691 | 0.989 | 0.274 | 0.581 | 25 | 0.369 |
| S1_freeze10 | 0.557 / 0.705 | 0.624 | 0.761 | 0.873 / 0.541 | 0.689 | 0.419 | 0.789 | 9 | 0.368 |
| S2_nofreeze | 0.475 / 0.694 | 0.652 | 0.790 | 0.844 / 0.709 | 0.773 | 0.355 | 0.758 | 17 | 0.431 |
| wise_S2_a0.3 | 0.760 / 0.896 | 0.693 | 0.767 | 0.888 / 0.641 | 0.728 | 0.498 | 0.809 | 21 | 0.367 |
| wise_S2_a0.5 | 0.716 / 0.869 | 0.721 | 0.801 | 0.888 / 0.719 | 0.753 | 0.523 | 0.807 | 52 | 0.388 |
| wise_S2_a0.7 | 0.628 / 0.792 | 0.673 | 0.802 | 0.875 / 0.721 | 0.769 | 0.468 | 0.794 | 50 | 0.389 |
| wise_S2_a0.85 | 0.563 / 0.743 | 0.662 | 0.797 | 0.863 / 0.707 | 0.769 | 0.426 | 0.771 | 44 | 0.398 |
| lowlr_S2 | 0.410 / 0.568 | 0.557 | 0.787 | 0.853 / 0.728 | 0.772 | 0.369 | 0.785 | 8 | 0.394 |
| l2sp_lam_mid | 0.683 / 0.863 | 0.754 | 0.800 | 0.877 / 0.667 | 0.736 | 0.446 | 0.785 | 7 | 0.445 |
| l2sp_lam_lo | 0.563 / 0.700 | 0.593 | 0.797 | 0.873 / 0.701 | 0.749 | 0.304 | 0.753 | 8 | 0.396 |
| l2sp_lam_hi | 0.869 / 0.984 | 0.732 | 0.762 | 0.842 / 0.557 | 0.688 | 0.431 | 0.753 | 6 | 0.416 |
| ewc | 0.514 / 0.716 | 0.627 | 0.782 | 0.868 / 0.638 | 0.766 | 0.371 | 0.758 | 8 | 0.391 |
| l2sp_mid_coco5k | 0.891 / 0.962 | 0.798 | 0.773 | 0.887 / 0.585 | 0.742 | 0.701 | 0.803 | 9 | 0.478 |
| t1_bigtruck | 0.940 / 0.984 | 0.762 | 0.769 | 0.883 / 0.599 | 0.754 | 0.666 | 0.816 | 9 | 0.505 |
| t2_truckx2 | 0.907 / 0.989 | 0.767 | 0.786 | 0.884 / 0.599 | 0.750 | 0.661 | 0.816 | 7 | 0.489 |
| t2_truckx2_s0 | 0.918 / 0.973 | 0.757 | 0.793 | 0.881 / 0.642 | 0.760 | 0.716 | 0.798 | 11 | 0.488 |
| t2_truckx2_s1 | 0.885 / 0.973 | 0.760 | 0.787 | 0.883 / 0.617 | 0.759 | 0.700 | 0.794 | 9 | 0.490 |

## Độ dao động theo hạt giống — t2_truckx2 (seed 42, 0, 1)

| Chỉ số | t2_truckx2 | t2_truckx2_s0 | t2_truckx2_s1 | Trung bình ± SD |
|---|---|---|---|---|
| Ngã tư xe máy 0,25 | 0.907 | 0.918 | 0.885 | 0.903 ± 0.017 |
| Ngã tư xe máy 0,1 | 0.989 | 0.973 | 0.973 | 0.978 ± 0.009 |
| Ngã tư ô tô | 0.767 | 0.757 | 0.760 | 0.761 ± 0.005 |
| SOICT mAP50 | 0.786 | 0.793 | 0.787 | 0.789 ± 0.004 |
| Xe máy SOICT | 0.884 | 0.881 | 0.883 | 0.883 ± 0.001 |
| Xe tải SOICT | 0.599 | 0.642 | 0.617 | 0.620 ± 0.022 |
| Test v10 | 0.750 | 0.760 | 0.759 | 0.756 ± 0.005 |
| Ấn Độ ô tô | 0.661 | 0.716 | 0.700 | 0.692 ± 0.028 |
| Camera VN | 0.816 | 0.798 | 0.794 | 0.803 ± 0.012 |
| Hộp khổng lồ | 7 | 11 | 9 | 9 ± 2 |
| mAP tập ngoài | 0.489 | 0.488 | 0.490 | 0.489 ± 0.001 |

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
