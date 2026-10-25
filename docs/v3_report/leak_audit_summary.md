# Leak audit — kết quả định lượng

**Ngày:** 2026-10-26
**Script:** [scripts/data_prep/audit_leak.py](../../scripts/data_prep/audit_leak.py)
**Raw output:** [leak_audit.json](leak_audit.json)

Chạy audit tự động trên `data/merged_v3` và `data/merged_v4` để xác minh phát hiện leak trong [DATA_LEAK_FINDINGS.md](DATA_LEAK_FINDINGS.md).

---

## 1. Số liệu tổng

| Dataset | Videos | Train frames | Val frames | Test frames | Val leak | Test leak |
|---|---|---|---|---|---|---|
| merged_v3 | 15 | 20,551 | 1,779 | 1,053 | **100%** | **100%** |
| merged_v4 | 17 | 49,746 | 1,779 | 1,053 | **100%** | **100%** |

V4 thêm 2 pseudo-video `vncctv_train` + `vncctv_test` (29,195 frame CCTV Đà Nẵng) nhưng **chỉ vào train** → không giúp val/test (vẫn giữ nguyên split V3).

→ Confirm giả thuyết: V4 ≤ V3 không phải vì CCTV data xấu, mà vì val set đã saturate do leak sẵn.

---

## 2. Chi tiết video (merged_v3)

**Train-only (không leak, dùng được):**
```
soictnight_cam_03   5961 frames   (đêm)
soictnight_cam_05   4876 frames   (đêm)
soictnight_cam_08   2997 frames   (đêm)
```

**Leaky (có mặt trong cả 3 split):**
```
vnv3                        train=1235 val=156  test=156   ← 1 video duy nhất
soict24_cam_01              train=950  val=276  test=126
soict24_cam_04              train=716  val=218  test=106
soict24_cam_02              train=702  val=198  test=100
soict24_cam_06              train=695  val=202  test=103
soict24_cam_07              train=681  val=213  test=106
soict24_cam_09              train=694  val=213  test=93
soict24_cam_10              train=366  val=87   test=47
cantho_CoopMart-8h10-8h30   train=302  val=95   test=85
cantho_camDLHB-13h20-13h57  train=247  val=82   test=87
cantho_CoopMart-8h-8h30     train=99   val=29   test=34
cantho_CoopMart-8h30-9h00   train=30   val=10   test=10
```

**Kết luận:** 12/15 video (= 80%) bị leak hoàn toàn. Toàn bộ val + test (2832 frames) đều từ video có trong train.

---

## 3. V5 split đề xuất (video-aware, no leak)

**Nguyên tắc:** mỗi video chỉ xuất hiện ở 1 split duy nhất.

### Train (11 videos → ~3-4k frames sau subsample 1/5)
| Video | Original | After subsample 1/5 |
|---|---|---|
| soictnight_cam_03 | 5961 | ~1192 |
| soictnight_cam_05 | 4876 | ~975 |
| soictnight_cam_08 | 2997 | ~599 |
| soict24_cam_01 | 950 | ~190 |
| soict24_cam_02 | 702 | ~140 |
| soict24_cam_04 | 716 | ~143 |
| soict24_cam_06 | 695 | ~139 |
| cantho_CoopMart-8h-8h30 | 99 | 99 (giữ nguyên) |
| cantho_CoopMart-8h10-8h30 | 302 | ~60 |
| vnv3 | 1235 | ~247 |
| vncctv (tùy chọn) | - | giữ nguyên role OOD |

**Tổng train effective: ~3.8k ảnh thật đa dạng** (vs 20.5k redundant hiện tại).

### Val (3 videos UNSEEN → ~1k frames)
- `soict24_cam_07` (day, 681 frames)
- `cantho_CoopMart-8h30-9h00` (day, 30 frames — nhỏ nhưng giữ)
- `cantho_camDLHB-13h20-13h57` (city, 247 frames)

### Test (2 videos UNSEEN → ~1k frames)
- `soict24_cam_09` (day, 694 frames)
- `soict24_cam_10` (day, 366 frames)

### OOD test (external, không eval chuẩn)
- `vn_cctv` Đà Nẵng (29k frames, 1 video) — chạy inference lấy số liệu generalization, không tính vào bảng mAP chính.

### Trade-offs của split này

**Pros:**
- Val/test từ camera KHÁC hoàn toàn với train → mAP đo thật khả năng generalize.
- Giữ được đa dạng ngày (soict24) + đêm (soictnight) + city (cantho) + VN domain (vnv3) trong train.
- OOD test (vn_cctv) cho thêm 1 chiều đánh giá robustness.

**Cons:**
- `vnv3` chỉ 1 video → train có đặc trưng vnv3 nhưng val không đo được loại cảnh này.
- Train đêm chiếm ~75% frame → có thể bias toward night features. **Giải pháp:** subsample night mạnh hơn (1/10) hoặc oversample day; cân nhắc khi retrain.
- Val chỉ ~1k frame → variance mAP sẽ cao hơn, cần report confidence interval hoặc stable epoch.

---

## 4. Dự đoán mAP honest

- Hiện tại (leaky): mAP50 ≈ 0.95
- Dự đoán V5 (honest, cùng model yolov8s): mAP50 ≈ **0.65-0.80**, mAP50-95 ≈ 0.40-0.55

Nếu đo trên vn_cctv OOD: dự đoán drop thêm 10-20 pt vì distribution khác hẳn (góc camera CCTV cao, low-res).

---

## 5. Next steps

1. ✅ Audit script + hard numbers (file này)
2. ⏭ Viết `scripts/data_prep/rebuild_v5_by_video.py` thực thi split trên
3. ⏭ Train `v5_honest.py` (yolov8s, 40ep, imgsz=640, fresh từ COCO)
4. ⏭ Eval V4(leaky) + V5(honest) trên cùng 2 bộ: `v5_test` + `vn_cctv_ood` → bảng 2×2
5. ⏭ Error analysis + viết lại section eval trong `final_report.ipynb`
