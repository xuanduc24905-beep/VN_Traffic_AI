# Audit leakage — Roboflow vietnam_vehicles v10 (standalone)

_Sinh tự động bởi [audit_v10_all.py](../scripts/data_prep/audit_v10/audit_v10_all.py) lúc 2026-11-14 13:09. Chỉ đọc dữ liệu, không sửa split._

## Kết luận

❌ **v10 LEAK** — train/valid/test gốc của v10 rò rỉ lẫn nhau (camera, xe, thời gian, khung hình gần trùng). **Không** được dùng valid/test của v10 để đánh giá, và **chưa** được coi là an toàn khi gộp vào V5+ cho tới khi xử lý các điểm ở mục 4.

**Bối cảnh V5+:** [build_v5plus_merged.py:3](../scripts/data_prep/build_v5plus_merged.py#L3) gộp **cả 3 split** của v10 vào **train** V5+, val/test giữ nguyên của V5. Do đó leak *nội bộ* v10 **không** làm sai số liệu eval của V5+ — nó chỉ làm valid/test gốc của v10 vô giá trị. Hai điều kiện còn lại trước khi coi V5+ là sạch:
1. **Chưa kiểm tra**: leak chéo v10 ↔ val/test V5 (cùng camera/khung hình với `vn_cantho`, `vnv3`).
2. **Rủi ro chất lượng nhãn**: trung bình chỉ ~1.3 box/ảnh trên ảnh CCTV nhiều xe, và các file cùng camera + cùng giây mang **bộ nhãn khác nhau** (vd `general_caunamprc554_50Y31618_13-02-2024_13-08-11` có 2 box, bản ở test chỉ 1 box) → nhiều xe trong ảnh không được gán nhãn, model bị dạy coi xe là nền. Cần kiểm tra trực quan một mẫu trước khi tin dùng.

_Ghi chú: docstring build_v5plus_merged.py ghi "~776 camera" — thực tế parse được 44 camera._

## 1. Tổng quan cấu trúc

- Nguồn: `data/raw_vietnam_vehicles_v10` (Roboflow `visionvn/vietnam_vehicles` v10, export 2024-09-05, CC BY 4.0, không augmentation)
- Ảnh: **4824** — train 3867 / valid 673 / test 284
- Class (6): container, oto, xe_ba_gac, xe_khach, xe_may, xe_tai
- Variant tiền tố filename: {'gen': 17, 'general': 4791, 'veh': 16} (+ 2 file dùng `general-` toàn dấu gạch ngang, 8 file có hậu tố `-1-`)
- Tỉ lệ parse: camera 100.0%, datetime 100.0%, vehicle_id 78.8%, biển số 100.0%
- **Camera unique: 44** (thuộc 33 địa điểm) — **không phải ~776 camera** như giả định ban đầu.
- vehicle_id unique: 1819 (ID trần) / 3654 (theo cặp camera+ID); biển số unique: 4081
- Khoảng thời gian: 2024-01-16 → 2024-09-04 (78 ngày khác nhau)

## 2. Verdict từng check

| Check | Ngưỡng | Kết quả | Verdict |
|---|---|---|---|
| (a) Exact duplicate (MD5) | 0 nhóm MD5 trùng cross-split, 0 source_name trùng cross-split | 0 nhóm cross-split, 0 within-split | ✅ PASS |
| (b) Camera leak | 0 camera_id xuất hiện ở >1 split (site-level chỉ WARN) | 38/44 camera ở >1 split; 100.0% ảnh test từ camera đã thấy | ❌ FAIL |
| (c) Vehicle leak | 0 (camera,vehicle_id,ngày) cross-split và 0 biển số cross-split | 3 track cross-split; 178 biển số cross-split (ảnh dính: valid 26.75%, test 24.3%) | ❌ FAIL |
| (d) Temporal leak | 0 cặp ảnh khác split, cùng camera + cùng ngày, cách nhau < 60s | 831 cặp; ảnh dính: valid 26.89%, test 30.99% | ❌ FAIL |
| (e) Near-duplicate pHash | 0 cặp cross-split có Hamming(pHash 64-bit) ≤ 5 | 203 cặp {'train-valid': 145, 'train-test': 51, 'valid-test': 7}; ảnh dính: valid 7.58%, test 7.04% | ❌ FAIL |
| (f) Label integrity | 0 lỗi format/class/bbox | 0 file lỗi | ✅ PASS |
| (f') Phân phối class | lệch ≤15pp giữa split | 0 class lệch | ✅ PASS |

## 3. Chi tiết

### (a) Exact duplicate

- 4824/4824 MD5 unique. Không có ảnh y hệt byte giữa các split.
- ⚠️ PASS ở đây **không** có nghĩa là không trùng: Roboflow re-encode ảnh nên cùng một khung hình có MD5 khác nhau → xem (d) và (e).

### (b) Camera leak

- Camera chỉ có ở 1 split: {'train': 6, 'valid': 0, 'test': 0}. Site-level: 30/33 địa điểm ở >1 split (⚠️ WARN).
- Split hiện tại là **random theo ảnh** (mặc định Roboflow) → hầu như mọi camera đều có mặt ở cả 3 split. Test set đo khả năng nhận diện trên **camera đã thấy**, không phải camera mới.
- Top camera (train/valid/test):
  - `xom7vinhtan554`: 607/118/62
  - `garagecuong554`: 348/72/37
  - `nga3phuocthe554`: 278/55/17
  - `tranhungdao59`: 273/41/19
  - `caunamprc554`: 237/65/18
  - `nga3duong554`: 183/51/18
  - `nga3doclap37`: 199/17/12
  - `caunamprc555`: 157/44/20

### (c) Vehicle leak

- Khoá (camera, vehicle_id, ngày): 3 track cross-split. Mẫu (cùng khung hình, chỉ khác tiền tố `general_`/`veh_`):
  - `train/general_hodabac554_ID130_container_model-_color-xanh_la_51R34695_bien_trang_23-03-2024_12-58-37` ↔ `valid/veh_hodabac554_ID130_container_model-_color-xanh_la_51R34695_bien_trang_23-03-2024_12-58-37`
  - `train/general_nga3phuocthe556_ID49_container_model-_color-xanh_duong_86R01125_bien_trang_28-03-2024_11-11-40` ↔ `valid/veh_nga3phuocthe556_ID49_container_model-_color-xanh_duong_86R01125_bien_trang_28-03-2024_11-11-40`
  - `train/veh_garagecuong554_ID997_container_color-nau_51C64047_bien_unknown_01-03-2024_21-58-12` ↔ `test/general_garagecuong554_ID997_container_color-nau_51C64047_bien_unknown_01-03-2024_21-58-12`
- Biển số: 178/4081 biển xuất hiện ở >1 split. Phần lớn là xe quay lại cùng camera ở ngày khác (vd xe buýt `60Y…` qua `caunamprc554` nhiều lần) → cùng ngoại hình xe ở train và test.
- vehicle_id trần (không kèm camera) **không** dùng làm verdict: đó là track-ID của tracker, reset theo camera.

### (d) Temporal leak

- 831 cặp khác split cùng camera, cùng ngày, |Δt|<60s. Theo cặp split: {'train-valid': 523, 'test-train': 272, 'test-valid': 36}.
- Phân bố Δt: {'0-1s': 47, '16-30s': 196, '31-59s': 362, '2-5s': 71, '6-15s': 155}. Các cặp **Δt = 0s** là cùng một khung hình CCTV được lưu thành nhiều file (mỗi xe trong khung một file, khác biển số/ID) rồi bị Roboflow chia vào split khác nhau.
- Mẫu:
  - Δt=0s: `train/general_caunamprc554_50Y31618_13-02-2024_13-08-11` ↔ `test/general_caunamprc554_60T31616_13-02-2024_13-08-11`
  - Δt=0s: `train/general_caunamprc554_60Y31618_05-02-2024_11-16-58` ↔ `valid/general_caunamprc554_60T31618_05-02-2024_11-16-58`
  - Δt=0s: `train/general_caunamprc555_60Y32573_02-02-2024_12-35-18` ↔ `valid/general_caunamprc555_60Y325735_02-02-2024_12-35-18`
  - Δt=0s: `train/general_garagecuong554_51R21336_09-02-2024_13-56-50` ↔ `valid/general_garagecuong554_51R21330_09-02-2024_13-56-50`

### (e) Near-duplicate (pHash)

- Histogram Hamming: {'0': 5, '2': 37, '4': 161}; 100.0% cặp cùng camera.
- Near-dup trong cùng split (tham khảo): {'train': 689, 'valid': 5, 'test': 1}.
- Mẫu Hamming=0 (ảnh giống hệt về thị giác nhưng khác file):
  - `train/general_hungvuong556_ID315_xe_may_model-_color-_61F133073_bien_trang_04-09-2024_10-40-41` ↔ `valid/general_hungvuong556_ID316_xe_may_model-_color-trang_54S82138_bien_trang_04-09-2024_10-40-42`
  - `train/general_xom7vinhtan554_86H93166_04-02-2024_15-59-10` ↔ `valid/general_xom7vinhtan554_85F100509_04-02-2024_15-59-10`
  - `train/general_tranquangdieu556_ID373_xe_may_model-_color-trang_61N58078_bien_trang_04-09-2024_10-20-47` ↔ `test/general_tranquangdieu556_ID348_xe_may_model-_color-trang_61P13796_bien_trang_04-09-2024_10-18-52`

### (f) Label integrity & phân phối class

| Split | Ảnh | Box | Box/ảnh | Ảnh rỗng | container | oto | xe_ba_gac | xe_khach | xe_may | xe_tai |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 3867 | 5049 | 1.31 | 1 | 7.82% | 4.56% | 7.66% | 4.52% | 54.96% | 20.48% |
| valid | 673 | 838 | 1.25 | 0 | 14.68% | 2.74% | 11.46% | 2.15% | 46.78% | 22.2% |
| test | 284 | 358 | 1.26 | 0 | 15.64% | 3.35% | 13.69% | 1.96% | 47.21% | 18.16% |

- Không có lỗi format/class_id/toạ độ (tái sử dụng `check_label_file` của [validate_labels.py:35](../scripts/data_prep/validate_labels.py#L35)).
- Không class nào lệch >15pp giữa split, nhưng container/xe_ba_gac ở valid/test gần **gấp đôi** train (~15% vs ~8%). Imbalance nội tại max/min = 13.2× (xe_may chiếm ~53%).

## 4. Cần fix gì trước khi gộp (ĐỀ XUẤT — chưa thực hiện)

Vì v10 trong V5+ chỉ đóng vai trò **dữ liệu train bổ sung**, có 2 hướng tuỳ mục đích:

**Hướng A — Chỉ dùng v10 để train (khuyến nghị, đơn giản nhất):**
1. Gộp **toàn bộ** train+valid+test của v10 vào train của V5+; **không** dùng valid/test của v10 để báo cáo số liệu. Leak nội bộ v10 khi đó vô hại vì không có split eval nào của v10.
2. Vẫn phải kiểm tra **leak chéo v10 ↔ V5 test/val**: chạy pHash + MD5 giữa ảnh v10 và val/test của V5 (V5 có `vn_cantho`, `vnv3` cũng là CCTV Việt Nam — có thể trùng camera).
3. Dedup khung hình trùng trong v10 (cùng camera + cùng giây / pHash≤2) để tránh over-weight; giữ một bản và **gộp nhãn** của các bản (mỗi file có thể chỉ gán nhãn một số xe trong khung).

**Hướng B — Cần v10 có valid/test riêng đáng tin (vd báo cáo mAP trên v10):**
1. Rebuild split theo **site_id** (group split, 33 địa điểm), không theo ảnh — cùng tinh thần [rebuild_v5_by_video.py](../scripts/data_prep/rebuild_v5_by_video.py). Lưu ý site lớn nhất chiếm 16.3% ảnh nên tỉ lệ 80/10/10 chỉ xấp xỉ; nên stratify theo class (container/xe_ba_gac/xe_may tập trung ở một số site).
   - Top site: `xom7vinhtan` 787 (16.3%), `caunamprc` 541 (11.2%), `garagecuong` 457 (9.5%), `nga3phuocthe` 368 (7.6%), `tranhungdao` 333 (6.9%), `nga3doclap` 330 (6.8%), `tranquangdieu` 300 (6.2%), `nga3duong` 263 (5.5%)
2. Group theo site (không phải camera_id) vì nhiều kênh (`554/555/556`) cùng một ngã ba nhìn chung cảnh.
3. Sau khi split theo site, kiểm tra lại leak biển số chéo site: hiện có 256 biển xuất hiện ở >1 site (768 ảnh) — chuyển các ảnh này về cùng phía hoặc loại khỏi eval.
4. Dedup như A.3 trước khi split.
5. Chạy lại `audit_v10_all.py` trên bản rebuild; chỉ gộp khi (b)(c)(d)(e) đều PASS.

**Không làm:** dùng valid/test hiện tại của v10 để chọn checkpoint hay báo cáo — số liệu sẽ bị thổi phồng giống V1-V4.

## 5. File output

- `reports/v10_survey.json` — survey
- `reports/v10_exact_duplicates.json` — (a) exact duplicate
- `reports/v10_camera_leak.json` — (b) camera leak
- `reports/v10_vehicle_leak.json` — (c) vehicle leak
- `reports/v10_temporal_leak.json` — (d) temporal leak
- `reports/v10_phash_near_duplicates.json` — (e) pHash near-dup
- `reports/v10_label_integrity.json` — (f) label integrity

Tái tạo: `conda activate yolov8_ft && python scripts/data_prep/audit_v10/audit_v10_all.py`
