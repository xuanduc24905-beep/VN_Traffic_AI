# Audit leakage — Roboflow vietnam_vehicles v10 (standalone)

_Sinh tự động bởi [audit_v10_all.py](../scripts/data_prep/audit_v10/audit_v10_all.py) lúc 2026-11-14 13:35. Chỉ đọc dữ liệu, không sửa split._

## Kết luận

| Câu hỏi | Trả lời |
|---|---|
| v10 standalone có sạch (split nội bộ)? | ❌ **Không** — train/valid/test gốc rò rỉ lẫn nhau (camera, xe, thời gian, khung gần trùng). Không được dùng valid/test của v10 để đánh giá. |
| Gộp v10 vào **train** V5+ (val/test giữ của V5) có làm leak eval V5+? | ✅ **Không** — 0 ảnh/khung/camera của v10 trùng với val/test V5 (check (g)). |
| Nhãn có hợp lệ? | ✅ Format hợp lệ; ✅ độ đầy đủ ngang V5 (~2.77% vs 2.73% xe chưa gán, check (h)) |
| Ảnh có overlay vẽ sẵn? | ❌ **83.3%** ảnh mẫu (CI95 72.0–90.7%) có tag/khung của hệ thống tracking gốc — check (h) |

**→ Về leakage, v10 an toàn để gộp vào train V5+ theo đúng cách [build_v5plus_merged.py:3](../scripts/data_prep/build_v5plus_merged.py#L3) đang làm** (gộp cả 3 split v10 vào train, val/test giữ của V5): leak nội bộ v10 không chạm tới eval V5+. **Không** an toàn nếu dùng split gốc của v10 làm val/test.

**Còn lại — chất lượng dữ liệu (không phải leak):**
- Giả thuyết "thiếu nhãn nhiều" (do chỉ ~1.3 box/ảnh) **không được xác nhận**: theo detector COCO, tỉ lệ xe chưa gán nhãn của v10 ≈ V5 (~2.77%, cận dưới). Ảnh v10 vốn ít xe (snapshot vi phạm, tập trung 1–3 xe).
- **Overlay vẽ sẵn lên pixel ở ~83.3% ảnh**: tag ID/biển số/màu đặt ngay góc box xe được theo dõi → nguy cơ model học lối tắt "tag màu = xe". Không làm sai số liệu trên test V5 (ảnh V5 không có overlay) nhưng có thể làm giảm lợi ích của v10. Nên ablation (mục 4).

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
| (g) Chéo v10 ↔ val/test V5 | MD5=0, pHash≤5=0, NCC nền camera <0.90 và < min đối chứng | MD5 0, pHash 0 (min d=10), NCC max 0.5001 | ✅ PASS |
| (h) Thiếu nhãn (detector COCO) | % xe chưa gán v10 − V5 ≤ 5pp | v10 2.77% vs V5 2.73% | ✅ PASS |
| (h') Overlay vẽ sẵn (review tay 60 ảnh) | cận dưới CI95 ≤ 5% | 50/60 = 83.3% (CI95 [72.0, 90.7]) | ❌ FAIL |

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

### (g) Leak chéo v10 ↔ val/test V5

- So 4824 ảnh v10 (cả 3 split, vì đều vào train V5+) với 2966 ảnh val/test V5: {'cantho_CoopMart-8h30-9h00': 50, 'cantho_camDLHB-13h20---13h57': 416, 'soict24_cam_07': 1000, 'soict24_cam_09': 1000, 'soict24_cam_10': 500}.
- MD5 trùng: 0. pHash: 0 cặp ≤5, cặp gần nhất d=10 (và là cảnh khác hẳn khi xem bằng mắt).
- Ảnh nền camera (trung bình xám ≤150 ảnh/camera, so NCC):
  - Đối chứng dương — cùng camera chia 2 nửa: trung vị 0.9651, nhỏ nhất 0.7669
  - Baseline v10 khác địa điểm: trung vị 0.0906, max 0.7172; cùng địa điểm khác kênh: max 0.5402
  - **Chéo v10 ↔ V5 eval: max 0.5001** — thấp hơn hẳn đối chứng cùng camera (0.7669), nằm trong vùng "khác camera".
  - Top cặp: `hodabac554`↔`soict24_cam_09` 0.5001, `ngabathomo554`↔`soict24_cam_09` 0.4587, `hodabac554`↔`soict24_cam_07` 0.453
- Sanity: val/test của `merged_v5plus` trùng khớp V5 và không chứa file `v10_*`: ✅.
- Preview kiểm tra bằng mắt: [reports/v10_cross_v5_preview.jpg](../reports/v10_cross_v5_preview.jpg) (trái: v10, phải: V5).
- Giới hạn: không bắt được trường hợp cùng địa điểm nhưng góc quay khác hẳn (NCC nền thấp). Với 44 camera v10 (tên địa điểm VN) và V5 eval (SOICT24 + Cần Thơ) thì khả năng này thấp.

### (h) Độ đầy đủ nhãn & overlay vẽ sẵn

- Detector: yolov8m.pt (COCO, chỉ inference), conf≥0.5, lớp ['car', 'motorcycle', 'bus', 'truck']; xe "chưa gán" = IoU<0.3 với mọi GT, không nằm trong GT (IoA<0.6), diện tích ≥0.2% ảnh.

| | v10 | V5 val/test (đối chứng) |
|---|---|---|
| Ảnh | 4824 | 2966 |
| Box GT/ảnh | 1.29 | 20.72 |
| % ảnh có ≥1 xe chưa gán | 3.32% | 37.26% |
| **Ước lượng % xe chưa gán** | **2.77%** | **2.73%** |

- % ảnh cao hơn ở V5 chỉ vì V5 có nhiều xe/ảnh (cơ hội sót cao hơn); so theo % xe thì hai bộ ngang nhau. Detector COCO bỏ sót nhiều xe máy nhìn từ trên cao → đây là **cận dưới** cho cả hai.
- Preview (xanh lá = GT, đỏ = xe chưa gán): [reports/v10_missing_labels_preview.jpg](../reports/v10_missing_labels_preview.jpg).
- Overlay: review bằng mắt 60 ảnh ngẫu nhiên (seed 2024) → 50 ảnh có overlay (83.3%, CI95 72.0–90.7% ≈ 3472–4375 ảnh toàn bộ v10). Sheet: `reports/v10_overlay_review_sheet_{0,1,2}.jpg`, kết quả: `reports/v10_overlay_manual_review.json`.
  - Tag gắn đúng vị trí xe được theo dõi (góc trên-trái bbox) → trùng vị trí box GT: model có thể học lối tắt 'tag màu = xe'.
  - Overlay chỉ gắn cho một số xe trong khung (vd #31, #32: 4-5 xe, chỉ 2 tag). Giả thuyết 'GT cũng chỉ phủ xe có tag' KHÔNG được xác nhận bởi check detector (% xe chưa gán của v10 ≈ V5).
  - Đa giác ROI xanh dương (#19, #23, #41, #44) và tag ID xanh xuất hiện ở camera 2024-09-04 (Nguyen Van Troi, Nga 3 Doc Lap, Tran Quang Dieu...).
- Heuristic màu tự động đã thử và bị loại: chỉ bắt <3% ảnh v10, dương tính giả 5–8% trên V5.

## 4. Cần fix gì trước khi gộp (ĐỀ XUẤT — chưa thực hiện)

Vì v10 trong V5+ chỉ đóng vai trò **dữ liệu train bổ sung**, có 2 hướng tuỳ mục đích:

**Hướng A — Chỉ dùng v10 để train (khuyến nghị, đơn giản nhất):**
1. Gộp **toàn bộ** train+valid+test của v10 vào train của V5+; **không** dùng valid/test của v10 để báo cáo số liệu. Leak nội bộ v10 khi đó vô hại vì không có split eval nào của v10.
2. Leak chéo v10 ↔ val/test V5: ✅ đã kiểm tra, PASS (check (g)).
3. (Tuỳ chọn) Dedup khung hình gần trùng trong v10 (pHash≤2 cùng camera) để tránh over-weight một số cảnh; không ảnh hưởng tính trung thực của eval.
4. **Ablation overlay** (đáng làm nhất): train thêm 1 run V5+ bỏ (hoặc che) các ảnh v10 có overlay, so mAP trên test V5 với V5+ hiện tại. Nếu ngang/hơn → bỏ ảnh overlay; nếu kém hơn → overlay không gây hại đáng kể, giữ nguyên.

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
- `reports/v10_cross_v5_leak.json` — (g) leak chéo v10 ↔ V5 eval
- `reports/v10_cross_v5_preview.jpg` — (g) preview
- `reports/v10_label_completeness.json` — (h) độ đầy đủ nhãn + overlay
- `reports/v10_missing_labels_preview.jpg` — (h) preview thiếu nhãn
- `reports/v10_overlay_manual_review.json` — (h') review overlay thủ công

Tái tạo: `conda activate yolov8_ft && python scripts/data_prep/audit_v10/audit_v10_all.py`
