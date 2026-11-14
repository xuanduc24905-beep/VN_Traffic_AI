# Datasheet — V5+clean (V5 + vietnam_vehicles v10, đã làm sạch)

_Sinh tự động bởi [verify_v5plus_clean.py](../scripts/data_prep/verify_v5plus_clean.py) từ dataset build bởi [build_v5plus_clean.py](../scripts/data_prep/build_v5plus_clean.py) (git `2030a6d`, 2026-11-14T14:01:17)._

## Kết luận

✅ **Cả 4 biến thể qua toàn bộ kiểm tra — sẵn sàng train.**

| Kiểm tra | `v5base` | `full` | `nooverlay` | `ovmatched` |
|---|---|---|---|---|
| V1 Test = V5 test; val = V5 val bỏ Cần Thơ (SHA-256) | ✅ | ✅ | ✅ | ✅ |
| V2 Cấu trúc (symlink, nhãn, không v10 trong eval) | ✅ | ✅ | ✅ | ✅ |
| V3 Nhãn hợp lệ (validate_labels) | ✅ | ✅ | ✅ | ✅ |
| V4 Leak cấp video (audit_leak) | ✅ | ✅ | ✅ | ✅ |
| V6 Class v10 sau remap ∈ schema V5 | ✅ | ✅ | ✅ | ✅ |
| V5 Leak nội dung: pHash train ↔ val/test ≤5 | ⊂ `full` | ✅ 0 cặp | ⊂ `full` | ⊂ `full` |
| V8 Leak cấp camera: NCC ảnh nền val/test vs train < đối chứng dương | ⊂ `full` | ✅ 0.5059 < 0.6774 | ⊂ `full` | ⊂ `full` |
| V7 Ablation hợp lệ (nooverlay toàn không-ID, ovmatched toàn có-ID, cùng cỡ & camera) | – | – | ✅ | ✅ |

_Train của mọi biến thể là tập con của train `full` (V5 train + v10 đã lọc), val/test giống hệt nhau → kết quả V5/V8 trên `full` áp dụng cho cả 4._

## 1. Mục đích

Bổ sung dữ liệu CCTV Việt Nam (Roboflow `visionvn/vietnam_vehicles` v10) vào train của V5 mà **không** làm sai benchmark: **test giữ nguyên V5** (SOICT24 cam_09/10, camera chưa thấy) để so sánh được với mọi kết quả trước; **val bỏ phần Cần Thơ** vì cùng camera với train (mục 3). Bản này sửa các vấn đề phát hiện trong audit ([v10_audit_summary.md](v10_audit_summary.md)).

## 2. Thành phần

| Biến thể | YAML | Train (V5 + v10) | % v10 | v10 có overlay | Val | Test | Mục đích |
|---|---|---|---|---|---|---|---|
| `v5base` | `data/data_v5_cleanval.yaml` | 2833 (2833 + 0) | 0.0% | 0% | 1000 | 1500 | Baseline V5 train lại cùng giao thức |
| `full` | `data/data_v5plus_clean.yaml` | 6969 (2833 + 4136) | 59.35% | 87.55% | 1000 | 1500 | **Bản chính** |
| `nooverlay` | `data/data_v5plus_clean_nooverlay.yaml` | 3348 (2833 + 515) | 15.38% | 0.0% | 1000 | 1500 | Ablation: v10 không overlay |
| `ovmatched` | `data/data_v5plus_clean_ovmatched.yaml` | 3348 (2833 + 515) | 15.38% | 100.0% | 1000 | 1500 | Đối chứng: v10 có overlay, cùng cỡ & camera |

Số box theo class:

| Phần | motorcycle | car | bus | truck |
|---|---|---|---|---|
| V5 train | 10124 | 5656 | 1399 | 3102 |
| v10 (full) | 3044 | 224 | 231 | 1813 |
| v10 (nooverlay) | 78 | 28 | 4 | 512 |
| v10 (ovmatched) | 128 | 31 | 51 | 397 |
| Val (V5, bỏ Cần Thơ) | 5852 | 1654 | 447 | 1913 |
| Test (V5) | 39807 | 8031 | 756 | 852 |

## 3. Sửa val: bỏ video Cần Thơ (leak cấp camera trong V5 gốc)

Kiểm tra pHash train↔eval phát hiện 102 cặp gần trùng giữa **V5 train** và **V5 val**, toàn bộ thuộc Cần Thơ. Kiểm bằng ảnh nền camera và bằng mắt: 4 video Cần Thơ (2 ở train: CoopMart 8h–8h30, 8h10–8h30; 2 ở val: CoopMart 8h30–9h00, camDLHB 13h20) là **cùng một camera** (NCC 0.98 / 0.76; đối chứng cùng video ≥ 0.73). `audit_leak.py` không bắt được vì so theo tên video, không theo camera.
- Hệ quả: 466/1466 frame val (32%) đến từ camera đã thấy khi train → val lạc quan, ảnh hưởng chọn checkpoint & early-stop. **Test (SOICT cam_09/10) không bị ảnh hưởng** (pHash gần nhất d=12, NCC ≤ 0.35).
- Sửa: val = `soict24_cam_07` (1000 frame, camera riêng). 466 frame Cần Thơ **không** chuyển sang train (giữ V5 train y nguyên). Test giữ nguyên tuyệt đối (fingerprint V1).
- Vì đổi val → cách chọn checkpoint khác các run cũ; do đó có biến thể `v5base` để **train lại baseline V5** theo đúng giao thức mới. So sánh khoa học là `v5base` vs `full` (cùng val, cùng test, cùng cấu hình).
- Kết quả cũ (V5 0.639, V5+ 0.747 mAP50 trên test) vẫn hợp lệ về test nhưng checkpoint được chọn bằng val có leak.

Video val/test và camera train gần nhất (NCC ảnh nền):

- `test:soict24_cam_09` → `v10:hodabac554`: 0.5059
- `test:soict24_cam_10` → `soict24_cam_02`: 0.284
- `val:soict24_cam_07` → `soict24_cam_06`: 0.4748

## 4. Quy tắc làm sạch v10 (cố định trước khi train)

Từ 4824 ảnh v10 → giữ **4136**.

| Quy tắc | Lý do | Ảnh loại |
|---|---|---|
| R4 nhãn lỗi/rỗng | an toàn | 1 |
| R1 bỏ **cả ảnh** có `xe_ba_gac` | V5 không có class này. Bản V5+ cũ chỉ xoá box → ~110 xe ba gác thành "nền" chưa gán nhãn (false negative có hệ thống) | 529 |
| R2 gộp ảnh trùng thật: cùng camera + pHash≤4 + (cùng biển số hoặc cách ≤2s) | tránh over-weight. **Không** dùng pHash thuần: camera tĩnh → 6/8 cặp pHash≤2 kiểm bằng mắt là xe khác | 14 |
| R3 bỏ ảnh có xe chưa gán nhãn theo detector COCO (yolov8m.pt, conf≥0.5) | giảm false negative trong nhãn | 144 |

Class remap v10→V5: `container`→truck, `oto`→car, `xe_ba_gac`→DROP_IMAGE, `xe_khach`→bus, `xe_may`→motorcycle, `xe_tai`→truck.
Danh sách từng ảnh bị loại + lý do: `reports/v5plus_clean/manifest_full.json` (`v10_excluded`).

## 5. Overlay vẽ sẵn & thiết kế ablation

~83% ảnh v10 có tag/khung của hệ thống tracking gốc vẽ lên pixel, đặt ngay tại xe được gán nhãn → nguy cơ model học lối tắt "tag màu = xe". Không làm sai số trên test V5 (V5 không có overlay) nhưng có thể làm giảm lợi ích của v10.
- Quy tắc **tên file có vehicle ID ⇔ có overlay**: tune 59/60; **held-out mù 40/40** (nhận định ghi trước khi mở đáp án — `reports/v5plus_clean/overlay_rule_validation.json`).
- `nooverlay` vs `ovmatched`: cùng V5 train, cùng số ảnh v10 (515), cùng phân bố camera (515/515 khớp camera, seed 0) → khác biệt chủ yếu là overlay.
- Giới hạn: ảnh không-ID chụp 01–02/2024, ảnh có-ID cùng camera chụp 03–04/2024 → còn lệch thời điểm/thời tiết; phân bố class v10 hai bên cũng không trùng khít (bảng mục 2).

## 6. Giao thức train/đánh giá (đề xuất để kết quả có giá trị khoa học)

**Vì sao phải train lại cả baseline:** so sánh cũ V5 0.639 → V5+ 0.747 mAP50 **không quy được cho v10** — hai run khác nhau đồng thời ở imgsz (640→960), tổng epoch (40→80 do warm-start), augmentation (mixup/copy_paste/xoay), lr0 và lịch lr. Riêng imgsz 960 đã có thể tăng mạnh AP xe máy nhỏ.

1. **Một recipe chung cho mọi biến thể**, chỉ đổi dữ liệu: init COCO `yolov8s.pt`, imgsz 960, 60 epoch, không early-stop, cùng augmentation — [v5plus_clean_train.py](../scripts/training/v5plus_clean_train.py) (`RECIPE`). Không warm-start từ `train_v8s_v5_honest/best.pt` vì checkpoint đó được chọn bằng val có leak.
2. **Chọn checkpoint chỉ bằng val sạch** (cam_07); script chạy **test V5 đúng một lần** sau khi train xong và lưu `reports/v5plus_clean/results/<run>.json`. Không chỉnh quy tắc làm sạch/recipe sau khi nhìn test.
3. Thứ tự: (a) `v5base` và `full` seed 42 — cặp so sánh chính. (b) `nooverlay` và `ovmatched` seed 42. (c) Nếu chênh lệch nhỏ (<~1 điểm mAP50) → thêm seed 0 và 1 (`--seed 42 0 1`), báo trung bình ± độ lệch chuẩn (`--summarize`); một seed không đủ để kết luận khác biệt nhỏ.
   - Số epoch bằng nhau ⇒ `full` có nhiều iteration hơn `v5base` (2.5× ảnh). Đây là quy ước phổ biến; nếu cần tách hiệu ứng "nhiều bước tối ưu hơn" thì chạy thêm `v5base` với 150 epoch.
4. Báo cáo mAP50, mAP50-95 và AP theo class; với ablation overlay báo cả kích thước dữ liệu.

## 7. Hạn chế đã biết

- v10 vẫn chiếm ~59% ảnh train bản `full`, khác domain (snapshot vi phạm, 1–3 xe/ảnh) so với test SOICT (mật độ cao, ~33 box/ảnh).
- R3 dùng detector COCO → chỉ bắt xe chưa gán mà COCO thấy (cận dưới); không áp cho V5 train (giữ nguyên bản đã dùng ở mọi kết quả trước để so sánh được).
- 83% ảnh trong `full` vẫn có overlay; ablation ở mục 4 dùng để đo ảnh hưởng thay vì đoán.
- `data/merged_v5plus` (bản cũ) được giữ nguyên làm baseline tái lập.

## 8. Tái lập

```bash
conda activate yolov8_ft
python scripts/data_prep/build_v5plus_clean.py
python scripts/data_prep/verify_v5plus_clean.py
```
Manifest từng biến thể (nguồn, camera, overlay của từng ảnh v10; fingerprint val/test): `reports/v5plus_clean/manifest_{full,nooverlay,ovmatched}.json`.
