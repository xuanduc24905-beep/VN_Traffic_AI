# Data leak trong V3 — ghi chép phát hiện

**Ngày:** 2026-10-25
**Context:** Sau khi train xong `train_v8s_v4_vn_cctv` (40 epoch), kiểm tra chất lượng dataset V3 (base) + V4 (V3 + vn_cctv Đà Nẵng) để tìm hướng cải thiện. Phát hiện vấn đề nghiêm trọng về **evaluation methodology**.

---

## 1. mAP đang có — không phải "best model" như tưởng

| Run | imgsz | data | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **train_v8s_v3_1280** | 1280 | V3 (20.5k) | **0.954** | **0.674** |
| train_v8s_v3_baseline | 640 | V3 | 0.951 | 0.669 |
| train_v8s_v4_finetune | 640 | V4 (49.5k) | 0.948 | 0.665 |
| train_v8s_v4_vn_cctv | 640 | V4 | 0.946 | 0.663 |

Thêm 29k ảnh Đà Nẵng (V4) **không giúp** cải thiện trên V3 val set. Hướng điều tra: tại sao?

---

## 2. Vấn đề 1 — vn_cctv (Đà Nẵng)

`data/raw_vn_cctv/` (26k train + 2.9k test):

- **Toàn bộ là frame liên tiếp từ 1 video.** Frame ID: 000001 → 029198, 90% khoảng cách giữa 2 frame kề nhau = 1 → chỉ là vài trăm scene độc lập được upsample thành 26k ảnh.
- **Class imbalance cực mạnh:** car 66.4%, motorbike 31.7%, bus 1.1%, truck 0.8%. Bus + truck tổng cộng ~2%.
- **Ảnh low-res 640x360** — train imgsz=1280 không có lợi vì ảnh gốc nhỏ hơn.
- `data.yaml` khai báo `val: ../valid/images` nhưng folder `valid/` không tồn tại — chỉ có train + test.

Script `scripts/data_prep/merge_v4.py` đã remap class đúng, nhưng dataset này về bản chất không đủ chất lượng để train.

---

## 3. Vấn đề 2 — V3 có data leak 100%

Khi kiểm tra V3 kỹ hơn, phát hiện cả 3 split (train/val/test) đều được tạo bằng cách **random-split frames** từ cùng video, dẫn đến **leak hoàn toàn** ở cấp độ video.

### Số liệu

| | train | val | test |
|---|---|---|---|
| Số video riêng biệt | 15 | 12 | 12 |
| Video chung với train | — | **12/12** | **12/12** |
| Frames từ video đã có trong train | — | **1779/1779 (100%)** | **1053/1053 (100%)** |

Chi tiết các video bị leak:

```
soict24_cam_01   train=950   val=276   test=126
soict24_cam_02   train=702   val=198   test=100
soict24_cam_04   train=716   val=218   test=106
soict24_cam_06   train=695   val=202   test=103
soict24_cam_07   train=681   val=213   test=106
soict24_cam_09   train=694   val=213   test=93
soict24_cam_10   train=366   val=87    test=47
vnv3             train=1235  val=156   test=156   (một video duy nhất!)
cantho_* (4 video)  tất cả đều leak train/val/test
```

Chỉ 3 video `soictnight_cam_03/05/08` (~14k frames) là train-only, không có trong val/test.

### Hệ quả

- mAP50 = 0.954 trên V3_1280 **không đo generalization** — nó đo khả năng "nhớ cảnh".
- Frame_100 trong train và frame_101 trong val gần như giống hệt → model chỉ cần "nhớ" là xong.
- Thêm vn_cctv (V4) không giúp vì mAP đã saturate do leak.
- Chênh lệch v3_1280 (0.954) vs v3_baseline (0.951) có thể hoàn toàn là noise từ leak chứ không phải lợi ích thực của imgsz 1280.

---

## 4. Vấn đề 3 — Chất lượng các sub-dataset trong V3

### SOICT (18.6k train, 188k boxes)
- 40.7% boxes là small (`w×h < 0.0025`, tương đương <32px @640) → khó cho imgsz 640
- Chia thành soictnight (3 cam đêm, 14k frames liên tiếp) + soict24 (10 cam ngày, 4.8k frames)
- Phân phối class: motorcycle 60% / car 23% / bus 6% / truck 11% — tạm cân bằng
- gap<=3 giữa frame kề nhau: 95-100% → cũng là video frames

### Cantho (678 train, 6.7k boxes)
- 4 video, 56-66% frame liên tiếp
- **Bus chỉ có 38 boxes (0.6%)** — gần như không có bus
- Phân phối: motorcycle 48% / car 47% / bus 0.6% / truck 3.5%

### VNv3 (1235 train, 7.2k boxes)
- **1 video duy nhất**, 79% frame liên tiếp → toàn bộ 1547 ảnh = 1 scene
- Class balance tốt nhất: motorcycle 31% / car 39% / bus 5.6% / truck 24.6%

---

## 5. Baseline để so sánh fair

| Mốc | Dataset | mAP50 | mAP50-95 |
|---|---|---|---|
| YOLOv8s pretrained (official) | COCO val2017 (80 cls) | 0.618 | 0.449 |
| v8s zero-shot | V3 val (4 cls, không remap COCO→V3) | ~0.12 | ~0.03 |
| v8s finetune V3 (có leak) | V3 val | 0.954 | 0.674 |

Gain thật của finetune (nếu bỏ qua leak): COCO→V3 = **+0.22 mAP50-95**. Phần "0.67 → 0.95" chủ yếu là leak, không phải tiến bộ thực.

---

## 6. Kế hoạch khắc phục

1. **Resplit V3 theo video_id** (`scripts/data_prep/resplit_v3_by_video.py` — chưa viết):
   - Train: 10/15 video
   - Val: 2-3 video riêng
   - Test: 2-3 video riêng khác
   - Mỗi video chỉ xuất hiện trong 1 split
2. **Subsample video dense:** giữ 1 frame mỗi 5-10 frames để giảm redundancy, tăng diversity hiệu dụng.
3. **Train lại v8s @ 1280** trên split mới (~4-5h GPU).
4. **Dùng vn_cctv làm out-of-domain test set** thay vì gộp vào train.

Dự đoán sau khi fix: mAP50 ~0.70-0.82 (số thật, thấp hơn 0.95 ảo nhưng có giá trị khoa học).

---

## 7. Insight cho báo cáo

- "Finetune domain adaptation thực sự cải thiện performance (+22% mAP50-95 so với COCO baseline)."
- "Phát hiện video-level leak trong methodology ban đầu — mAP 0.95 ảo. Sau khi resplit theo video_id, mAP thực là X.XX."
- "Thêm dataset (V3→V4) không giúp khi evaluation còn bị leak — chỉ ra tầm quan trọng của methodology trên tối ưu architecture/data."

Câu chuyện này mạnh hơn 1 con số mAP cao — nó cho thấy quy trình nghiên cứu thực tế: observe → verify → identify issue → fix.
