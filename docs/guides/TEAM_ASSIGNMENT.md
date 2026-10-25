# Phân công nhóm — VN Traffic AI

## Người 1 — Object Detection

**Module:** [src/detection/](../src/detection/)

**Việc cụ thể:**
- Chuẩn bị dataset gộp bằng `scripts/data_prep/merge_datasets.py` (phối hợp với Người 4).
- Train YOLOv8 trên merged data 5 lớp qua `src/detection/train.py`.
- Đánh giá mAP@0.5 và mAP@0.5:0.95 trên tập test qua `src/detection/evaluate.py`.
- Tuning `conf` và `iou` threshold, chọn cấu hình tốt nhất bàn giao cho Người 2.

**Deliverable:**
- File `runs/detect/train/weights/best.pt`.
- Bảng mAP per-class trên test set `results/tables/detection_map.csv`.

**Command mẫu:**
```bash
python -m src.detection.train --data data/data.yaml --epochs 50 --batch 16
python -m src.detection.evaluate --weights runs/detect/train/weights/best.pt --split test
```

---

## Người 2 — Tracking + Counting

**Module:** [src/tracking/](../src/tracking/)

**Việc cụ thể:**
- Tích hợp YOLOv8 Track với ByteTrack và BoT-SORT (đã có sẵn trong Ultralytics).
- Test giữ ID xuyên frame trên video mẫu VisDrone-VID.
- Thiết kế `counter.py`: line/zone crossing, chống đếm trùng.
- Định nghĩa line/zone trong `configs/counting_zones.json` cho từng video test.

**Deliverable:**
- File `src/tracking/track.py` + `src/tracking/counter.py` chạy được.
- Video demo có bbox + ID + counter overlay (2-3 phút).

**Command mẫu:**
```bash
python -m src.tracking.track \
    --weights runs/detect/train/weights/best.pt \
    --source data/test_videos/raw/nga_tu_01.mp4 \
    --tracker bytetrack.yaml
```

---

## Người 3 — Statistics + Visualization

**Module:** [src/stats/](../src/stats/)

**Việc cụ thể:**
- Nhận dữ liệu event từ pipeline (CSV: frame, time_sec, line, track_id, class_name).
- Aggregate: thống kê theo loại xe, theo thời gian (phút), theo line.
- Xuất CSV kết quả và biểu đồ.

**Deliverable:**
- `results/tables/nga_tu_01_summary.csv`.
- `results/figures/counts_per_class.png`.
- `results/figures/counts_per_minute.png`.

**Code sẵn có:** [src/stats/aggregator.py](../src/stats/aggregator.py) và
[src/stats/visualize.py](../src/stats/visualize.py).

---

## Người 4 — Evaluation + System Integration

**Module:** [src/evaluation/](../src/evaluation/) và [src/pipeline/](../src/pipeline/)

**Việc cụ thể:**
- Quay 2-3 video test tại ngã tư/đường VN (5-10 phút mỗi video).
- Sinh template ground truth bằng `scripts/data_prep/prepare_test_video.py`.
- Đếm thủ công phương tiện qua từng line điền vào JSON GT.
- Chạy pipeline `src/pipeline/run.py` end-to-end.
- Tính Accuracy, MAE, MAPE bằng `src/evaluation/compare.py`.
- Viết report tổng hợp so sánh 3 tracker (ByteTrack vs BoT-SORT vs no-tracker).

**Deliverable:**
- 2-3 video test có ground truth JSON.
- Bảng so sánh Accuracy/MAE/MAPE giữa các cấu hình `results/tables/evaluation.csv`.

**Command mẫu:**
```bash
# Tạo template GT
python scripts/data_prep/prepare_test_video.py \
    --video data/test_videos/raw/nga_tu_01.mp4 \
    --out data/test_videos/ground_truth/nga_tu_01.json

# Chạy pipeline
python -m src.pipeline.run \
    --video data/test_videos/raw/nga_tu_01.mp4 \
    --weights runs/detect/train/weights/best.pt \
    --tracker bytetrack.yaml \
    --counting-config configs/counting_zones.json \
    --video-key nga_tu_01 \
    --out-csv results/tables/nga_tu_01_events.csv \
    --out-video results/videos/nga_tu_01_out.mp4

# Đánh giá
python -m src.evaluation.compare \
    --pred results/tables/nga_tu_01_events.csv \
    --gt data/test_videos/ground_truth/nga_tu_01.json
```

---

## Timeline gợi ý (6 tuần)

| Tuần | Việc |
|---|---|
| 1 | Setup env, download data (Người 1 + 4), quay video test (Người 4) |
| 2 | Merge dataset, train baseline (Người 1); prototype tracker (Người 2) |
| 3 | Fine-tune detection, test bytetrack vs botsort (Người 1 + 2); label GT (Người 4) |
| 4 | Counter chống đếm trùng (Người 2), stats + viz (Người 3) |
| 5 | Tích hợp pipeline, chạy trên 2-3 video test (Người 4) |
| 6 | Đánh giá cuối, viết báo cáo, quay video demo |
