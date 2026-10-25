# Scripts

Standalone CLI scripts tổ chức theo chức năng.

| Folder | Nội dung |
|---|---|
| `data_prep/` | Import các dataset (SOICT, VNv3, Cantho, UA-DETRAC) và merge |
| `training/` | Train model V2, V3, V3-1280 + experiments phụ |
| `evaluation/` | Eval mAP, counting accuracy, FPS benchmark |
| `demo/` | Scripts demo end-to-end pipeline |
| `utils/` | Tiện ích chung (strip_emojis, train.sh) |

## Code chính (Python package)

Phần core nằm ở `src/` — detection, tracking, pipeline, evaluation, stats.
Scripts ở đây gọi `src/` để chạy workflow cụ thể.

## Workflow điển hình

```bash
# 1. Chuẩn bị data
python scripts/data_prep/import_soict.py
python scripts/data_prep/merge_v3.py

# 2. Train
python scripts/training/v3_vn_only.py

# 3. Eval
python scripts/evaluation/eval_all_models.py
python scripts/evaluation/eval_coco_vs_v3.py

# 4. Chạy pipeline trên video
python -m src.pipeline.run --video data/test_videos/raw/demo_traffic.mp4 \
    --weights weights/v3_vn_best.pt --video-key demo_traffic \
    --out-csv results/tables/demo_events.csv \
    --out-video results/videos/demo_annotated.mp4
```
