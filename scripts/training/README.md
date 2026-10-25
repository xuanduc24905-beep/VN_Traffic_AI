# Training scripts

Tất cả script train đều có **smart start**: skip nếu đã xong, resume nếu dở, fresh nếu chưa có. `save_period=1` để resume được nếu crash.

| Script | Dataset | Config | Thời gian | Kết quả |
|---|---|---|---|---|
| `v2_fix_small_car.py` | `data/data.yaml` (DETRAC-heavy, 140k) | yolov8s warm từ V1, imgsz=512, scale+mixup | ~3h | mAP50=0.634 (val VN) |
| `v3_vn_only.py` | `data/data_v3.yaml` (VN-only, 20.5k) | yolov8s fresh COCO, imgsz=640, batch=16 | ~3h | **mAP50=0.951** |
| `v3_vn_only_1280.py` | `data/data_v3.yaml` | yolov8s fresh COCO, **imgsz=1280**, batch=8 | ~5-8h | TBD (target: motorcycle mAP50-95 tăng) |

## Chạy

```bash
python scripts/training/v3_vn_only.py
```

Weights lưu ở `runs/detect/{run_name}/weights/best.pt`.

## experiments/

Các experiment phụ dùng để báo cáo ablation study:
- `v2_freeze10.py` — freeze 10 layer backbone (test generalization)
- `v2_class_weights.py` — boost cls loss cho motorcycle (fix underfit)
- `v2_yolov8m.py` — YOLOv8m from scratch (architecture scaling)
