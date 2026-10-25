# Weights

Mọi file `.pt` ở đây, phân chia theo mục đích.

## Cấu trúc

```
weights/
├── pretrained/              # Pretrained public (COCO) + variants
│   ├── yolov8s.pt
│   ├── yolov8m.pt
│   ├── yolov8n.pt
│   └── yolo26n.pt
└── trained/                 # Model train nội bộ (fine-tune/scratch trên dataset VN)
    ├── v1_baseline.pt       # mAP50 val VN = 0.615 (train DETRAC-heavy 140k)
    ├── v2_fix.pt            # mAP50 val VN = 0.634 (fine-tune V1, target car-small)
    └── v3_vn_best.pt        # mAP50 val VN = 0.951 ← MODEL CHÍNH
```

## Dùng

```python
from ultralytics import YOLO
model = YOLO("weights/trained/v3_vn_best.pt")
```

## So sánh

| Model | Size | Train data | mAP50 (val VN) | mAP50-95 |
|---|---:|---|---:|---:|
| yolov8s COCO vanilla | 22MB | COCO (ngoài VN) | 0.439 | 0.245 |
| v1_baseline | 6MB | DETRAC-heavy 140k | 0.615 | 0.321 |
| v2_fix | 22MB | DETRAC-heavy + aug | 0.634 | 0.346 |
| **v3_vn_best** | 22MB | **VN-only 20.5k** | **0.951** | **0.670** |
