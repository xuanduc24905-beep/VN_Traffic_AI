# Cross-domain evaluation — V5+ CCTV (ngang) vs Vietnam Vehicles aerial (trên cao)

## Bảng chính: mAP@0.5 trên các bộ test

| Mô hình | V5+ test (CCTV) | Aerial test | Aerial val | Gap CCTV→Aerial |
|---|---|---|---|---|
| YOLOv8s-COCO (zero-shot) | 0.0922 | — | — | — |
| V5 honest (640) | 0.6648 | — | — | — |
| V5+ 960px | 0.7819 | — | — | — |
| p4_scale (cuối) | 0.7869 | — | — | — |
| p4_imgsz1280 | 0.7745 | — | — | — |
| V6 pseudo | 0.7608 | — | — | — |
| r_p2 (ablation) | 0.7530 | — | — | — |
| r_clspw (ablation) | 0.7927 | — | — | — |
| r_progressive (ablation) | 0.7711 | — | — | — |

## Bảng phụ: mAP@0.5:0.95 (strict metric)

| Mô hình | V5+ test | Aerial test | Aerial val |
|---|---|---|---|
| YOLOv8s-COCO (zero-shot) | 0.0221 | — | — |
| V5 honest (640) | 0.3399 | — | — |
| V5+ 960px | 0.4134 | — | — |
| p4_scale (cuối) | 0.4235 | — | — |
| p4_imgsz1280 | 0.4179 | — | — |
| V6 pseudo | 0.4111 | — | — |
| r_p2 (ablation) | 0.3950 | — | — |
| r_clspw (ablation) | 0.4226 | — | — |
| r_progressive (ablation) | 0.4147 | — | — |
