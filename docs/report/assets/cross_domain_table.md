# Cross-domain evaluation — V5+ CCTV (ngang) vs Vietnam Vehicles aerial (trên cao)

## Bảng chính: mAP@0.5 trên các bộ test

| Mô hình | V5+ test (CCTV) | Aerial test | Aerial val | Gap CCTV→Aerial test |
|---|---|---|---|---|
| YOLOv8s-COCO (zero-shot) | 0.0922 | 0.0672 | 0.1087 | +0.0249 |
| V5 honest (640) | 0.6648 | 0.4297 | 0.4931 | +0.2351 |
| V5+ 960px | 0.7819 | 0.5994 | 0.6141 | +0.1826 |
| p4_scale | 0.7869 | 0.6373 | 0.6483 | +0.1497 |
| p4_imgsz1280 | 0.7745 | 0.6015 | 0.6374 | +0.1729 |
| V6 pseudo | 0.7608 | 0.5671 | 0.5832 | +0.1937 |
| r_p2 (ablation) | 0.7530 | 0.5369 | 0.5516 | +0.2161 |
| r_clspw (mô hình cuối) | 0.7927 | 0.5676 | 0.6135 | +0.2251 |
| r_progressive (ablation) | 0.7711 | 0.6115 | 0.6297 | +0.1596 |

## Bảng phụ: mAP@0.5:0.95 (strict metric)

| Mô hình | V5+ test | Aerial test | Aerial val |
|---|---|---|---|
| YOLOv8s-COCO (zero-shot) | 0.0221 | 0.0272 | 0.0469 |
| V5 honest (640) | 0.3399 | 0.2011 | 0.2400 |
| V5+ 960px | 0.4134 | 0.2972 | 0.3137 |
| p4_scale | 0.4235 | 0.3290 | 0.3171 |
| p4_imgsz1280 | 0.4179 | 0.2997 | 0.3197 |
| V6 pseudo | 0.4111 | 0.2937 | 0.2936 |
| r_p2 (ablation) | 0.3950 | 0.2557 | 0.2637 |
| r_clspw (mô hình cuối) | 0.4226 | 0.2550 | 0.2856 |
| r_progressive (ablation) | 0.4147 | 0.3062 | 0.3150 |
