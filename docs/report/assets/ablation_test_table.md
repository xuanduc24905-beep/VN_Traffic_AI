# Kết quả ablation trên TEST set (data_v5plus)

| Mô hình | mAP@50 | mAP@50-95 | P | R | AP motorcycle | AP car | AP bus | AP truck |
|---|---|---|---|---|---|---|---|---|
| v5plus_baseline | 0.7819 | 0.4134 | 0.7677 | 0.7073 | 0.8406 | 0.8839 | 0.7164 | 0.6868 |
| r_p2 | — | — | — | — | — | — | — | — | *(missing)* |
| r_clspw | — | — | — | — | — | — | — | — | *(missing)* |
| r_progressive | — | — | — | — | — | — | — | — | *(missing)* |
| p4_scale_from_baseline | 0.7869 | 0.4235 | 0.7590 | 0.7325 | 0.8616 | 0.8851 | 0.7217 | 0.6793 |
| p4_imgsz1280_from_baseline | 0.7745 | 0.4179 | 0.7287 | 0.7431 | 0.8624 | 0.8888 | 0.7418 | 0.6048 |

## Delta so với v5plus_baseline

| Mô hình | Δ mAP@50 | Δ motorcycle | Δ car | Δ bus | Δ truck |
|---|---|---|---|---|---|
| p4_scale_from_baseline | +0.0050 | +0.0210 | +0.0012 | +0.0053 | -0.0075 |
| p4_imgsz1280_from_baseline | -0.0075 | +0.0219 | +0.0049 | +0.0254 | -0.0820 |
