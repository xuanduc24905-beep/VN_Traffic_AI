# Phase 4 — V7 pipeline (DETRAC bus/truck boost + imgsz=1280)

Pipeline self-contained cho experiment V7: thêm DETRAC bus/truck subset
nhỏ vào V5 train, train yolov8s @ imgsz=1280 warm start từ V5+.

## Files trong folder này

| File | Mô tả |
|---|---|
| `import_detrac_bustruck.py` | Phase 4a — audit DETRAC videos, pick top-K bus/van heavy, video-aware subsample, filter class (bỏ car + others), remap bus→2 / van→3 |
| `build_v7_merged.py` | Phase 4b — merge V5 train (2.8k VN clean) + DETRAC subset (~550 frames) → `data/merged_v7/` |
| `v7_train_1280.py` | Phase 4c — yolov8s @ 1280, batch 8, warm từ V5+ best.pt, lr 0.002, cos_lr, 40ep |
| `run_pipeline.sh` | Orchestrator chạy tuần tự 4a→4b→4c→4d với logging + smart resume |

## Shared dependencies (không trong folder này)

- `scripts/eval/eval_all_models_full.py` — eval script dùng chung cho mọi
  pipeline (V5+/V6/V7). Phase 4d gọi script này.
- UA-DETRAC data tại `/home/xuand/yolov8_finetune/data/detrac/` (external)
- V5+ weights: `runs/detect/train_v8s_v5plus_960/weights/best.pt`

## Cách chạy

### 1 lệnh (background, recommended)

```bash
cd /home/xuand/vn_traffic_ai
nohup bash scripts/phase4_v7/run_pipeline.sh > logs/phase4_v7.log 2>&1 &
disown
```

### Foreground (xem realtime)

```bash
cd /home/xuand/vn_traffic_ai
bash scripts/phase4_v7/run_pipeline.sh
```

### Track progress

```bash
tail -f logs/phase4_v7.log                 # overall
tail -f logs/phase4_03_train_v7.log        # training (phase 4c, lâu nhất)
wc -l runs/detect/train_v8s_v7_1280/results.csv   # minus 1 = epoch done
```

## Config chính

| Phase | Config |
|---|---|
| DETRAC subset | Top 6 bus-heavy + top 6 van-heavy videos, stride 20, ~550 frames |
| Class mapping | DETRAC car (0) → DROP, bus (1) → V5 bus (2), van (2) → V5 truck (3), others (3) → DROP |
| Train V7 | yolov8s, imgsz=1280, batch=8, lr0=0.002, cos_lr, patience=15, 40ep, mixup=0.1, copy_paste=0.3 |
| Warm start | V5+ best.pt (giữ features V5+ học) |

## Expected results

- V5+ hiện: mAP50 test = 0.782
- V7 dự kiến: mAP50 test ~0.80-0.83
- Motor: 0.84 → ~0.86-0.88 (do 1280)
- Bus: 0.72 → ~0.76-0.80 (DETRAC bus examples)
- Truck: 0.69 → ~0.72-0.76 (DETRAC van examples)
- Car: giữ ~0.88 (không add vào train)

## Khác biệt với experiment DETRAC CŨ (failed)

| | Cũ (failed, mAP 0.62) | V7 mới |
|---|---|---|
| DETRAC classes | car + bus + van | **Chỉ bus + van** |
| Subset size | ~60k full | **~550 frames** (top-K) |
| Video split | Random (leak) | **Video-aware stride 20** |
| Init | Fresh COCO | **Warm từ V5+** |
| Imgsz | 512 | **1280** |

Thay đổi này đáp ứng empirical finding: naive DETRAC merge gây class
confusion. V7 limit DETRAC thành booster nhỏ cho bus/truck thay vì
dominant data source.

## Thời gian

- 4a (import): ~5 min
- 4b (build): <1 min
- 4c (train): **~2-3h** (batch 8 @ 1280, GPU-bound)
- 4d (eval): ~20-30 min
- **Tổng: ~2.5-3.5h**

## Output

- `runs/detect/train_v8s_v7_1280/weights/best.pt` — V7 model
- `data/merged_v7/` — V7 dataset (symlinks)
- `data/detrac_bustruck_subset/manifest.json` — reproducibility
- `results/v5_cross_eval/FINAL_COMPARISON.md` — updated bảng với V7
