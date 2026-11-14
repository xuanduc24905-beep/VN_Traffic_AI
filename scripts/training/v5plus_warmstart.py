"""V5+ warm-start training từ V5 best weights.

Strategy:
- Init weights: V5 best (không phải COCO) → giữ feature đã học từ V5
- Data: data_v5plus.yaml (V5 train + vietnam_vehicles v10)
- Val/test giữ nguyên V5 → so sánh fair với V5 baseline
- LR thấp (0.001) vì warm-start
- Heavy augmentation để bù dataset size ~7k

Expected: V5 mAP50 ~0.70 → V5+ mAP50 ~0.73-0.78.

Smart start: SKIP / RESUME / FRESH.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path("/home/xuand/vn_traffic_ai").resolve()
RUNS_DIR     = PROJECT_ROOT / "runs" / "detect"
DATA_YAML    = PROJECT_ROOT / "data" / "data_v5plus.yaml"
V5_BEST      = PROJECT_ROOT / "runs" / "detect" / "train_v8s_v5_honest" / "weights" / "best.pt"
COCO_FALLBACK = PROJECT_ROOT / "yolov8s.pt"

assert DATA_YAML.exists(), f"Thiếu {DATA_YAML}. Chạy build_v5plus_merged.py trước."

RUN_NAME = "train_v8s_v5plus_warm"
RUN_DIR  = RUNS_DIR / RUN_NAME

CFG = dict(
    data        = str(DATA_YAML),
    epochs      = 40,
    batch       = 16,
    imgsz       = 960,
    lr0         = 0.001,         # warm-start → LR thấp
    lrf         = 0.01,
    cos_lr      = True,
    patience    = 15,
    optimizer   = "SGD",
    seed        = 42,
    workers     = 8,
    cache       = "disk",
    amp         = True,
    save_period = 5,
    project     = str(RUNS_DIR),
    name        = RUN_NAME,
    exist_ok    = True,
    verbose     = True,
    # Heavy aug bù dataset nhỏ
    mixup       = 0.15,
    copy_paste  = 0.3,
    hsv_h       = 0.015,
    hsv_s       = 0.7,
    hsv_v       = 0.4,
    degrees     = 5,
    scale       = 0.5,
    fliplr      = 0.5,
    flipud      = 0.0,
)


def pick_init_weights() -> Path:
    if V5_BEST.exists():
        print(f"[INIT] Warm-start từ V5 best: {V5_BEST}")
        return V5_BEST
    assert COCO_FALLBACK.exists(), f"Thiếu cả V5 best lẫn {COCO_FALLBACK}"
    print(f"[INIT] V5 best không tồn tại, fallback COCO: {COCO_FALLBACK}")
    return COCO_FALLBACK


def main() -> int:
    last_pt     = RUN_DIR / "weights" / "last.pt"
    best_pt     = RUN_DIR / "weights" / "best.pt"
    results_csv = RUN_DIR / "results.csv"

    done = 0
    if results_csv.exists():
        try:
            done = len(pd.read_csv(results_csv))
        except Exception as e:
            print(f"[WARN] {e}")

    target = CFG["epochs"]
    print(f"[STATE] done={done}/{target}, last={last_pt.exists()}, best={best_pt.exists()}")

    if done >= target and best_pt.exists():
        print(f"[SKIP] Đã xong. Best: {best_pt}")
        return 0

    if last_pt.exists() and done > 0:
        print(f"[RESUME] từ epoch {done}/{target}")
        model = YOLO(str(last_pt))
        model.train(resume=True)
    else:
        init = pick_init_weights()
        print(f"[FRESH] imgsz={CFG['imgsz']}, batch={CFG['batch']}, lr0={CFG['lr0']}")
        model = YOLO(str(init))
        model.train(**CFG)

    print(f"\n[DONE] Best: {best_pt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
