"""V7 — yolov8s @ imgsz=1280, warm start từ V5+ best.pt.

Config:
- Data: merged_v7 (V5 train + DETRAC bus/truck subset)
- Init: V5+ best.pt (KHÔNG fresh COCO) → giữ features đã học của V5+
- imgsz=1280 (max native SOICT), batch=8 (12GB VRAM)
- lr0=0.002 (thấp vì warm start, không muốn phá V5+ learned features)
- cos_lr, patience=15, 40ep
- mixup=0.1, copy_paste=0.3 (giữ augment V5+)

Expected: V5+ 0.782 → V7 ~0.80-0.83 (motor + bus/truck tăng)

Smart start: SKIP / RESUME / FRESH (warm from V5+).
"""
import sys
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path("/home/xuand/vn_traffic_ai").resolve()
RUNS_DIR     = PROJECT_ROOT / "runs" / "detect"
DATA_YAML    = PROJECT_ROOT / "data" / "data_v7.yaml"
# Warm từ V5+ best (không phải COCO)
INIT_WEIGHTS = PROJECT_ROOT / "runs/detect/train_v8s_v5plus_960/weights/best.pt"

assert INIT_WEIGHTS.exists(), f"Thiếu {INIT_WEIGHTS}. Train V5+ trước."
assert DATA_YAML.exists(), f"Thiếu {DATA_YAML}. Build V7 trước."

RUN_NAME = "train_v8s_v7_1280"
RUN_DIR  = RUNS_DIR / RUN_NAME

CFG = dict(
    data        = str(DATA_YAML),
    epochs      = 40,
    batch       = 8,              # ← 12GB VRAM constraint ở 1280
    imgsz       = 1280,           # ← MAX resolution
    lr0         = 0.002,          # ← thấp vì warm start
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
    mixup       = 0.1,
    copy_paste  = 0.3,
)


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
        print(f"[SKIP] Done. Best: {best_pt}")
        return 0

    if last_pt.exists() and done > 0:
        print(f"[RESUME] từ epoch {done}/{target}")
        model = YOLO(str(last_pt))
        model.train(resume=True)
    else:
        print(f"[FRESH WARM] init từ V5+ {INIT_WEIGHTS}, imgsz=1280")
        model = YOLO(str(INIT_WEIGHTS))
        model.train(**CFG)

    print(f"\n[DONE] Best: {best_pt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
