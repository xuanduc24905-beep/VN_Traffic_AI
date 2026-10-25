"""V5+ — upgrade V5 lên imgsz=960 + augmentation nâng cao.

Config:
- Same data (merged_v5, 2.8k train, no leak)
- yolov8s standard, imgsz=960 (vs 640 cũ)
- batch=16 (vừa khít 12GB VRAM ở 960)
- cos_lr=True, lr0=0.005 (thấp hơn vì resolution lớn)
- mixup=0.1 + copy_paste=0.3 cho dense VN scenes
- patience=15, save_period=5

Expected: V5 mAP50 0.665 → V5+ mAP50 ~0.70-0.73

Smart start: SKIP / RESUME / FRESH.
"""
import sys
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path("/home/xuand/vn_traffic_ai").resolve()
RUNS_DIR     = PROJECT_ROOT / "runs" / "detect"
DATA_YAML    = PROJECT_ROOT / "data" / "data_v5.yaml"
INIT_WEIGHTS = PROJECT_ROOT / "yolov8s.pt"

assert INIT_WEIGHTS.exists(), f"Thiếu {INIT_WEIGHTS}"
assert DATA_YAML.exists(), f"Thiếu {DATA_YAML}"

RUN_NAME = "train_v8s_v5plus_960"
RUN_DIR  = RUNS_DIR / RUN_NAME

CFG = dict(
    data        = str(DATA_YAML),
    epochs      = 40,
    batch       = 16,
    imgsz       = 960,            # ← upgrade từ 640
    lr0         = 0.005,          # ← thấp hơn 0.01 vì imgsz lớn + data nhỏ
    lrf         = 0.01,
    cos_lr      = True,           # ← cosine LR schedule
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
    # Augmentation nâng cao cho dense scenes
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
        print(f"[FRESH] init từ COCO {INIT_WEIGHTS}, imgsz=960")
        model = YOLO(str(INIT_WEIGHTS))
        model.train(**CFG)

    print(f"\n[DONE] Best: {best_pt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
