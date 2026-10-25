"""V6 — train trên V5 clean + pseudo-labeled vn_cctv (self-training round 1).

Config: same V5+ (yolov8s @ 960) để fair comparison.
Expected: V5+ ~0.71 → V6 ~0.73-0.77 overall + robustness tăng trên vn_cctv OOD.

Smart start: SKIP / RESUME / FRESH.
"""
import sys
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path("/home/xuand/vn_traffic_ai").resolve()
RUNS_DIR     = PROJECT_ROOT / "runs" / "detect"
DATA_YAML    = PROJECT_ROOT / "data" / "data_v6.yaml"
INIT_WEIGHTS = PROJECT_ROOT / "yolov8s.pt"

assert INIT_WEIGHTS.exists(), f"Thiếu {INIT_WEIGHTS}"
assert DATA_YAML.exists(), f"Thiếu {DATA_YAML}. Chạy build_v6_merged.py trước."

RUN_NAME = "train_v8s_v6_pseudo"
RUN_DIR  = RUNS_DIR / RUN_NAME

CFG = dict(
    data        = str(DATA_YAML),
    epochs      = 40,
    batch       = 16,
    imgsz       = 960,
    lr0         = 0.005,
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
        print(f"[FRESH] init từ COCO {INIT_WEIGHTS}")
        model = YOLO(str(INIT_WEIGHTS))
        model.train(**CFG)

    print(f"\n[DONE] Best: {best_pt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
