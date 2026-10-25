"""Train V5 — honest baseline trên split video-aware (no leak).

Dataset: data/merged_v5 (train 2.8k, val 1.5k, test 1.5k — tất cả từ video khác nhau)
Weights init: yolov8s.pt (COCO pretrained)
Mục tiêu: lấy số liệu mAP THẬT, dự kiến 0.65-0.80 (so với 0.95 ảo trước đây)

Config:
- yolov8s, imgsz=640, batch=16
- 40 epochs, patience=15 (val set nhỏ → noise cao, không stop sớm)
- SGD cosine, lr0=0.01 (fresh train, không fine-tune)
- cache=disk vì ảnh ít hơn trước

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
assert DATA_YAML.exists(), f"Thiếu {DATA_YAML}. Chạy rebuild_v5_by_video.py trước."

RUN_NAME = "train_v8s_v5_honest"
RUN_DIR  = RUNS_DIR / RUN_NAME

CFG = dict(
    data        = str(DATA_YAML),
    epochs      = 40,
    batch       = 16,
    imgsz       = 640,
    lr0         = 0.01,
    lrf         = 0.01,
    patience    = 15,          # val nhỏ → noise, cần patience cao
    optimizer   = "SGD",
    seed        = 42,
    workers     = 8,
    cache       = "disk",
    amp         = True,
    save_period = 5,           # ít checkpoint hơn (val/test đã riêng)
    project     = str(RUNS_DIR),
    name        = RUN_NAME,
    exist_ok    = True,
    verbose     = True,
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
