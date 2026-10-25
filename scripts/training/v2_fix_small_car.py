"""Train V2 (small_car targeted) — standalone, no notebook/VSCode needed.

Smart start:
- SKIP nếu đã train đủ epoch (dùng best.pt sẵn có)
- RESUME nếu có last.pt + results.csv > 0 epoch
- FRESH warm-start từ baseline nếu chưa có gì

Chạy:
    cd /home/xuand/vn_traffic_ai
    conda activate vehicle_env   # hoặc dùng python đầy đủ path
    python scripts/week2/01_train_v2_small_car.py
"""
import sys
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path("/home/xuand/vn_traffic_ai").resolve()
RUNS_DIR     = PROJECT_ROOT / "runs" / "detect"
DATA_YAML    = PROJECT_ROOT / "data" / "data.yaml"

BASELINE_WEIGHTS = RUNS_DIR / "train_v8s_1round_merged_v2" / "weights" / "best.pt"
assert BASELINE_WEIGHTS.exists(), f"Thiếu baseline {BASELINE_WEIGHTS}"

RUN_NAME = "train_v8s_fix_v2_small_car"
RUN_DIR  = RUNS_DIR / RUN_NAME

CFG = dict(
    data        = str(DATA_YAML),
    epochs      = 20,
    batch       = 20,
    imgsz       = 512,
    lr0         = 0.003,
    lrf         = 0.1,
    patience    = 6,
    optimizer   = "SGD",
    scale       = 0.9,
    mosaic      = 1.0,
    close_mosaic = 0,
    multi_scale = True,
    mixup       = 0.15,
    seed        = 0,
    workers     = 8,
    cache       = "disk",
    amp         = True,
    save_period = 1,
    project     = str(RUNS_DIR),
    name        = RUN_NAME,
    exist_ok    = True,
    verbose     = True,
)


def main():
    last_pt     = RUN_DIR / "weights" / "last.pt"
    best_pt     = RUN_DIR / "weights" / "best.pt"
    results_csv = RUN_DIR / "results.csv"

    done_epochs = 0
    if results_csv.exists():
        try:
            done_epochs = len(pd.read_csv(results_csv))
        except Exception as e:
            print(f"[WARN] results.csv không đọc được: {e}")

    target = CFG["epochs"]
    print(f"[STATE] done={done_epochs}/{target}, "
          f"last_pt={'YES' if last_pt.exists() else 'NO'}, "
          f"best_pt={'YES' if best_pt.exists() else 'NO'}, "
          f"results_csv={'YES' if results_csv.exists() else 'NO'}")

    if done_epochs >= target and best_pt.exists():
        print(f"[SKIP] Đã train xong {done_epochs} epoch. Best: {best_pt}")
        return 0

    if last_pt.exists() and done_epochs > 0 and results_csv.exists():
        print(f"[RESUME] từ epoch {done_epochs}/{target} dùng last.pt")
        model = YOLO(str(last_pt))
        model.train(resume=True)
    else:
        if last_pt.exists():
            print(f"[WARN] last.pt tồn tại nhưng ko match results.csv → ignore")
        print(f"[FRESH] warm-start từ {BASELINE_WEIGHTS}")
        model = YOLO(str(BASELINE_WEIGHTS))
        model.train(**CFG)

    print(f"\n[DONE] Best: {best_pt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
