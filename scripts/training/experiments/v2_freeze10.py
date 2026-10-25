"""Experiment A: freeze=10 — Transfer learning strategy.

Hypothesis: Full fine-tune bị over-adapt vào DETRAC (TQ). Freeze backbone COCO
(10 layer đầu) → chỉ train neck + head → giữ feature extractor generic → tốt
hơn cho video VN out-of-distribution.

Expected:
- mAP overall có thể giảm nhẹ (~1%)
- Nhưng Recall trên scene mới (video thật) tốt hơn
- Train nhanh hơn ~30% (ít params update)

Chạy:
    cd /home/xuand/vn_traffic_ai
    /home/xuand/miniconda3/envs/vehicle_env/bin/python scripts/week2/02_train_freeze10.py
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

RUN_NAME = "train_v8s_freeze10"
RUN_DIR  = RUNS_DIR / RUN_NAME

CFG = dict(
    data        = str(DATA_YAML),
    epochs      = 20,
    batch       = 24,              # có thể cao hơn V2 vì freeze → ít memory
    imgsz       = 512,
    lr0         = 0.001,            # thấp vì warm-start
    lrf         = 0.1,
    patience    = 6,
    optimizer   = "SGD",
    freeze      = 10,               # <<< KEY: freeze 10 layer đầu backbone
    # Augmentation (giữ moderate, không aggressive như V2)
    scale       = 0.5,
    mosaic      = 1.0,
    close_mosaic = 5,
    mixup       = 0.0,              # tắt mixup, giảm overhead
    # Infra
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
          f"best_pt={'YES' if best_pt.exists() else 'NO'}")

    if done_epochs >= target and best_pt.exists():
        print(f"[SKIP] Đã train xong. Best: {best_pt}")
        return 0

    if last_pt.exists() and done_epochs > 0 and results_csv.exists():
        print(f"[RESUME] từ epoch {done_epochs}/{target}")
        model = YOLO(str(last_pt))
        model.train(resume=True)
    else:
        if last_pt.exists():
            print(f"[WARN] last.pt orphan → ignore")
        print(f"[FRESH] warm-start từ {BASELINE_WEIGHTS}")
        model = YOLO(str(BASELINE_WEIGHTS))
        model.train(**CFG)

    print(f"\n[DONE] Best: {best_pt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
