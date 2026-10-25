"""Experiment B: Boost classification loss weight — fix underfit cho class ít data.

Hypothesis: motorcycle 0.85% bbox → underfit (mAP@0.5:0.95 = 0.609, thấp nhất).
Tăng classification loss weight `cls=2.0` (default 0.5) → model focus hơn vào
discrimination giữa các class → giúp class khó (motorcycle) + reduce class
confusion (truck vs van).

NOTE: Ultralytics v8 KHÔNG support per-class weight native. `cls=X` là global
weight cho tất cả class. Để true per-class weighting phải patch v8DetectionLoss
trong source. Script này dùng global cls boost — "easy version" nhưng vẫn
tendency tốt hơn.

Expected:
- motorcycle mAP@0.5:0.95: +3-7% (focus hơn vào class khó)
- Class confusion rate (van→truck): giảm
- Có thể trade-off nhẹ: box localization precision (vì weight shifted sang cls)

Chạy:
    cd /home/xuand/vn_traffic_ai
    /home/xuand/miniconda3/envs/vehicle_env/bin/python scripts/week2/03_train_class_weights.py
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

RUN_NAME = "train_v8s_cls_boost"
RUN_DIR  = RUNS_DIR / RUN_NAME

CFG = dict(
    data        = str(DATA_YAML),
    epochs      = 20,
    batch       = 24,
    imgsz       = 512,
    lr0         = 0.001,
    lrf         = 0.1,
    patience    = 6,
    optimizer   = "SGD",
    # <<< KEY: boost classification loss weight
    cls         = 2.0,              # default 0.5 → 2.0 (4× focus vào cls)
    box         = 7.5,              # giữ default
    dfl         = 1.5,              # giữ default
    # Augmentation (moderate)
    scale       = 0.5,
    mosaic      = 1.0,
    close_mosaic = 5,
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
