"""Train V3 baseline — fresh baseline trên merged_v3 (~45k, 55% VN).

Config conservative:
- yolov8s.pt (COCO pretrained) → true baseline, không warm từ V2
- 40 epochs, imgsz=640, batch=16, lr0=0.01 (default SGD)
- Không multi_scale/mixup → chỉ default augment để baseline so sánh fair

Smart start:
- SKIP nếu đã train xong
- RESUME từ last.pt nếu có results.csv > 0
- FRESH init từ yolov8s.pt nếu chưa có

Chạy:
    cd /home/xuand/vn_traffic_ai
    python scripts/week2/06_train_v3_baseline.py
"""
import sys
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path("/home/xuand/vn_traffic_ai").resolve()
RUNS_DIR     = PROJECT_ROOT / "runs" / "detect"
DATA_YAML    = PROJECT_ROOT / "data" / "data_v3.yaml"

INIT_WEIGHTS = PROJECT_ROOT / "yolov8s.pt"
assert INIT_WEIGHTS.exists(), f"Thiếu {INIT_WEIGHTS}"
assert DATA_YAML.exists(), f"Thiếu {DATA_YAML}. Chạy merge_v3.py trước."

RUN_NAME = "train_v8s_v3_baseline"
RUN_DIR  = RUNS_DIR / RUN_NAME

CFG = dict(
    data        = str(DATA_YAML),
    epochs      = 40,
    batch       = 16,
    imgsz       = 640,
    lr0         = 0.01,
    lrf         = 0.01,
    patience    = 15,
    optimizer   = "SGD",
    seed        = 42,
    workers     = 8,
    cache       = "disk",
    amp         = True,
    save_period = 1,
    project     = str(RUNS_DIR),
    name        = RUN_NAME,
    exist_ok    = True,
    verbose     = True,
)


def main() -> int:
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
            print("[WARN] last.pt có nhưng ko match results.csv → ignore")
        print(f"[FRESH] init từ {INIT_WEIGHTS}")
        model = YOLO(str(INIT_WEIGHTS))
        model.train(**CFG)

    print(f"\n[DONE] Best: {best_pt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
