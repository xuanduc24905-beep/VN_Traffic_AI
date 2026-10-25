"""Experiment C: YOLOv8m — Architecture scaling.

Hypothesis: Model lớn hơn (v8m: 25M params vs v8s: 11M) → mAP cao hơn 2-3%,
nhưng FPS giảm ~50%. Trade-off giữa accuracy và speed cho realtime VN traffic.

Expected:
- mAP@0.5 ~0.88 (vs v8s baseline 0.858)
- FPS: ~35 (vs v8s ~72)
- VRAM: ~16GB train (có thể OOM nếu batch quá cao)

Chạy from scratch (không warm-start — khác architecture), ~5-7h trên RTX 3500 Ada.

⚠️ CHẠY ĐÊM: tốn thời gian nhất trong 3 experiment Tuần 2. Monitor nhiệt độ.

Chạy:
    cd /home/xuand/vn_traffic_ai
    /home/xuand/miniconda3/envs/vehicle_env/bin/python scripts/week2/04_train_yolov8m.py
"""
import sys
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path("/home/xuand/vn_traffic_ai").resolve()
RUNS_DIR     = PROJECT_ROOT / "runs" / "detect"
DATA_YAML    = PROJECT_ROOT / "data" / "data.yaml"

RUN_NAME = "train_v8m_merged"
RUN_DIR  = RUNS_DIR / RUN_NAME

CFG = dict(
    data        = str(DATA_YAML),
    epochs      = 30,              # ít hơn baseline (50) vì nặng
    batch       = 16,              # thấp hơn (v8m lớn hơn, VRAM 12GB giới hạn)
    imgsz       = 512,
    lr0         = 0.01,            # fresh training → lr cao như baseline
    lrf         = 0.01,
    patience    = 8,
    optimizer   = "SGD",
    close_mosaic = 10,
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
        print(f"[FRESH] train from scratch với yolov8m.pt (COCO pretrained)")
        model = YOLO("yolov8m.pt")  # Ultralytics tự download nếu chưa có
        model.train(**CFG)

    print(f"\n[DONE] Best: {best_pt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
