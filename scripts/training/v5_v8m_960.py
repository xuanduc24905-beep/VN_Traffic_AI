"""Fine-tune YOLOv8m trên V5 với cấu hình cho xe nhỏ và dữ liệu ít.

Config:
- imgsz=960 để giữ chi tiết xe nhỏ (motorcycle có box median ~0.2% ảnh)
- AdamW với learning rate thấp để fine-tune từ pretrained weights
- Mosaic/scale vừa phải để tránh làm xe nhỏ khó thấy hơn
- Không dùng dropout: Ultralytics chỉ áp dụng dropout cho classification

Smart start: SKIP / RESUME / FRESH.
"""
import sys
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path("/home/xuand/vn_traffic_ai").resolve()
RUNS_DIR     = PROJECT_ROOT / "runs" / "detect"
DATA_YAML    = PROJECT_ROOT / "data" / "data_v5.yaml"
INIT_WEIGHTS = PROJECT_ROOT / "yolov8m.pt"

assert INIT_WEIGHTS.exists(), f"Thiếu {INIT_WEIGHTS}"
assert DATA_YAML.exists(), f"Thiếu {DATA_YAML}"

RUN_NAME = "train_v8m_v5_960"
RUN_DIR  = RUNS_DIR / RUN_NAME

CFG = dict(
    data        = str(DATA_YAML),
    epochs      = 60,
    batch       = 4,
    imgsz       = 960,
    lr0         = 0.001,
    lrf         = 0.01,
    cos_lr      = True,
    patience    = 20,
    optimizer   = "AdamW",
    weight_decay= 0.0005,
    mosaic      = 0.7,
    close_mosaic= 10,
    scale       = 0.3,
    seed        = 42,
    workers     = 8,
    cache       = "disk",
    amp         = True,
    save_period = 5,
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
        print(
            f"[FRESH] fine-tune từ {INIT_WEIGHTS}, "
            f"imgsz={CFG['imgsz']}, batch={CFG['batch']}, "
            f"optimizer={CFG['optimizer']}, lr0={CFG['lr0']}"
        )
        model = YOLO(str(INIT_WEIGHTS))
        model.train(**CFG)

    print(f"\n[DONE] Best: {best_pt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
