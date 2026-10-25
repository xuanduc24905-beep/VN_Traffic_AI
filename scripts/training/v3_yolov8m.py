"""Train yolov8m VN-only — model lớn hơn V3-s để test scaling.

Hypothesis: yolov8m (25.9M params, 2.3× yolov8s) có capacity cao hơn
→ cải thiện mọi class, đặc biệt motorcycle (class khó nhất của V3-s với
mAP50-95 chỉ 0.584).

Config: FRESH từ yolov8m.pt (COCO), imgsz=640, batch=10 (fit 12GB VRAM),
40 epoch, config conservative giống V3-s.

Expected:
- mAP50: 0.951 (V3-s) → ~0.96-0.97
- Motorcycle mAP50-95: 0.584 → ~0.60-0.63
- FPS giảm từ 72 → ~35 (tradeoff)

Smart start: SKIP / RESUME / FRESH như các script khác.
Chạy:
    python scripts/training/v3_yolov8m.py
"""
import sys
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path("/home/xuand/vn_traffic_ai").resolve()
RUNS_DIR     = PROJECT_ROOT / "runs" / "detect"
DATA_YAML    = PROJECT_ROOT / "data" / "data_v3.yaml"

# yolov8m pretrained (COCO) — ở root
INIT_WEIGHTS = PROJECT_ROOT / "yolov8m.pt"
assert INIT_WEIGHTS.exists(), f"Thiếu {INIT_WEIGHTS}"
assert DATA_YAML.exists(), f"Thiếu {DATA_YAML}"

RUN_NAME = "train_v8m_v3_vn"
RUN_DIR  = RUNS_DIR / RUN_NAME

CFG = dict(
    data        = str(DATA_YAML),
    epochs      = 40,
    batch       = 10,            # yolov8m 2.3× heavier → giảm từ 16 (v8s) → 10
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
        print(f"[FRESH] init từ {INIT_WEIGHTS}, imgsz={CFG['imgsz']}, batch={CFG['batch']}")
        model = YOLO(str(INIT_WEIGHTS))
        model.train(**CFG)

    print(f"\n[DONE] Best: {best_pt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
