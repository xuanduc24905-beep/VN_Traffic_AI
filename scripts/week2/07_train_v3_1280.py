"""Train V3 với imgsz=1280 — target: tăng mAP50-95 motorcycle (từ 0.585).

Lý do: motorcycle VN thường nhỏ (xa camera). imgsz=640 có thể crop/downscale
mất feature. imgsz=1280 giữ nhiều pixel hơn cho bbox nhỏ → localize tốt hơn.

Trade-off:
- 4× pixel so với 640 → batch phải giảm (16 → 8) + thời gian ~2× (ước 5-7h).
- Dự kiến mAP50 không tăng nhiều (đã 0.95), chủ yếu cải thiện mAP50-95.

Smart start: SKIP / RESUME / FRESH như các script khác.
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
assert DATA_YAML.exists(), f"Thiếu {DATA_YAML}"

RUN_NAME = "train_v8s_v3_1280"
RUN_DIR  = RUNS_DIR / RUN_NAME

CFG = dict(
    data        = str(DATA_YAML),
    epochs      = 40,
    batch       = 8,            # giảm vì imgsz 1280 (từ 16 @ 640)
    imgsz       = 1280,
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
