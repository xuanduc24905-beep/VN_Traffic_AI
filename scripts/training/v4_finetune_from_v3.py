"""V4 fine-tune — warm từ V3 best.pt, train tiếp trên merged_v4 (có vn_cctv).

Hypothesis: V3 đã học tốt feature VN. Fine-tune nhẹ với data mới (vn_cctv)
+ lr thấp sẽ cải thiện mà không mất feature cũ.

Config:
- Warm từ: weights/v3_vn_best.pt (V3 mAP50=0.951)
- Dataset: data/data_v4.yaml (merged_v4 = V3 + vn_cctv, 49.7k ảnh)
- epochs=15 (ít vì chỉ fine-tune), patience=8
- lr0=0.001 (thấp 10× so với fresh train)
- batch=16, imgsz=640
- SGD cosine

Expected:
- mAP50 (val VN): 0.951 → 0.96-0.97 (+0.5-2%)
- Thời gian: ~2-2.5h (vs V4 fresh 6-7h)

Smart start: SKIP / RESUME / FRESH.
"""
import sys
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path("/home/xuand/vn_traffic_ai").resolve()
RUNS_DIR     = PROJECT_ROOT / "runs" / "detect"
DATA_YAML    = PROJECT_ROOT / "data" / "data_v4.yaml"

# Warm từ V3 best.pt (không phải COCO)
INIT_WEIGHTS = PROJECT_ROOT / "weights" / "v3_vn_best.pt"
assert INIT_WEIGHTS.exists(), f"Thiếu {INIT_WEIGHTS}"
assert DATA_YAML.exists(), f"Thiếu {DATA_YAML}. Chạy merge_v4.py trước."

RUN_NAME = "train_v8s_v4_finetune"
RUN_DIR  = RUNS_DIR / RUN_NAME

CFG = dict(
    data        = str(DATA_YAML),
    epochs      = 15,            # fine-tune ngắn
    batch       = 16,
    imgsz       = 640,
    lr0         = 0.001,          # thấp 10× vì fine-tune
    lrf         = 0.01,
    patience    = 8,             # early stop sớm hơn
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
        print(f"[FRESH FINETUNE] warm từ {INIT_WEIGHTS}, lr0={CFG['lr0']}")
        model = YOLO(str(INIT_WEIGHTS))
        model.train(**CFG)

    print(f"\n[DONE] Best: {best_pt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
