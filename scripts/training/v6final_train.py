"""Train mô hình cuối cho project cuối kì trên bộ V6final (data/data_v6final.yaml).

  F1  v6final_f1_freeze10 : init yolov8s.pt COCO, freeze=10, 30 epoch (mô hình cuối)
  F2  v6final_f2_nofreeze : y hệt F1 nhưng không freeze (đối chứng)
  F1M v6finalmio_f1_freeze10 : như F1 trên V6final + 4.500 ảnh MIO-TCD mới (data/data_v6final_mio.yaml)
  F2M v6finalmio_f2_nofreeze : như F2 trên cùng bộ đó
  S1  v6split_f1_freeze10 : như F1M nhưng trên V6split (val/test chia lại theo camera, data/data_v6split.yaml)
  S2  v6split_f2_nofreeze : như F2M trên V6split
Recipe còn lại giống V5+clean / V5fix (RECIPE trong v5plus_clean_train.py), chỉ đổi số epoch thành 30
vì các run trước đều đạt đỉnh val quanh epoch 20.

Chạy:
    python scripts/training/v6final_train.py --run f1
    python scripts/training/v6final_train.py --run f2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "training"))
from v5plus_clean_train import RECIPE  # noqa: E402

DATA = ROOT / "data" / "data_v6final.yaml"
RUNS = ROOT / "runs" / "detect"
SEED = 42
EPOCHS = 30
DATA_MIO = ROOT / "data" / "data_v6final_mio.yaml"
DATA_SPLIT = ROOT / "data" / "data_v6split.yaml"
RUNS_CFG = {"f1": ("v6final_f1_freeze10", 10, DATA), "f2": ("v6final_f2_nofreeze", None, DATA),
            "f1m": ("v6finalmio_f1_freeze10", 10, DATA_MIO), "f2m": ("v6finalmio_f2_nofreeze", None, DATA_MIO),
            "s1": ("v6split_f1_freeze10", 10, DATA_SPLIT), "s2": ("v6split_f2_nofreeze", None, DATA_SPLIT)}


def run(key: str):
    from ultralytics import YOLO

    name, freeze, data = RUNS_CFG[key]
    last = RUNS / name / "weights" / "last.pt"
    if last.exists():
        YOLO(str(last)).train(resume=True)
        return
    recipe = dict(RECIPE, epochs=EPOCHS)
    YOLO(str(ROOT / "yolov8s.pt")).train(data=str(data), seed=SEED, patience=EPOCHS, freeze=freeze,
                                         project=str(RUNS), name=name, **recipe)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=list(RUNS_CFG), required=True)
    run(ap.parse_args().run)
