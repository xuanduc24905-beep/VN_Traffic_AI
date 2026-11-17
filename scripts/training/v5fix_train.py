"""Train detector V5fix voi backbone dong bang (freeze=10).

  D1  v5fix_d1_freeze10   : init yolov8s.pt COCO, freeze=10 suot 60 epoch, recipe = V5+clean
  D2  v5fix_d2_unfreeze   : tu trang thai D1 sau 50 epoch (epoch50_done.pt), mo khoa toan mang,
                            10 epoch, lr0=0.0002 (mosaic tat giong 10 epoch cuoi cua D1)

  C   v5fix_c_nofreeze    : DOI CHUNG — y het D1 nhung freeze=None (train toan mang tu COCO).
                            So C vs D1 = tach rieng tac dong cua viec dong bang backbone.
  M1  v5fixmio_d1_freeze10: y het D1 nhung data = V5fix + 3.000 anh MIO-TCD (data_v5fix_mio.yaml).
                            So M1 vs D1 = tac dong cua MIO-TCD.

Chay:
    python scripts/training/v5fix_train.py --run d1
    python scripts/training/v5fix_train.py --run d2
    python scripts/training/v5fix_train.py --run c
    python scripts/training/v5fix_train.py --run m1
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT / "scripts" / "training"))
from v5plus_clean_train import RECIPE  # noqa: E402  (cung recipe voi train_v8s_v5pc_full_s42)

DATA = ROOT / "data" / "data_v5fix.yaml"
RUNS = ROOT / "runs" / "detect"
D1, D2, C, M1 = "v5fix_d1_freeze10", "v5fix_d2_unfreeze", "v5fix_c_nofreeze", "v5fixmio_d1_freeze10"
DATA_MIO = ROOT / "data" / "data_v5fix_mio.yaml"
SEED = 42


def run_d1():
    from ultralytics import YOLO

    def keep_epoch50(trainer):
        if trainer.epoch == 49:  # 0-based -> vua xong 50 epoch (con mosaic)
            shutil.copyfile(trainer.last, trainer.wdir / "epoch50_done.pt")

    last = RUNS / D1 / "weights" / "last.pt"
    if last.exists():
        model = YOLO(str(last))
        model.add_callback("on_model_save", keep_epoch50)
        model.train(resume=True)
        return
    model = YOLO(str(ROOT / "yolov8s.pt"))
    model.add_callback("on_model_save", keep_epoch50)
    model.train(data=str(DATA), seed=SEED, patience=RECIPE["epochs"], freeze=10,
                project=str(RUNS), name=D1, **RECIPE)


def run_d2():
    from ultralytics import YOLO

    init = RUNS / D1 / "weights" / "epoch50_done.pt"
    assert init.exists(), f"Thieu {init} — chay D1 truoc"
    recipe = dict(RECIPE, epochs=10, lr0=0.0002, lrf=0.1, warmup_epochs=0.0, close_mosaic=10, save_period=-1)
    YOLO(str(init)).train(data=str(DATA), seed=SEED, patience=10, freeze=None,
                          project=str(RUNS), name=D2, **recipe)


def run_c():
    from ultralytics import YOLO

    last = RUNS / C / "weights" / "last.pt"
    if last.exists():
        YOLO(str(last)).train(resume=True)
        return
    YOLO(str(ROOT / "yolov8s.pt")).train(data=str(DATA), seed=SEED, patience=RECIPE["epochs"], freeze=None,
                                         project=str(RUNS), name=C, **RECIPE)


def run_m1():
    from ultralytics import YOLO

    last = RUNS / M1 / "weights" / "last.pt"
    if last.exists():
        YOLO(str(last)).train(resume=True)
        return
    YOLO(str(ROOT / "yolov8s.pt")).train(data=str(DATA_MIO), seed=SEED, patience=RECIPE["epochs"], freeze=10,
                                         project=str(RUNS), name=M1, **RECIPE)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["d1", "d2", "c", "m1"], required=True)
    {"d1": run_d1, "d2": run_d2, "c": run_c, "m1": run_m1}[ap.parse_args().run]()
