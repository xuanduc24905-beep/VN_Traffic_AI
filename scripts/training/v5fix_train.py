"""Train detector V5fix voi backbone dong bang (freeze=10).

  D1  v5fix_d1_freeze10   : init yolov8s.pt COCO, freeze=10 suot 60 epoch, recipe = V5+clean
  D2  v5fix_d2_unfreeze   : tu trang thai D1 sau 50 epoch (epoch50_done.pt), mo khoa toan mang,
                            10 epoch, lr0=0.0002 (mosaic tat giong 10 epoch cuoi cua D1)

  C   v5fix_c_nofreeze    : DOI CHUNG — y het D1 nhung freeze=None (train toan mang tu COCO).
                            So C vs D1 = tach rieng tac dong cua viec dong bang backbone.
  M1  v5fixmio_d1_freeze10: y het D1 nhung data = V5fix + 3.000 anh MIO-TCD (data_v5fix_mio.yaml).
                            So M1 vs D1 = tac dong cua MIO-TCD.
  CT  v5fixmio_ctmask_probe11: y het M1 (lich 60 epoch) nhung anh Can Tho to xam 35% trai;
                            DUNG sau 11 epoch de so voi epoch10.pt cua M1 (cung moc, cung lich lr).

Chay:
    python scripts/training/v5fix_train.py --run d1
    python scripts/training/v5fix_train.py --run d2
    python scripts/training/v5fix_train.py --run c
    python scripts/training/v5fix_train.py --run m1
    python scripts/training/v5fix_train.py --run ct
    python scripts/training/v5fix_train.py --run mk
  MK  v5fixmio_mask_probe15: y het M1 (lich 60 epoch) nhung to xam MOI xe thieu nhan o 4 nguon VN
                            (data_v5fix_mio_mask.yaml); DUNG sau 15 epoch. So voi M1 cung moc.
  MK30 v5fixmio_mask_e30  : mo hinh hoan chinh tren data che xam, lich 30 epoch (cos lr + tat mosaic
                            10 epoch cuoi), con lai = RECIPE. 30 epoch vi M1 dat dinh val o epoch 20.
    python scripts/training/v5fix_train.py --run mk30
  MK15C v5fixmio_mask_e15c : TIEP TUC tu last.pt cua probe 15 epoch (MK) them 15 epoch voi lich moi:
                            lr0 = lr tai epoch 15 cua lich 60 (0.00855) -> cos ve 1e-4, khong warmup,
                            tat mosaic 10 epoch cuoi. Tong 30 epoch, khong train lai tu dau.
    python scripts/training/v5fix_train.py --run mk15c
  MK60 v5fixmio_mask_e60   : CONG THUC Y HET M1 (lich 60 epoch, seed 42, freeze=10) tren data che xam.
                            Khoi dong tu epoch10.pt cua probe MK (cung lich, con optimizer) -> resume toi 60.
    python scripts/training/v5fix_train.py --run mk60
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
CT = "v5fixmio_ctmask_probe11"
DATA_CT = ROOT / "data" / "data_v5fix_mio_ctmask.yaml"
PROBE_LAST_EPOCH = 10  # 0-based -> dung sau 11 epoch, trung moc epoch10.pt cua M1
MK = "v5fixmio_mask_probe15"
DATA_MK = ROOT / "data" / "data_v5fix_mio_mask.yaml"
MK30 = "v5fixmio_mask_e30"
MK15C = "v5fixmio_mask_e15c"
MK60 = "v5fixmio_mask_e60"
MK_LAST_EPOCH = 14  # 0-based -> dung sau 15 epoch (van luu epoch10.pt de so dung moc voi M1)
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


def run_ct():
    from ultralytics import YOLO

    def stop_probe(trainer):
        if trainer.epoch >= PROBE_LAST_EPOCH:
            trainer.stop = True  # giu nguyen lich 60 epoch, chi dung som

    last = RUNS / CT / "weights" / "last.pt"
    model = YOLO(str(last)) if last.exists() else YOLO(str(ROOT / "yolov8s.pt"))
    model.add_callback("on_fit_epoch_end", stop_probe)
    if last.exists():
        model.train(resume=True)
        return
    model.train(data=str(DATA_CT), seed=SEED, patience=RECIPE["epochs"], freeze=10,
                project=str(RUNS), name=CT, **RECIPE)


def run_mk():
    from ultralytics import YOLO

    def stop_probe(trainer):
        if trainer.epoch >= MK_LAST_EPOCH:
            trainer.stop = True  # giu nguyen lich 60 epoch, chi dung som

    last = RUNS / MK / "weights" / "last.pt"
    model = YOLO(str(last)) if last.exists() else YOLO(str(ROOT / "yolov8s.pt"))
    model.add_callback("on_fit_epoch_end", stop_probe)
    if last.exists():
        model.train(resume=True)
        return
    model.train(data=str(DATA_MK), seed=SEED, patience=RECIPE["epochs"], freeze=10,
                project=str(RUNS), name=MK, **RECIPE)


def run_mk30():
    from ultralytics import YOLO

    last = RUNS / MK30 / "weights" / "last.pt"
    if last.exists():
        YOLO(str(last)).train(resume=True)
        return
    recipe = dict(RECIPE, epochs=30)
    YOLO(str(ROOT / "yolov8s.pt")).train(data=str(DATA_MK), seed=SEED, patience=30, freeze=10,
                                         project=str(RUNS), name=MK30, **recipe)


def run_mk15c():
    from ultralytics import YOLO

    last = RUNS / MK15C / "weights" / "last.pt"
    if last.exists():
        YOLO(str(last)).train(resume=True)
        return
    init = RUNS / MK / "weights" / "last.pt"
    assert init.exists(), f"Thieu {init} — chay --run mk truoc"
    lr0 = 0.00855  # = lr cua lich 60 epoch tai epoch 15
    recipe = dict(RECIPE, epochs=15, lr0=lr0, lrf=1e-4 / lr0, warmup_epochs=0.0, close_mosaic=10)
    YOLO(str(init)).train(data=str(DATA_MK), seed=SEED, patience=15, freeze=10,
                          project=str(RUNS), name=MK15C, **recipe)


def run_mk60():
    import csv
    import torch
    from ultralytics import YOLO

    new = RUNS / MK60
    last = new / "weights" / "last.pt"
    if not last.exists():
        src = RUNS / MK / "weights" / "epoch10.pt"  # epoch index 10 = xong 11 epoch, con optimizer + EMA
        ck = torch.load(src, map_location="cpu", weights_only=False)
        assert ck.get("optimizer") is not None and ck["train_args"]["epochs"] == 60
        (new / "weights").mkdir(parents=True, exist_ok=True)
        ck["train_args"].update(name=MK60, save_dir=str(new))
        torch.save(ck, last)
        rows = list(csv.reader(open(RUNS / MK / "results.csv")))
        with open(new / "results.csv", "w", newline="") as f:  # giu header + 11 epoch dau
            csv.writer(f).writerows(rows[: ck["epoch"] + 2])
        shutil.copyfile(RUNS / MK / "args.yaml", new / "args.yaml")
    YOLO(str(last)).train(resume=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["d1", "d2", "c", "m1", "ct", "mk", "mk30", "mk15c", "mk60"], required=True)
    {"d1": run_d1, "d2": run_d2, "c": run_c, "m1": run_m1, "ct": run_ct, "mk": run_mk, "mk30": run_mk30, "mk15c": run_mk15c, "mk60": run_mk60}[ap.parse_args().run]()
