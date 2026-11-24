"""Bước 3 (LR thấp) và bước 5 (L2-SP, EWC) — chống quên kiến thức COCO khi fine-tune YOLOv8s.

Mọi run dùng đúng cấu hình S2 (V6split, không freeze, 30 epoch, imgsz 960, RECIPE) để so được với S1/S2.

  lowlr    : như S2 nhưng lr0 = 0.002 (S2 dùng 0.01)
  l2sp_lo/mid/hi : loss + λ·Σ(w − w_COCO)² trên backbone + neck (layer 0..21), λ = 0.1 / 1 / 10 × λ_c
  ewc      : loss + λ_mid·Σ F̂·(w − w_COCO)²,  F̂ = Fisher (tính trên ảnh COCO) chuẩn hoá về trung bình 1
  l2sp_coco5k : như l2sp_mid nhưng train trên V6split + 5.000 ảnh COCO (replay, build_v6split_coco.py)
  probe    : chạy ngắn với λ = 0 để đo λ_c (xem dưới) và kiểm tra trainer

Cài đặt: phạt được cộng trực tiếp vào gradient trong optimizer_step (sau khi unscale AMP, trước clip/step):
  grad += 2λ·F̂·(w − w_COCO). Không đổi cấu trúc model → best.pt mở được ở mọi nơi như bình thường.
Weight decay mặc định vẫn giữ.
λ_c = (trung bình |grad dữ liệu| ở backbone) / (2 · trung bình |w_S2 − w_COCO|): ở mức trôi trọng số của S2,
lực kéo về COCO bằng lực của dữ liệu. Đo bằng run probe → weights/chong_quen_lambda.json.
Fisher: scripts/training/chong_quen_fisher.py → weights/chong_quen_fisher.pt

Ghim Ultralytics 8.4.142 (optimizer_step được chép theo bản này).

Chạy: python scripts/training/chong_quen_train.py --run probe|lowlr|l2sp_lo|l2sp_mid|l2sp_hi|ewc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "training"))
from v5plus_clean_train import RECIPE  # noqa: E402

DATA = ROOT / "data" / "data_v6split.yaml"
DATA_COCO = ROOT / "data" / "data_v6split_coco.yaml"  # V6split + 5.000 ảnh COCO (replay)
RUNS = ROOT / "runs" / "detect"
COCO = ROOT / "yolov8s.pt"
S2_BEST = ROOT / "runs" / "detect" / "v6split_f2_nofreeze" / "weights" / "best.pt"
LAMBDA_JSON = ROOT / "weights" / "chong_quen_lambda.json"
FISHER_PT = ROOT / "weights" / "chong_quen_fisher.pt"
SEED, EPOCHS, HEAD_IDX = 42, 30, 22
ULTRA_VERSION = "8.4.142"

RUNS_CFG = {
    "lowlr": dict(name="cq_lowlr_s2", lam=None, lr0=0.002),
    "l2sp_lo": dict(name="cq_l2sp_lo", lam=0.1),
    "l2sp_mid": dict(name="cq_l2sp_mid", lam=1.0),
    "l2sp_hi": dict(name="cq_l2sp_hi", lam=10.0),
    "ewc": dict(name="cq_ewc", lam=1.0, fisher=True),
    "l2sp_coco5k": dict(name="cq_l2sp_mid_coco5k", lam=1.0, data=DATA_COCO),
    "probe": dict(name="cq_probe", lam=0.0, probe=True),
}


def penalized(name: str) -> bool:
    parts = name.split(".")
    return parts[0] == "model" and parts[1].isdigit() and int(parts[1]) < HEAD_IDX


def make_trainer(lam_abs: float, fisher: dict | None, probe: bool):
    from ultralytics.models.yolo.detect import DetectionTrainer

    class AnchorTrainer(DetectionTrainer):
        """DetectionTrainer + phạt kéo trọng số về COCO (L2-SP / EWC)."""

        _ref = None
        _stats = []

        def _anchor(self):
            if self._ref is None:
                from ultralytics import YOLO
                coco = YOLO(str(COCO)).model.float().state_dict()
                ref, fis = {}, {}
                for n, p in self.model.named_parameters():
                    if penalized(n) and n in coco and coco[n].shape == p.shape:
                        ref[n] = coco[n].to(p.device, torch.float32)
                        if fisher is not None:
                            fis[n] = fisher[n].to(p.device, torch.float32)
                type(self)._ref, type(self)._fis = ref, fis
            return self._ref, self._fis

        def optimizer_step(self):
            self.scaler.unscale_(self.optimizer)
            ref, fis = self._anchor()
            if probe:
                g = [p.grad.abs().mean().item() for n, p in self.model.named_parameters()
                     if n in ref and p.grad is not None]
                if g:
                    type(self)._stats.append(sum(g) / len(g))
            if lam_abs > 0:
                with torch.no_grad():
                    for n, p in self.model.named_parameters():
                        if n in ref and p.grad is not None:
                            d = p.detach().float() - ref[n]
                            if fis:
                                d = d * fis[n]
                            p.grad.add_((2.0 * lam_abs) * d.to(p.grad.dtype))
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)

    return AnchorTrainer


def drift_s2() -> float:
    from ultralytics import YOLO
    a = YOLO(str(S2_BEST)).model.float().state_dict()
    c = YOLO(str(COCO)).model.float().state_dict()
    tot, n = 0.0, 0
    for k, v in a.items():
        if penalized(k) and k in c and v.dtype.is_floating_point and c[k].shape == v.shape and \
                not k.endswith(("running_mean", "running_var", "num_batches_tracked")):
            tot += (v - c[k]).abs().sum().item()
            n += v.numel()
    return tot / n


def run(key: str):
    import ultralytics
    from ultralytics import YOLO

    assert ultralytics.__version__ == ULTRA_VERSION, f"Cần Ultralytics {ULTRA_VERSION}, đang có {ultralytics.__version__}"
    cfg = RUNS_CFG[key]
    recipe = dict(RECIPE, epochs=EPOCHS)
    if cfg.get("lr0"):
        recipe["lr0"] = cfg["lr0"]
    probe = cfg.get("probe", False)
    if probe:
        recipe.update(epochs=1, fraction=0.3, save_period=-1, plots=False, exist_ok=True)

    lam_abs, fisher = 0.0, None
    if cfg["lam"]:
        lam_c = json.loads(LAMBDA_JSON.read_text())["lambda_c"]
        lam_abs = cfg["lam"] * lam_c
    if cfg.get("fisher"):
        fisher = torch.load(FISHER_PT, map_location="cpu")
    trainer = make_trainer(lam_abs, fisher, probe) if (cfg["lam"] is not None) else None

    out = RUNS / cfg["name"]
    last = out / "weights" / "last.pt"
    if last.exists() and not probe:
        YOLO(str(last)).train(resume=True, trainer=trainer)
    else:
        YOLO(str(COCO)).train(data=str(cfg.get("data", DATA)), seed=SEED, patience=EPOCHS, freeze=None, project=str(RUNS),
                              name=cfg["name"], trainer=trainer, **recipe)
    if trainer is not None and not probe:
        (out / "chong_quen_cfg.json").write_text(json.dumps(
            {"lambda_rel": cfg["lam"], "lambda_abs": lam_abs, "fisher": bool(fisher),
             "penalized_layers": f"0..{HEAD_IDX - 1}"}, indent=1))
    if probe:
        st = trainer._stats
        st = st[len(st) // 4:] if len(st) > 8 else st
        g = sorted(st)[len(st) // 2]
        d = drift_s2()
        res = {"median_mean_abs_grad": g, "mean_abs_drift_S2": d, "lambda_c": g / (2 * d), "n_steps": len(trainer._stats)}
        LAMBDA_JSON.parent.mkdir(parents=True, exist_ok=True)
        LAMBDA_JSON.write_text(json.dumps(res, indent=1))
        print("[probe]", res, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=list(RUNS_CFG), required=True)
    run(ap.parse_args().run)
