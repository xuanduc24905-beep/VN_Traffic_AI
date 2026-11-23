"""Tính Fisher information (empirical, theo từng ảnh) của YOLOv8s COCO cho EWC.

F_i = trung bình qua ảnh COCO của (∂ loss / ∂ w_i)², dùng loss detection của Ultralytics với nhãn COCO thật
(coco128, 128 ảnh; 3 lượt có tăng cường dữ liệu → ~384 mẫu gradient). Chỉ giữ tham số layer 0..21
(backbone + neck, phần dùng chung với model 4 lớp), rồi chuẩn hoá về trung bình 1 để λ so được với L2-SP.

Cách làm: DetectionTrainer với batch 1, nbs 1 (mỗi bước = 1 ảnh), lr0 = 0 và thay optimizer_step bằng việc
cộng dồn grad² — không cập nhật trọng số.

Chạy: python scripts/training/chong_quen_fisher.py   → weights/chong_quen_fisher.pt
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "weights" / "chong_quen_fisher.pt"
HEAD_IDX = 22
ACC: dict = {}
N = [0]


def penalized(name: str) -> bool:
    parts = name.split(".")
    return parts[0] == "model" and parts[1].isdigit() and int(parts[1]) < HEAD_IDX


def main():
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer

    class FisherTrainer(DetectionTrainer):
        def optimizer_step(self):
            self.scaler.unscale_(self.optimizer)
            for n, p in self.model.named_parameters():
                if penalized(n) and p.grad is not None:
                    g2 = p.grad.detach().float() ** 2
                    ACC[n] = ACC[n] + g2 if n in ACC else g2.clone()
            N[0] += 1
            self.scaler.update()
            self.optimizer.zero_grad()

    YOLO(str(ROOT / "yolov8s.pt")).train(
        data="coco128.yaml", trainer=FisherTrainer, epochs=3, batch=1, nbs=1, imgsz=640, lr0=0.0, lrf=1.0,
        warmup_epochs=0, val=False, plots=False, save=False, workers=2, amp=False, freeze=None,
        project=str(ROOT / "runs" / "detect"), name="cq_fisher_tmp", exist_ok=True, seed=0, deterministic=True)
    fis = {n: (v / N[0]).cpu() for n, v in ACC.items()}
    mean = torch.cat([v.flatten() for v in fis.values()]).mean()
    fis = {n: v / mean for n, v in fis.items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fis, OUT)
    allv = torch.cat([v.flatten() for v in fis.values()])
    info = {"n_samples": N[0], "n_tensors": len(fis), "n_params": int(allv.numel()),
            "quantiles_normalized": {q: float(allv.quantile(q)) for q in (0.1, 0.5, 0.9, 0.99)} if allv.numel() < 16e6 else None}
    (OUT.with_suffix(".json")).write_text(json.dumps(info, indent=1))
    print("[fisher]", info, flush=True)


if __name__ == "__main__":
    main()
