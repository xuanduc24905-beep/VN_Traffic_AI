"""Bước 6 — Trộn trọng số với COCO (WiSE-FT, Wortsman và cộng sự 2022), KHÔNG train lại.

Phiên bản 1 (MIX_UPTO = 22): trộn backbone + neck → HỎNG (mAP ≈ 0 ở α ≤ 0.5), vì head 4 lớp là phần mới,
chỉ khớp với đặc trưng neck đã fine-tune; BN running stats sau khi trộn cũng lệch.
Phiên bản 2 (mặc định): chỉ trộn BACKBONE (layer 0..9), giữ neck + head của bản fine-tune, rồi HIỆU CHỈNH
LẠI BN của riêng các layer được trộn (0..9): forward không gradient trên 256 ảnh train để tính lại running
mean/var; BN của neck/head giữ nguyên (hiệu chỉnh cả neck làm tăng hộp khổng lồ: 17 → 55 ở S2).
S1 freeze 10 layer đầu nên backbone S1 = COCO → trộn không đổi gì, chỉ áp cho S2.
  w = α · w_finetune + (1 − α) · w_COCO
α = 1 là bản fine-tune. Thử α ∈ {0.3, 0.5, 0.7, 0.85}.

Chạy: python scripts/training/chong_quen_wise.py
Kết quả: weights/wise/wise_{S1,S2}_a{α}.pt (gitignored) + weights/wise/manifest.json
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "weights" / "wise"
BASES = {"S2": "runs/detect/v6split_f2_nofreeze/weights/best.pt"}
ALPHAS = (0.3, 0.5, 0.7, 0.85)
MIX_UPTO = 10  # layer 0..9 = backbone
BN_IMAGES = 256


def recalibrate_bn(model):
    """Tính lại running mean/var của mọi BN trên BN_IMAGES ảnh train V6split (letterbox 960, không gradient)."""
    import random

    import cv2
    import numpy as np
    from ultralytics.data.augment import LetterBox

    ims = sorted((ROOT / "data" / "merged_v6split" / "images" / "train").glob("*.jpg"))
    ims = random.Random(0).sample(ims, BN_IMAGES)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).float()
    bns = [mod for i, layer in enumerate(model.model) if i < MIX_UPTO
           for mod in layer.modules() if isinstance(mod, torch.nn.BatchNorm2d)]
    for mod in bns:
        mod.reset_running_stats()
        mod.momentum = None  # trung bình cộng dồn
    model.eval()
    for mod in bns:
        mod.train()  # chỉ BN của các layer được trộn cập nhật thống kê; neck/head giữ nguyên
    lb = LetterBox((960, 960), auto=False)
    with torch.no_grad():
        for i in range(0, len(ims), 16):
            x = np.stack([lb(image=cv2.imread(str(p)))[..., ::-1].transpose(2, 0, 1) for p in ims[i:i + 16]])
            model(torch.from_numpy(np.ascontiguousarray(x)).to(dev).float() / 255)
    for mod in bns:
        mod.momentum = 0.03
    model.eval().cpu()


def main():
    from ultralytics import YOLO

    OUT.mkdir(parents=True, exist_ok=True)
    coco = YOLO(str(ROOT / "yolov8s.pt")).model.float().state_dict()
    man = {}
    for tag, path in BASES.items():
        for a in ALPHAS:
            m = YOLO(str(ROOT / path))
            sd = m.model.float().state_dict()
            mixed, kept = 0, 0
            for k, v in sd.items():
                idx = int(k.split(".")[1])
                if idx >= MIX_UPTO or k not in coco or coco[k].shape != v.shape:
                    kept += 1
                    continue
                if v.dtype.is_floating_point:
                    sd[k] = a * v + (1 - a) * coco[k].to(v.dtype)
                    mixed += 1
            m.model.load_state_dict(sd)
            recalibrate_bn(m.model)
            out = OUT / f"wise_{tag}_a{a}.pt"
            m.save(str(out))
            man[out.name] = {"base": path, "alpha": a, "mixed_layers": f"0..{MIX_UPTO - 1}", "tensors_mixed": mixed,
                             "tensors_kept": kept, "bn_recalibrated_on": BN_IMAGES}
            print(out.name, man[out.name], flush=True)
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
