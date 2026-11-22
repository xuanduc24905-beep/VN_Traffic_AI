"""Thử nghiệm: V5fix+MIO nhưng tô xám 35% bên trái mọi ảnh Cần Thơ (làn ngược chiều không được gán nhãn).

Cần Thơ = 1 camera cố định; nhóm gán nhãn gốc bỏ trống lòng đường bên trái dải phân cách.
Đo trên 259 ảnh train: tô xám x < 35% loại 66% xe thiếu nhãn (228/346) và không mất hộp nhãn nào.
Mọi thứ khác giữ y hệt data_v5fix_mio.yaml (symlink) -> so trực tiếp với M1.

Chạy: python scripts/data_prep/build_v5fix_mio_ctmask.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "merged_v5fix_mio"
OUT = ROOT / "data" / "merged_v5fix_mio_ctmask"
YAML = ROOT / "data" / "data_v5fix_mio_ctmask.yaml"
MASK_FRAC = 0.35
GRAY = (114, 114, 114)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    stats = {"masked_images": 0, "boxes_crossing_mask": 0}
    for sp in ("train", "val", "test"):
        (OUT / "images" / sp).mkdir(parents=True)
        (OUT / "labels" / sp).mkdir(parents=True)
        for im in (BASE / "images" / sp).iterdir():
            if im.suffix != ".jpg":
                continue
            lab = BASE / "labels" / sp / f"{im.stem}.txt"
            shutil.copyfile(lab, OUT / "labels" / sp / lab.name)
            if sp == "train" and im.name.startswith("cantho_"):
                img = cv2.imread(str(im))
                w = img.shape[1]
                img[:, : int(round(MASK_FRAC * w))] = GRAY
                cv2.imwrite(str(OUT / "images" / sp / im.name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                stats["masked_images"] += 1
                if lab.stat().st_size:
                    a = np.loadtxt(lab, ndmin=2)
                    stats["boxes_crossing_mask"] += int(((a[:, 1] - a[:, 3] / 2) < MASK_FRAC).sum())
            else:
                (OUT / "images" / sp / im.name).symlink_to(im.resolve())
    YAML.write_text(
        "# V5fix + MIO-TCD + tô xám 35% trái ảnh Cần Thơ — sinh bởi scripts/data_prep/build_v5fix_mio_ctmask.py\n"
        f"path: {OUT}\ntrain: images/train\nval:   images/val\ntest:  images/test\n\n"
        "nc: 4\nnames:\n  0: motorcycle\n  1: car\n  2: bus\n  3: truck\n")
    (ROOT / "reports" / "v5fix_mio" / "ctmask_stats.json").write_text(json.dumps(stats, indent=1))
    print(stats)


if __name__ == "__main__":
    main()
