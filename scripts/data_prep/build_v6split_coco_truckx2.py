"""T2 — V6split + 5.000 ảnh COCO (giống bộ đang dùng) + NHÂN ĐÔI ảnh train VN có xe tải.

Mục đích: lấy lại AP xe tải trên SOICT (0,67 → 0,58 khi trộn COCO vì truck COCO bị tô xám hết) mà không
đụng tới quy ước nhãn COCO. Ảnh VN có ít nhất 1 hộp truck (v10_, v10r_, soict24_, soictnight_, vnv3_)
được thêm một bản sao (hardlink, tiền tố dup_) → Ultralytics coi là 2 ảnh, mỗi epoch thấy 2 lần
(và mỗi lần tăng cường dữ liệu khác nhau).

Cần chạy trước: python scripts/data_prep/build_v6split_coco.py
Chạy: python scripts/data_prep/build_v6split_coco_truckx2.py
Kết quả: data/merged_v6split_coco_tx2, data/data_v6split_coco_tx2.yaml, reports/v6split_coco_tx2/manifest.json
"""
from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "merged_v6split_coco"
OUT = ROOT / "data" / "merged_v6split_coco_tx2"
YAML = ROOT / "data" / "data_v6split_coco_tx2.yaml"
REPORT = ROOT / "reports" / "v6split_coco_tx2"
VN_PREFIX = ("v10_", "v10r_", "soict24_", "soictnight_", "vnv3_")
TRUCK = 3
NAMES = ["motorcycle", "car", "bus", "truck"]


def link(src: Path, dst: Path):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main():
    assert BASE.exists(), "Chưa có data/merged_v6split_coco — chạy build_v6split_coco.py trước"
    if OUT.exists():
        shutil.rmtree(OUT)
    for sp in ("train", "val", "test", "test_vn_v10"):
        (OUT / "images" / sp).mkdir(parents=True)
        (OUT / "labels" / sp).mkdir(parents=True)
        for im in (BASE / "images" / sp).glob("*.jpg"):
            link(im, OUT / "images" / sp / im.name)
            shutil.copyfile(BASE / "labels" / sp / f"{im.stem}.txt", OUT / "labels" / sp / f"{im.stem}.txt")
    dup = Counter()
    for lp in sorted((BASE / "labels" / "train").glob("*.txt")):
        if not lp.name.startswith(VN_PREFIX):
            continue
        cls = [int(l.split()[0]) for l in lp.read_text().splitlines() if l.strip()]
        if TRUCK not in cls:
            continue
        link(BASE / "images" / "train" / f"{lp.stem}.jpg", OUT / "images" / "train" / f"dup_{lp.stem}.jpg")
        shutil.copyfile(lp, OUT / "labels" / "train" / f"dup_{lp.name}")
        dup[next(p for p in VN_PREFIX if lp.name.startswith(p)).rstrip("_")] += 1
    tb = Counter()
    for lp in (OUT / "labels" / "train").glob("*.txt"):
        for line in lp.read_text().splitlines():
            if line.strip():
                tb[NAMES[int(line.split()[0])]] += 1
    counts = {sp: len(list((OUT / "images" / sp).glob("*.jpg"))) for sp in ("train", "val", "test", "test_vn_v10")}
    YAML.write_text("# T2: V6split + 5.000 COCO + nhân đôi ảnh VN có xe tải — build_v6split_coco_truckx2.py\n"
                    f"path: {OUT}\ntrain: images/train\nval:   images/val\ntest:  images/test\n\n"
                    "nc: 4\nnames:\n  0: motorcycle\n  1: car\n  2: bus\n  3: truck\n")
    REPORT.mkdir(parents=True, exist_ok=True)
    man = {"built_at": datetime.now().isoformat(timespec="seconds"), "base": str(BASE.relative_to(ROOT)),
           "counts": counts, "train_boxes": dict(tb), "duplicated_vn_truck_images": dict(dup)}
    (REPORT / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    print(json.dumps(man, ensure_ascii=False))


if __name__ == "__main__":
    main()
