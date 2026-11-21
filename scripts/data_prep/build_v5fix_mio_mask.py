"""V5fix+MIO + tô xám MỌI xe thiếu nhãn trong 4 nguồn Việt Nam (cantho, soict24, soictnight, vnv3).

Xe thiếu nhãn = hộp yolov8x (COCO) thấy nhưng nhãn gốc không có, lấy từ
outputs/review_missing_labels/<nguồn>/proposals/*.txt (sinh bởi export_missing_label_view.py).
Thêm: Cần Thơ tô xám 35% bên trái (làn ngược chiều không bao giờ được gán nhãn).

An toàn: sau khi tô xám, TRẢ LẠI nguyên ảnh trong mọi hộp có nhãn -> không che mất xe đang được dạy.
Không thêm nhãn nào. Val/test và các nguồn khác giữ y hệt data_v5fix_mio.yaml -> so trực tiếp với M1.

Chạy: python scripts/data_prep/export_missing_label_view.py   (nếu chưa có proposals)
      python scripts/data_prep/build_v5fix_mio_mask.py
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "merged_v5fix_mio"
OUT = ROOT / "data" / "merged_v5fix_mio_mask"
YAML = ROOT / "data" / "data_v5fix_mio_mask.yaml"
PROPOSALS = ROOT / "outputs" / "review_missing_labels"
REPORT = ROOT / "reports" / "v5fix_mio"
SOURCES = ("cantho", "soict24", "soictnight", "vnv3")
CANTHO_STRIP = 0.35
PAD = 2  # px nới hộp che
GRAY = (114, 114, 114)


def yolo_to_xyxy(a: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.c_[(a[:, 1] - a[:, 3] / 2) * w, (a[:, 2] - a[:, 4] / 2) * h,
                 (a[:, 1] + a[:, 3] / 2) * w, (a[:, 2] + a[:, 4] / 2) * h].reshape(-1, 4)


def load(p: Path) -> np.ndarray:
    return np.loadtxt(p, ndmin=2) if p.exists() and p.stat().st_size else np.zeros((0, 5))


def main():
    for s in SOURCES:
        assert (PROPOSALS / s / "proposals").is_dir(), f"Thiếu {PROPOSALS / s} — chạy export_missing_label_view.py trước"
    if OUT.exists():
        shutil.rmtree(OUT)
    st = Counter()
    for sp in ("train", "val", "test"):
        (OUT / "images" / sp).mkdir(parents=True)
        (OUT / "labels" / sp).mkdir(parents=True)
        for im in (BASE / "images" / sp).iterdir():
            if im.suffix != ".jpg":
                continue
            lab = BASE / "labels" / sp / f"{im.stem}.txt"
            shutil.copyfile(lab, OUT / "labels" / sp / lab.name)
            src = im.name.split("_")[0]
            prop = PROPOSALS / src / "proposals" / f"{im.stem}.txt"
            if sp != "train" or src not in SOURCES or not (prop.exists() or src == "cantho"):
                (OUT / "images" / sp / im.name).symlink_to(im.resolve())
                continue
            img = cv2.imread(str(im))
            orig = img.copy()
            h, w = img.shape[:2]
            if src == "cantho":
                img[:, : int(round(CANTHO_STRIP * w))] = GRAY
            for x1, y1, x2, y2 in yolo_to_xyxy(load(prop), w, h):
                cv2.rectangle(img, (int(x1) - PAD, int(y1) - PAD), (int(x2) + PAD, int(y2) + PAD), GRAY, -1)
                st[f"masked_boxes_{src}"] += 1
            for x1, y1, x2, y2 in yolo_to_xyxy(load(lab), w, h):  # trả lại xe có nhãn
                xa, ya, xb, yb = max(int(x1), 0), max(int(y1), 0), min(int(np.ceil(x2)), w), min(int(np.ceil(y2)), h)
                img[ya:yb, xa:xb] = orig[ya:yb, xa:xb]
            cv2.imwrite(str(OUT / "images" / sp / im.name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            st[f"images_{src}"] += 1
    YAML.write_text(
        "# V5fix + MIO-TCD + tô xám xe thiếu nhãn ở 4 nguồn VN — sinh bởi scripts/data_prep/build_v5fix_mio_mask.py\n"
        f"path: {OUT}\ntrain: images/train\nval:   images/val\ntest:  images/test\n\n"
        "nc: 4\nnames:\n  0: motorcycle\n  1: car\n  2: bus\n  3: truck\n")
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "mask_stats.json").write_text(json.dumps(dict(st), indent=1))
    print(dict(st))


if __name__ == "__main__":
    main()
