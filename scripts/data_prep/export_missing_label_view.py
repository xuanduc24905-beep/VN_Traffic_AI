"""Xuat thu muc XEM cac xe ma yolov8x (COCO) bat duoc nhung nhan goc KHONG co.

Chi doc du lieu, khong sua nhan. Nguon: phan V5 train (cantho, soict24, soictnight, vnv3).

Output: outputs/review_missing_labels/
  <nguon>/rNNNN_missK_<ten>.jpg anh co >= 1 xe nghi thieu nhan; r0001 = anh thieu nhieu nhat, K = so xe thieu
                              xanh la mong = nhan goc, do day = v8x de xuat (lop + conf)
  <nguon>/proposals/<ten>.txt hop de xuat dang YOLO (4 lop V5) — de nap vao CVAT/Label Studio
  summary.csv                 moi dong 1 anh: nguon, ten, so nhan goc, so de xuat, lop
  README.txt                  tom tat theo nguon + cach doc

Tieu chi "thieu nhan": hop v8x conf >= 0.40, IoU < 0.3 voi moi hop goc va nam trong hop goc < 60%.

Chay: python scripts/data_prep/export_missing_label_view.py
"""
from __future__ import annotations

import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "merged_v5fix"
OUT = ROOT / "outputs" / "review_missing_labels"
SOURCES = ("cantho", "soict24", "soictnight", "vnv3")
COCO_TO_V5 = {3: 0, 2: 1, 5: 2, 7: 3}
NAMES = ["motorcycle", "car", "bus", "truck"]
WEIGHTS, IMGSZ, CONF = "yolov8x.pt", 1280, 0.40
IOU_MATCH, IOA_INSIDE = 0.3, 0.6


def iou_ioa(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b))), np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    aa = (a[:, 2:] - a[:, :2]).prod(1)[:, None]
    ab = (b[:, 2:] - b[:, :2]).prod(1)[None, :]
    return inter / (aa + ab - inter + 1e-9), inter / (aa + 1e-9)


def main():
    from ultralytics import YOLO

    if OUT.exists():
        shutil.rmtree(OUT)
    model = YOLO(str(ROOT / WEIGHTS))
    rows, per_src = [], defaultdict(Counter)
    for src in SOURCES:
        (OUT / src / "proposals").mkdir(parents=True)
        ims = sorted(p for p in (SRC / "images" / "train").glob(f"{src}_*") if p.suffix == ".jpg")
        for i in range(0, len(ims), 8):
            batch = ims[i:i + 8]
            for p, r in zip(batch, model.predict([str(x) for x in batch], imgsz=IMGSZ, conf=CONF,
                                                 classes=list(COCO_TO_V5), verbose=False, half=True)):
                img = r.orig_img.copy()
                h, w = img.shape[:2]
                lp = SRC / "labels" / "train" / (p.stem + ".txt")
                g = np.loadtxt(lp, ndmin=2) if lp.exists() and lp.stat().st_size else np.zeros((0, 5))
                gb = np.c_[(g[:, 1] - g[:, 3] / 2) * w, (g[:, 2] - g[:, 4] / 2) * h,
                           (g[:, 1] + g[:, 3] / 2) * w, (g[:, 2] + g[:, 4] / 2) * h].reshape(-1, 4)
                d = r.boxes.xyxy.cpu().numpy()
                dc = np.array([COCO_TO_V5[int(c)] for c in r.boxes.cls.cpu().numpy()], dtype=int)
                ds = r.boxes.conf.cpu().numpy()
                iou, ioa = iou_ioa(d, gb)
                miss = np.ones(len(d), bool) if len(gb) == 0 else (iou.max(1) < IOU_MATCH) & (ioa.max(1) < IOA_INSIDE)
                per_src[src]["images"] += 1
                per_src[src]["gt_boxes"] += len(g)
                if not miss.any():
                    continue
                per_src[src]["images_with_missing"] += 1
                th = max(1, round(w / 640))
                for b in gb:
                    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 0), th)
                lines = []
                for b, c, s in zip(d[miss], dc[miss], ds[miss]):
                    per_src[src][f"missing_{NAMES[c]}"] += 1
                    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), th + 1)
                    cv2.putText(img, f"{NAMES[c]} {s:.2f}", (int(b[0]), max(int(b[1]) - 4, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4 * th, (0, 0, 255), th)
                    lines.append(f"{c} {(b[0] + b[2]) / 2 / w:.6f} {(b[1] + b[3]) / 2 / h:.6f} "
                                 f"{(b[2] - b[0]) / w:.6f} {(b[3] - b[1]) / h:.6f}")
                n = int(miss.sum())
                cv2.imwrite(str(OUT / src / f"miss{n:02d}_{p.name}"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                (OUT / src / "proposals" / f"{p.stem}.txt").write_text("\n".join(lines))
                cls = Counter(NAMES[c] for c in dc[miss])
                rows.append([src, p.name, len(g), n] + [cls.get(k, 0) for k in NAMES])
        files = sorted((OUT / src).glob("miss*.jpg"), key=lambda f: (-int(f.name[4:6]), f.name))
        for rank, f in enumerate(files, 1):
            f.rename(f.with_name(f"r{rank:04d}_{f.name}"))
        print(f"{src}: {dict(per_src[src])}", flush=True)
    rows.sort(key=lambda r: (r[0], -r[3]))
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["source", "image", "n_gt", "n_missing"] + [f"missing_{k}" for k in NAMES])
        wr.writerows(rows)
    lines = ["Xe yolov8x bat duoc ma nhan goc khong co (V5 train). CHI DE XEM — chua sua nhan nao.",
             f"Detector {WEIGHTS} imgsz={IMGSZ} conf>={CONF}; thieu = IoU<{IOU_MATCH} voi moi hop goc va "
             f"nam trong hop goc <{int(IOA_INSIDE * 100)}%.",
             "Mau: xanh la mong = nhan goc; do = v8x de xuat (lop + conf). Ten file: r0001 = anh thieu nhieu nhat, missK = so xe thieu.",
             "Luu y: v8x co the bat nham (bong, bien quang cao, xe lam, xe dap) — day la DE XUAT, can nguoi duyet.",
             "proposals/*.txt = hop de xuat dang YOLO 4 lop (0 motorcycle, 1 car, 2 bus, 3 truck).", ""]
    for src in SOURCES:
        c = per_src[src]
        miss = {k[8:]: v for k, v in c.items() if k.startswith("missing_")}
        lines.append(f"{src}: {c['images_with_missing']}/{c['images']} anh co xe thieu; "
                     f"nhan goc {c['gt_boxes']} hop; de xuat thieu {sum(miss.values())} hop {miss}")
    (OUT / "README.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-len(SOURCES):]))


if __name__ == "__main__":
    main()
