"""Dung tap danh gia NGOAI (external benchmark) tu cac clip camera tren cao.

- Trich khung deu tu moi clip -> data/ext_benchmark/images/<group>_<clip>_<frame>.jpg
- Gan nhan NHAP bang yolov8x.pt (COCO) -> data/ext_benchmark/labels/*.txt (schema 4 lop V5)
- Nhan nhap PHAI duoc sua tay truoc khi dung lam so lieu bao cao.
- Tap nay KHONG BAO GIO duoc dua vao train.

Chay: python scripts/data_prep/build_ext_benchmark.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "ext_benchmark"
REC = Path("/mnt/c/Users/xuand/Videos/Screen Recordings")

# (group, ten ngan, file, so khung lay)
CLIPS = [
    ("india", "c0919", REC / "Screen Recording 2026-09-19 231332.mp4", 40),
    ("india", "c0920", REC / "Screen Recording 2026-09-20 194658.mp4", 120),
    ("vncctv", "c1024", REC / "Screen Recording 2026-10-24 220935.mp4", 20),
    ("vncctv", "c1105", REC / "Screen Recording 2026-11-05 033122.mp4", 40),
    ("vncctv", "c1108", REC / "Screen Recording 2026-11-08 042445.mp4", 30),
]
COCO_TO_V5 = {3: 0, 2: 1, 5: 2, 7: 3}  # motorcycle, car, bus, truck
TEACHER, IMGSZ, CONF = "yolov8x.pt", 1280, 0.30


def main():
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    model = YOLO(TEACHER)
    manifest = {"teacher": TEACHER, "imgsz": IMGSZ, "conf": CONF,
                "labels_status": "DRAFT - chua sua tay", "clips": []}
    for group, short, path, n_take in CLIPS:
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_box = 0
        for idx in np.linspace(0, n - 2, n_take).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            stem = f"{group}_{short}_{int(idx):05d}"
            cv2.imwrite(str(OUT / "images" / f"{stem}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            res = model.predict(frame, imgsz=IMGSZ, conf=CONF, iou=0.6,
                                classes=list(COCO_TO_V5), agnostic_nms=True,
                                verbose=False)[0]
            lines = [f"{COCO_TO_V5[int(c)]} " + " ".join(f"{v:.6f}" for v in xywhn)
                     for c, xywhn in zip(res.boxes.cls.tolist(),
                                         res.boxes.xywhn.tolist())]
            (OUT / "labels" / f"{stem}.txt").write_text("\n".join(lines))
            n_box += len(lines)
        manifest["clips"].append({"group": group, "clip": short,
                                  "source": path.name, "frames": n_take,
                                  "draft_boxes": n_box})
        print(f"{group}/{short}: {n_take} khung, {n_box} hop nhap")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (ROOT / "data" / "data_ext_benchmark.yaml").write_text(
        "# Tap danh gia ngoai - KHONG dung de train. Nhan: xem manifest.json\n"
        f"path: {OUT}\ntrain: images\nval: images\ntest: images\n"
        "nc: 4\nnames:\n  0: motorcycle\n  1: car\n  2: bus\n  3: truck\n")


if __name__ == "__main__":
    main()
