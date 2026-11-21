"""Dựng bộ V6final cho project cuối kì: sửa lỗi dữ liệu một lần, giữ nguyên val/test.

Xuất phát từ data/merged_v5fix_mio (V5 + v10 đã lọc + UA-DETRAC + MIO-TCD) và:
  1. BỎ toàn bộ Cần Thơ (1 camera, bỏ trống làn trái, xe bán tải gán truck, có hộp bus bao nhầm ô tô).
  2. BÙ NHÃN ở SOICT ngày, SOICT đêm, VNv3 bằng yolov8x (COCO, imgsz 1280). Một phát hiện được coi là
     xe thiếu nhãn khi IoU < 0,3 với mọi hộp có nhãn và nằm trong hộp có nhãn < 60% (IoA).
       - ô tô, xe buýt, độ tin cậy >= 0,5  -> THÊM NHÃN
       - xe tải (COCO gọi cả xe bán tải/SUV là truck, SOICT gán các xe này là car), xe máy (hộp COCO
         không bao người lái, lệch quy ước V5), và ô tô/xe buýt 0,40-0,5 -> TÔ XÁM (không thêm nhãn)
       - sau khi tô xám, trả lại nguyên pixel trong mọi hộp có nhãn (cũ và mới)
  3. LẤY LẠI xe tải v10: 598 ảnh trước đây bị bỏ vì có hộp > 35% ảnh, nay thu nhỏ trong khung 960 sao cho
     hộp lớn nhất <= 35% khung (không phóng to, cạnh ảnh không dưới 320 px).
Val (SOICT cam_07) và test (SOICT cam_09/10) giữ y hệt V5fix/V5+ để so sánh được.
Ảnh không đổi được hardlink (không tốn thêm dung lượng, vẫn là file thật).

Chạy: python scripts/data_prep/build_v6final.py
"""
from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "merged_v5fix_mio"
V10_SRC = ROOT / "data" / "merged_v5plus_clean"
V10_MANIFEST = ROOT / "reports" / "v5plus_clean" / "manifest_full.json"
OUT = ROOT / "data" / "merged_v6final"
YAML = ROOT / "data" / "data_v6final.yaml"
REPORT = ROOT / "reports" / "v6final"

SEED = 0
FIX_SOURCES = ("soict24", "soictnight", "vnv3")
DROP_PREFIX = "cantho_"
TEACHER, TEACHER_IMGSZ = "yolov8x.pt", 1280
CONF_ADD, CONF_MASK = 0.50, 0.40
IOU_MATCH, IOA_INSIDE = 0.3, 0.6
COCO2V5 = {3: 0, 2: 1, 5: 2, 7: 3}
ADD_CLASSES = {1, 2}  # car, bus
V10_MAX_BOX, CANVAS, MIN_SIDE = 0.35, 960, 320
GRAY = (114, 114, 114)
NAMES = ["motorcycle", "car", "bus", "truck"]


def load(p: Path) -> np.ndarray:
    return np.loadtxt(p, ndmin=2) if p.exists() and p.stat().st_size else np.zeros((0, 5))


def write(p: Path, rows) -> None:
    p.write_text("\n".join(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in rows))


def xyxy(a, w, h):
    return np.c_[(a[:, 1] - a[:, 3] / 2) * w, (a[:, 2] - a[:, 4] / 2) * h,
                 (a[:, 1] + a[:, 3] / 2) * w, (a[:, 2] + a[:, 4] / 2) * h].reshape(-1, 4)


def iou_ioa(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b))), np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    aa = (a[:, 2:] - a[:, :2]).prod(1)[:, None]
    ab = (b[:, 2:] - b[:, :2]).prod(1)[None, :]
    return inter / (aa + ab - inter + 1e-9), inter / (aa + 1e-9)


def link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main():
    from ultralytics import YOLO

    rng = random.Random(SEED)
    if OUT.exists():
        shutil.rmtree(OUT)
    for sp in ("train", "val", "test"):
        (OUT / "images" / sp).mkdir(parents=True)
        (OUT / "labels" / sp).mkdir(parents=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    st = defaultdict(Counter)
    review = []  # (tên ảnh, hộp gốc, hộp thêm, hộp che) cho ảnh mẫu

    # val/test: giữ y hệt
    for sp in ("val", "test"):
        for im in sorted((BASE / "images" / sp).glob("*.jpg")):
            link(im, OUT / "images" / sp / im.name)
            shutil.copyfile(BASE / "labels" / sp / f"{im.stem}.txt", OUT / "labels" / sp / f"{im.stem}.txt")

    # train
    teacher = YOLO(str(ROOT / TEACHER))
    to_fix = []
    for im in sorted((BASE / "images" / "train").glob("*.jpg")):
        src = im.name.split("_")[0]
        if im.name.startswith(DROP_PREFIX):
            st["cantho"]["dropped"] += 1
            continue
        if src in FIX_SOURCES:
            to_fix.append(im)
            continue
        link(im, OUT / "images" / "train" / im.name)
        shutil.copyfile(BASE / "labels" / "train" / f"{im.stem}.txt", OUT / "labels" / "train" / f"{im.stem}.txt")
        st[src]["kept"] += 1

    print(f"[bù nhãn] {len(to_fix)} ảnh SOICT/VNv3, chạy {TEACHER}...", flush=True)
    for k in range(0, len(to_fix), 8):
        batch = to_fix[k:k + 8]
        for im, r in zip(batch, teacher.predict([str(p) for p in batch], imgsz=TEACHER_IMGSZ, conf=CONF_MASK,
                                                 classes=list(COCO2V5), verbose=False, half=True)):
            src = im.name.split("_")[0]
            img = r.orig_img.copy()
            h, w = img.shape[:2]
            gt = load(BASE / "labels" / "train" / f"{im.stem}.txt")
            gb = xyxy(gt, w, h)
            d = r.boxes.xyxy.cpu().numpy()
            dc = np.array([COCO2V5[int(c)] for c in r.boxes.cls.cpu().numpy()], dtype=int)
            ds = r.boxes.conf.cpu().numpy()
            iou, ioa = iou_ioa(d, gb)
            miss = np.ones(len(d), bool) if len(gb) == 0 else (iou.max(1) < IOU_MATCH) & (ioa.max(1) < IOA_INSIDE)
            add = miss & np.isin(dc, list(ADD_CLASSES)) & (ds >= CONF_ADD)
            mask = miss & ~add
            rows = [tuple(x) for x in gt]
            added = []
            for (x1, y1, x2, y2), c in zip(d[add], dc[add]):
                rows.append((c, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
                added.append((x1, y1, x2, y2, c))
                st[src][f"added_{NAMES[c]}"] += 1
            masked = []
            if mask.any():
                orig = img.copy()
                for (x1, y1, x2, y2), c in zip(d[mask], dc[mask]):
                    cv2.rectangle(img, (int(x1) - 2, int(y1) - 2), (int(x2) + 2, int(y2) + 2), GRAY, -1)
                    masked.append((x1, y1, x2, y2, c))
                    st[src][f"masked_{NAMES[c]}"] += 1
                keep_boxes = list(gb) + [a[:4] for a in added]
                for x1, y1, x2, y2 in keep_boxes:
                    xa, ya, xb, yb = max(int(x1), 0), max(int(y1), 0), min(int(np.ceil(x2)), w), min(int(np.ceil(y2)), h)
                    img[ya:yb, xa:xb] = orig[ya:yb, xa:xb]
                cv2.imwrite(str(OUT / "images" / "train" / im.name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                link(im, OUT / "images" / "train" / im.name)
            write(OUT / "labels" / "train" / f"{im.stem}.txt", rows)
            st[src]["kept"] += 1
            st[src]["images_changed"] += int(bool(added or masked))
            if added:
                review.append((im.name, gb.tolist(), added, masked))

    # lấy lại ảnh xe tải v10 bị bỏ vì hộp lớn
    man = json.loads(V10_MANIFEST.read_text())
    used = {p.name[len("v10_"):] for p in (BASE / "images" / "train").glob("v10_*.jpg")}
    restored = 0
    for rec in man["v10_files"]:
        name = rec["file"]
        lab = load(V10_SRC / "labels" / "train" / f"{Path(name).stem}.txt")
        if len(lab) == 0 or (lab[:, 3] * lab[:, 4]).max() <= V10_MAX_BOX:
            continue
        img = cv2.imread(str(V10_SRC / "images" / "train" / name))
        h, w = img.shape[:2]
        amax = float((lab[:, 3] * lab[:, 4]).max())
        # diện tích hộp trong khung = amax * (w*s/CANVAS)^2 <= V10_MAX_BOX; không phóng to (s <= 1)
        s = min(1.0, np.sqrt(V10_MAX_BOX / amax) * CANVAS / max(w, h))
        nw, nh = int(np.floor(w * s)), int(np.floor(h * s))
        if min(nw, nh) < MIN_SIDE:
            st["v10"]["restore_skipped_small"] += 1
            continue
        small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA) if s < 1 else img
        canvas = np.full((CANVAS, CANVAS, 3), GRAY, np.uint8)
        ox, oy = rng.randint(0, CANVAS - nw), rng.randint(0, CANVAS - nh)
        canvas[oy:oy + nh, ox:ox + nw] = small
        stem = f"v10r_{Path(name).stem}"
        cv2.imwrite(str(OUT / "images" / "train" / f"{stem}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows = [(c, (x * nw + ox) / CANVAS, (y * nh + oy) / CANVAS, bw * nw / CANVAS, bh * nh / CANVAS)
                for c, x, y, bw, bh in lab]
        assert max(r[3] * r[4] for r in rows) <= V10_MAX_BOX + 1e-4
        write(OUT / "labels" / "train" / f"{stem}.txt", rows)
        for r in rows:
            st["v10_restored"][f"box_{NAMES[int(r[0])]}"] += 1
        restored += 1
    st["v10_restored"]["images"] = restored

    # thống kê toàn bộ
    boxes = Counter()
    for lp in (OUT / "labels" / "train").glob("*.txt"):
        for row in load(lp):
            boxes[NAMES[int(row[0])]] += 1
    counts = {sp: len(list((OUT / "images" / sp).glob("*.jpg"))) for sp in ("train", "val", "test")}
    YAML.write_text(
        "# V6final — sinh bởi scripts/data_prep/build_v6final.py (bỏ Cần Thơ, bù nhãn, lấy lại xe tải v10)\n"
        "# Val/test giống hệt V5fix và V5+ (test SOICT cam_09/10)\n"
        f"path: {OUT}\ntrain: images/train\nval:   images/val\ntest:  images/test\n\n"
        "nc: 4\nnames:\n  0: motorcycle\n  1: car\n  2: bus\n  3: truck\n")
    rng2 = random.Random(1)
    man_out = {"built_at": datetime.now().isoformat(timespec="seconds"), "builder": "scripts/data_prep/build_v6final.py",
               "base": str(BASE.relative_to(ROOT)), "counts": counts, "train_boxes": dict(boxes),
               "per_source": {k: dict(v) for k, v in st.items()},
               "params": {"teacher": TEACHER, "imgsz": TEACHER_IMGSZ, "conf_add": CONF_ADD, "conf_mask": CONF_MASK,
                          "iou_match": IOU_MATCH, "ioa_inside": IOA_INSIDE, "add_classes": [NAMES[c] for c in ADD_CLASSES],
                          "v10_max_box": V10_MAX_BOX, "canvas": CANVAS, "min_side": MIN_SIDE},
               "review_sample": [r[0] for r in rng2.sample(review, min(100, len(review)))]}
    (REPORT / "manifest.json").write_text(json.dumps(man_out, indent=1, ensure_ascii=False))
    print(json.dumps({k: man_out[k] for k in ("counts", "train_boxes", "per_source")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
