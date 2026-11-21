"""V6final + 3.000 ảnh MIO-TCD mới, chọn có chủ đích để giúp bắt ô tô ở xa và đa dạng loại xe.

Khác lần thêm MIO-TCD trước (build_v5fix_mio.py, chọn theo xe tải/xe buýt và ngẫu nhiên):
  - 2.000 ảnh có NHIỀU Ô TÔ NHỎ (>= 3 hộp car/pickup/van có cạnh tương đương < 32 px), ưu tiên ảnh nhiều nhất
  - 1.000 ảnh có XE BÁN TẢI hoặc XE VAN (pickup_truck, work_van -> car, đúng quy ước của SOICT)
  - 800 ảnh có XE TẢI (articulated/single_unit truck), ưu tiên ảnh nhiều xe tải
  - 700 ảnh có XE BUÝT, ưu tiên ảnh nhiều xe buýt
GIỮ NGUYÊN 3.000 ảnh MIO-TCD cũ (mio_*, đã có trong V6final) và thêm 4.500 ảnh mới không trùng (mio2_*). Cùng quy tắc lọc như trước: bỏ ảnh 342x228, bỏ ảnh có hộp > 35% ảnh,
tô xám xe "không rõ loại", bỏ ảnh có > 2 xe yolov8x thấy mà nhãn không có.
Kết quả: data/merged_v6final_mio = V6final (hardlink) + 4.500 ảnh mới; val/test giữ nguyên.

Chạy: python scripts/data_prep/build_v6final_mio.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TAR = ROOT / "data" / "MIO-TCD-Localization.tar"
RAW = ROOT / "data" / "raw_mio_tcd"
BASE = ROOT / "data" / "merged_v6final"
OUT = ROOT / "data" / "merged_v6final_mio"
YAML = ROOT / "data" / "data_v6final_mio.yaml"
REPORT = ROOT / "reports" / "v6final_mio"
OLD_MANIFEST = ROOT / "reports" / "v5fix_mio" / "manifest.json"

SEED = 0
N_SMALL, N_DIVERSE, N_TRUCK, N_BUS = 2000, 1000, 800, 700
TRUCKS = {"articulated_truck", "single_unit_truck"}
SMALL_PX, MIN_SMALL = 32, 3
MAP = {"car": 1, "pickup_truck": 1, "work_van": 1, "bus": 2,
       "articulated_truck": 3, "single_unit_truck": 3, "motorcycle": 0}
CARS = {"car", "pickup_truck", "work_van"}
BLANK = {"motorized_vehicle", "non-motorized_vehicle"}
MAX_BOX_AREA, GRAY = 0.35, (114, 114, 114)
FILTER_WEIGHTS, FILTER_CONF, FILTER_MAX_UNLABELED = "yolov8x.pt", 0.5, 2
NAMES = ["motorcycle", "car", "bus", "truck"]


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
    gt = defaultdict(list)
    for r in csv.reader(open(RAW / "MIO-TCD-Localization" / "gt_train.csv")):
        gt[r[0]].append((r[1], *map(float, r[2:6])))
    used = {f[4:-4] for f in json.loads(OLD_MANIFEST.read_text())["mio_files"]}

    small = Counter()
    diverse = []
    for i, boxes in gt.items():
        if i in used:
            continue
        for c, x1, y1, x2, y2 in boxes:
            if c in CARS and math.sqrt(max(x2 - x1, 0) * max(y2 - y1, 0)) < SMALL_PX:
                small[i] += 1
        if any(c in ("pickup_truck", "work_van") for c, *_ in boxes):
            diverse.append(i)
    pool_small = [i for i, n in small.items() if n >= MIN_SMALL]
    rng.shuffle(pool_small)
    pool_small.sort(key=lambda i: -small[i])  # ổn định: nhiều ô tô nhỏ trước
    pool_small = pool_small[:int(N_SMALL * 2.6)]
    s_small = set(pool_small)
    rng.shuffle(diverse)
    pool_div = [i for i in diverse if i not in s_small][:int(N_DIVERSE * 2.6)]
    taken = s_small | set(pool_div)

    def heavy_pool(cls, n):
        ids = [i for i in gt if i not in used and i not in taken and any(c in cls for c, *_ in gt[i])]
        rng.shuffle(ids)
        ids.sort(key=lambda i: -sum(c in cls for c, *_ in gt[i]))
        return ids[:int(n * 2.6)]
    pool_truck = heavy_pool(TRUCKS, N_TRUCK)
    taken |= set(pool_truck)
    pool_bus = heavy_pool({"bus"}, N_BUS)
    print(f"[1/4] ứng viên: nhiều ô tô nhỏ {len(pool_small)}, xe bán tải/van {len(pool_div)}, "
          f"xe tải {len(pool_truck)}, xe buýt {len(pool_bus)}", flush=True)

    ex = RAW / "_extract"
    ex.mkdir(parents=True, exist_ok=True)
    need = [f"MIO-TCD-Localization/train/{i}.jpg" for i in pool_small + pool_div + pool_truck + pool_bus
            if not (ex / f"MIO-TCD-Localization/train/{i}.jpg").exists()]
    if need:
        lst = RAW / "_members_v6.txt"
        lst.write_text("\n".join(need))
        subprocess.run(["tar", "-xf", str(TAR), "-C", str(ex), "--no-same-owner", "-T", str(lst)], check=True)
    src_dir = ex / "MIO-TCD-Localization" / "train"

    det = YOLO(str(ROOT / FILTER_WEIGHTS))
    stats, kept, prepared = Counter(), {"small": [], "diverse": [], "truck": [], "bus": []}, {}
    for group, lst_ids, cap in (("small", pool_small, N_SMALL), ("diverse", pool_div, N_DIVERSE),
                                ("truck", pool_truck, N_TRUCK), ("bus", pool_bus, N_BUS)):
        for k in range(0, len(lst_ids), 16):
            if len(kept[group]) >= cap:
                break
            batch, imgs = [], []
            for i in lst_ids[k:k + 16]:
                img = cv2.imread(str(src_dir / f"{i}.jpg"))
                if img is None:
                    stats["missing"] += 1
                    continue
                h, w = img.shape[:2]
                if w < 600:
                    stats["drop_lowres"] += 1
                    continue
                if any((x2 - x1) * (y2 - y1) / (w * h) > MAX_BOX_AREA for _, x1, y1, x2, y2 in gt[i]):
                    stats["drop_giant_box"] += 1
                    continue
                for c, x1, y1, x2, y2 in gt[i]:
                    if c in BLANK:
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), GRAY, -1)
                batch.append(i)
                imgs.append(img)
            if not imgs:
                continue
            for i, img, r in zip(batch, imgs, det.predict(imgs, imgsz=960, conf=FILTER_CONF, classes=[2, 3, 5, 7],
                                                          verbose=False, half=True)):
                d = r.boxes.xyxy.cpu().numpy()
                g = np.array([[x1, y1, x2, y2] for _, x1, y1, x2, y2 in gt[i]]).reshape(-1, 4)
                iou, ioa = iou_ioa(d, g)
                matched = (iou.max(1) >= 0.3) | (ioa.max(1) >= 0.6) if len(g) else np.zeros(len(d), bool)
                if int((~matched).sum()) > FILTER_MAX_UNLABELED:
                    stats["drop_unlabeled_vehicle"] += 1
                    continue
                if len(kept[group]) < cap:
                    prepared[i] = img
                    kept[group].append(i)
        print(f"      {group}: giữ {len(kept[group])}", flush=True)
    print(f"[2/4] loại: {dict(stats)}", flush=True)

    if OUT.exists():
        shutil.rmtree(OUT)
    for sp in ("train", "val", "test"):
        (OUT / "images" / sp).mkdir(parents=True)
        (OUT / "labels" / sp).mkdir(parents=True)
        for im in (BASE / "images" / sp).glob("*.jpg"):
            link(im, OUT / "images" / sp / im.name)
            shutil.copyfile(BASE / "labels" / sp / f"{im.stem}.txt", OUT / "labels" / sp / f"{im.stem}.txt")
    boxes = Counter()
    for group, ids in kept.items():
        for i in ids:
            img = prepared[i]
            h, w = img.shape[:2]
            stem = f"mio2_{i}"
            cv2.imwrite(str(OUT / "images" / "train" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows = []
            for c, x1, y1, x2, y2 in gt[i]:
                if c not in MAP:
                    continue
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                rows.append(f"{MAP[c]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
                boxes[NAMES[MAP[c]]] += 1
                if c in ("pickup_truck", "work_van"):
                    boxes[c] += 1
            (OUT / "labels" / "train" / f"{stem}.txt").write_text("\n".join(rows))
    train_boxes = Counter()
    for lp in (OUT / "labels" / "train").glob("*.txt"):
        for line in lp.read_text().splitlines():
            if line.strip():
                train_boxes[NAMES[int(line.split()[0])]] += 1
    counts = {sp: len(list((OUT / "images" / sp).glob("*.jpg"))) for sp in ("train", "val", "test")}
    YAML.write_text(
        "# V6final (gồm 3.000 ảnh MIO-TCD cũ) + 4.500 ảnh MIO-TCD mới (ô tô nhỏ, bán tải/van, xe tải, xe buýt) — sinh bởi build_v6final_mio.py\n"
        "# MIO-TCD: CC BY-NC-SA 4.0. Val/test giống hệt V6final, V5fix, V5+.\n"
        f"path: {OUT}\ntrain: images/train\nval:   images/val\ntest:  images/test\n\n"
        "nc: 4\nnames:\n  0: motorcycle\n  1: car\n  2: bus\n  3: truck\n")
    REPORT.mkdir(parents=True, exist_ok=True)
    man = {"built_at": datetime.now().isoformat(timespec="seconds"), "base": str(BASE.relative_to(ROOT)),
           "counts": counts, "train_boxes": dict(train_boxes), "new_mio_boxes": dict(boxes),
           "new_mio_by_group": {k: len(v) for k, v in kept.items()}, "drop_stats": dict(stats),
           "params": {"small_px": SMALL_PX, "min_small": MIN_SMALL, "max_box_area": MAX_BOX_AREA,
                      "filter": [FILTER_WEIGHTS, FILTER_CONF, FILTER_MAX_UNLABELED]},
           "license": "MIO-TCD CC BY-NC-SA 4.0",
           "new_mio_files": {k: [f"mio2_{i}.jpg" for i in v] for k, v in kept.items()}}
    (REPORT / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    print("[3/4]", json.dumps({k: man[k] for k in ("counts", "train_boxes", "new_mio_boxes", "new_mio_by_group")}))

    col = [(0, 200, 0), (255, 140, 0), (0, 0, 255), (255, 0, 255)]
    for group in kept:
        tiles = []
        for i in random.Random(1).sample(kept[group], 16):
            img = cv2.imread(str(OUT / "images" / "train" / f"mio2_{i}.jpg"))
            h, w = img.shape[:2]
            for line in (OUT / "labels" / "train" / f"mio2_{i}.txt").read_text().splitlines():
                c, x, y, bw, bh = map(float, line.split())
                cv2.rectangle(img, (int((x - bw / 2) * w), int((y - bh / 2) * h)),
                              (int((x + bw / 2) * w), int((y + bh / 2) * h)), col[int(c)], 2)
            tiles.append(cv2.resize(img, (480, 320)))
        cv2.imwrite(str(REPORT / f"preview_{group}.jpg"),
                    np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, 16, 4)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("[4/4] preview:", REPORT)


if __name__ == "__main__":
    main()
