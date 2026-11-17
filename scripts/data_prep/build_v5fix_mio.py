"""Build V5fix+MIO: V5fix (giu nguyen) + toi da 3.000 anh MIO-TCD Localization (camera giao thong Bac My).

Muc dich: bo sung o to o xa (MIO gan nhan ca xe nho), xe tai va xe buyt — 2 lop V5fix con yeu.
Val/test GIONG HET V5fix -> so sanh truc tiep voi D1.

MIO-TCD (CC BY-NC-SA 4.0, chi dung phi thuong mai, phai ghi nguon):
  car, pickup_truck, work_van -> car      bus -> bus
  articulated_truck, single_unit_truck -> truck      motorcycle -> motorcycle
  motorized_vehicle, non-motorized_vehicle -> TO XAM (xe that nhung khong ro loai)
  pedestrian, bicycle -> nen (V5 cung khong gan nhan 2 lop nay)
Loc:
  - bo anh 342x228 (do phan giai qua thap)
  - bo anh co hop > 35% dien tich anh (cung quy tac voi v10)
  - bo anh co > 2 xe ma yolov8x thay nhung nhan khong co (cung quy tac voi UA-DETRAC)
  - lay mau co uu tien: 1.200 anh co xe tai, 900 anh co xe buyt, phan con lai ngau nhien

Chay: python scripts/data_prep/build_v5fix_mio.py
"""
from __future__ import annotations

import csv
import json
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
RAW = ROOT / "data" / "raw_mio_tcd"           # anh MIO da giai nen (chi .jpg + .csv)
BASE = ROOT / "data" / "merged_v5fix"
OUT = ROOT / "data" / "merged_v5fix_mio"
YAML = ROOT / "data" / "data_v5fix_mio.yaml"
REPORT = ROOT / "reports" / "v5fix_mio"

SEED = 0
N_TRUCK, N_BUS, N_TOTAL = 1200, 900, 3000
N_CANDIDATES = 6500
MAX_BOX_AREA = 0.35
MAP = {"car": 1, "pickup_truck": 1, "work_van": 1, "bus": 2,
       "articulated_truck": 3, "single_unit_truck": 3, "motorcycle": 0}
BLANK = {"motorized_vehicle", "non-motorized_vehicle"}
GRAY = (114, 114, 114)
FILTER_WEIGHTS, FILTER_CONF, FILTER_MAX_UNLABELED = "yolov8x.pt", 0.5, 2


def iou_ioa(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b))), np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    aa = (a[:, 2:] - a[:, :2]).prod(1)[:, None]
    ab = (b[:, 2:] - b[:, :2]).prod(1)[None, :]
    return inter / (aa + ab - inter + 1e-9), inter / (aa + 1e-9)


def extract(ids: list[str]) -> Path:
    """Giai nen gt_train.csv + cac anh can thiet vao thu muc rieng (khong chay gi tu ben trong)."""
    ex = RAW / "_extract"
    ex.mkdir(parents=True, exist_ok=True)
    lst = RAW / "_members.txt"
    names = ["MIO-TCD-Localization/gt_train.csv", "MIO-TCD-Localization/README.txt"]
    names += [f"MIO-TCD-Localization/train/{i}.jpg" for i in ids
              if not (ex / f"MIO-TCD-Localization/train/{i}.jpg").exists()]
    lst.write_text("\n".join(names))
    subprocess.run(["tar", "-xf", str(TAR), "-C", str(ex), "--no-same-owner", "-T", str(lst)], check=True)
    return ex / "MIO-TCD-Localization"


def read_gt(csv_path: Path):
    gt = defaultdict(list)
    for r in csv.reader(open(csv_path)):
        gt[r[0]].append((r[1], *map(float, r[2:6])))
    return gt


def main():
    rng = random.Random(SEED)
    RAW.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)

    # 1) gt + chon ung vien
    subprocess.run(["tar", "-xf", str(TAR), "-C", str(RAW), "--no-same-owner",
                    "MIO-TCD-Localization/gt_train.csv", "MIO-TCD-Localization/README.txt"], check=True)
    gt = read_gt(RAW / "MIO-TCD-Localization" / "gt_train.csv")
    ids = sorted(gt)
    has = lambda i, cls: any(b[0] in cls for b in gt[i])  # noqa: E731
    trucks = [i for i in ids if has(i, {"articulated_truck", "single_unit_truck"})]
    buses = [i for i in ids if has(i, {"bus"}) and i not in set(trucks)]
    rng.shuffle(trucks); rng.shuffle(buses)
    others = [i for i in ids if i not in set(trucks) | set(buses)]
    rng.shuffle(others)
    cand = {"truck": trucks[:int(N_CANDIDATES * 0.4)], "bus": buses[:int(N_CANDIDATES * 0.3)],
            "other": others[:int(N_CANDIDATES * 0.3)]}
    print("[1/4] ung vien:", {k: len(v) for k, v in cand.items()}, flush=True)
    src = extract([i for v in cand.values() for i in v])

    # 2) loc
    from ultralytics import YOLO
    det = YOLO(str(ROOT / FILTER_WEIGHTS))
    stats, kept = Counter(), defaultdict(list)
    prepared = {}
    for group, lst in cand.items():
        for k in range(0, len(lst), 16):
            batch, imgs = [], []
            for i in lst[k:k + 16]:
                img = cv2.imread(str(src / "train" / f"{i}.jpg"))
                if img is None:
                    stats["missing"] += 1; continue
                h, w = img.shape[:2]
                if w < 600:
                    stats["drop_lowres"] += 1; continue
                boxes = gt[i]
                if any((x2 - x1) * (y2 - y1) / (w * h) > MAX_BOX_AREA for _, x1, y1, x2, y2 in boxes):
                    stats["drop_giant_box"] += 1; continue
                for c, x1, y1, x2, y2 in boxes:
                    if c in BLANK:
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), GRAY, -1)
                batch.append(i); imgs.append(img)
            if not imgs:
                continue
            for i, img, r in zip(batch, imgs, det.predict(imgs, imgsz=960, conf=FILTER_CONF,
                                                          classes=[2, 3, 5, 7], verbose=False, half=True)):
                d = r.boxes.xyxy.cpu().numpy()
                g = np.array([[x1, y1, x2, y2] for _, x1, y1, x2, y2 in gt[i]]).reshape(-1, 4)
                iou, ioa = iou_ioa(d, g)
                matched = (iou.max(1) >= 0.3) | (ioa.max(1) >= 0.6) if len(g) else np.zeros(len(d), bool)
                if int((~matched).sum()) > FILTER_MAX_UNLABELED:
                    stats["drop_unlabeled_vehicle"] += 1; continue
                prepared[i] = img
                kept[group].append(i)
        print(f"      {group}: giu {len(kept[group])}", flush=True)

    # 3) lay mau theo uu tien
    pick = kept["truck"][:N_TRUCK] + kept["bus"][:N_BUS]
    pick += kept["other"][:max(0, N_TOTAL - len(pick))]
    print(f"[2/4] chon {len(pick)} anh; loai: {dict(stats)}", flush=True)

    # 4) merged = V5fix (symlink) + MIO
    if OUT.exists():
        shutil.rmtree(OUT)
    for sp in ("train", "val", "test"):
        (OUT / "images" / sp).mkdir(parents=True)
        (OUT / "labels" / sp).mkdir(parents=True)
        for im in (BASE / "images" / sp).iterdir():
            if im.suffix != ".jpg":
                continue
            (OUT / "images" / sp / im.name).symlink_to(im.resolve())
            shutil.copyfile(BASE / "labels" / sp / f"{im.stem}.txt", OUT / "labels" / sp / f"{im.stem}.txt")
    boxes = Counter()
    for i in pick:
        img = prepared[i]
        h, w = img.shape[:2]
        stem = f"mio_{i}"
        cv2.imwrite(str(OUT / "images" / "train" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows = []
        for c, x1, y1, x2, y2 in gt[i]:
            if c not in MAP:
                continue
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            rows.append(f"{MAP[c]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            boxes[["motorcycle", "car", "bus", "truck"][MAP[c]]] += 1
        (OUT / "labels" / "train" / f"{stem}.txt").write_text("\n".join(rows))
    YAML.write_text(
        "# V5fix + MIO-TCD — sinh boi scripts/data_prep/build_v5fix_mio.py\n"
        "# MIO-TCD: CC BY-NC-SA 4.0 (phi thuong mai, ghi nguon). Val/test giong het V5fix.\n"
        f"path: {OUT}\ntrain: images/train\nval:   images/val\ntest:  images/test\n\n"
        "nc: 4\nnames:\n  0: motorcycle\n  1: car\n  2: bus\n  3: truck\n")
    n_base = sum(1 for p in (BASE / "images" / "train").iterdir() if p.suffix == ".jpg")
    man = {"built_at": datetime.now().isoformat(timespec="seconds"), "seed": SEED, "base": str(BASE.relative_to(ROOT)),
           "counts": {"base_train": n_base, "mio_train": len(pick), "train_total": n_base + len(pick)},
           "mio_boxes": dict(boxes), "mio_drop_stats": dict(stats),
           "mio_pick_by_group": {"truck": min(N_TRUCK, len(kept["truck"])), "bus": min(N_BUS, len(kept["bus"])),
                                 "other": len(pick) - min(N_TRUCK, len(kept["truck"])) - min(N_BUS, len(kept["bus"]))},
           "params": {"max_box_area": MAX_BOX_AREA, "map": MAP, "blank": sorted(BLANK),
                      "filter": [FILTER_WEIGHTS, FILTER_CONF, FILTER_MAX_UNLABELED]},
           "license": "MIO-TCD CC BY-NC-SA 4.0", "mio_files": [f"mio_{i}.jpg" for i in pick]}
    (REPORT / "manifest.json").write_text(json.dumps(man, indent=1))
    print("[3/4]", json.dumps(man["counts"]), json.dumps(man["mio_boxes"]), flush=True)

    # preview
    col = [(0, 255, 0), (255, 160, 0), (0, 0, 255), (255, 0, 255)]
    tiles = []
    for i in random.Random(1).sample(pick, 16):
        img = cv2.imread(str(OUT / "images" / "train" / f"mio_{i}.jpg"))
        h, w = img.shape[:2]
        for line in (OUT / "labels" / "train" / f"mio_{i}.txt").read_text().splitlines():
            c, x, y, bw, bh = map(float, line.split())
            cv2.rectangle(img, (int((x - bw / 2) * w), int((y - bh / 2) * h)),
                          (int((x + bw / 2) * w), int((y + bh / 2) * h)), col[int(c)], 2)
        tiles.append(cv2.resize(img, (480, 320)))
    cv2.imwrite(str(REPORT / "preview_mio.jpg"), np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, 16, 4)]),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("[4/4] preview:", REPORT / "preview_mio.jpg")


if __name__ == "__main__":
    main()
