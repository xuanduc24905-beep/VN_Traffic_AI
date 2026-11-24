"""V6split + 5.000 ảnh COCO train2017 (rehearsal / replay — chống quên kiến thức COCO khi fine-tune).

Chọn ảnh (không lấy val2017):
  - 2.500 ảnh xe máy: mọi ảnh có >= 2 xe máy (~1.500, ưu tiên có người lái), bù bằng ảnh 1 xe máy có người lái
  - 1.700 ảnh có >= 4 ô tô, ưu tiên nhiều ô tô nhỏ (cạnh tương đương < 32 px)
  -   800 ảnh có xe buýt
Đổi nhãn sang quy ước SOICT (4 lớp motorcycle/car/bus/truck):
  - motorcycle: GỘP hộp xe máy với hộp người lái (dùng nhãn person GỐC của COCO, không đoán bằng model).
      Người lái = hộp person có phần ngang trùng >= 50% bề rộng người, đáy người nằm trong khoảng chiều cao xe,
      đỉnh người cao hơn đỉnh xe.
  - car → car, bus → bus
  - truck: COCO gộp cả xe bán tải (SOICT gọi là car) → KHÔNG dùng làm nhãn, tô xám (114) vùng đó
  - vùng iscrowd của car/motorcycle/bus/truck → tô xám
  - bicycle, person, ...: giữ nguyên ảnh, không gán nhãn (SOICT cũng không gán)
Kết quả: data/merged_v6split_coco (V6split hardlink + coco_*.jpg), data/data_v6split_coco.yaml,
         reports/v6split_coco/manifest.json + ảnh mẫu.
Giấy phép: nhãn COCO CC BY 4.0; ảnh theo điều khoản Flickr của từng ảnh (dùng cho nghiên cứu).

Chạy: python scripts/data_prep/build_v6split_coco.py
"""
from __future__ import annotations

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
ANN = ROOT / "data" / "raw_coco" / "annotations" / "instances_train2017.json"
IMG_DIR = ROOT / "data" / "raw_coco" / "train2017"
BASE = ROOT / "data" / "merged_v6split"
OUT = ROOT / "data" / "merged_v6split_coco"
YAML = ROOT / "data" / "data_v6split_coco.yaml"
REPORT = ROOT / "reports" / "v6split_coco"
URL = "http://images.cocodataset.org/train2017/{:012d}.jpg"

SEED = 0
N_MOTO, N_CAR, N_BUS = 2500, 1700, 800
CAT = {"person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "bus": 6, "truck": 8}
TO_V = {3: 1, 4: 0, 6: 2}  # car, motorcycle, bus
MASK_CATS = {8}  # truck
VEH = {3, 4, 6, 8}
GRAY = (114, 114, 114)
NAMES = ["motorcycle", "car", "bus", "truck"]


def rider_union(moto, persons):
    """moto, persons: [x, y, w, h] (COCO). Trả hộp xe máy đã gộp người lái và số người được gộp."""
    mx1, my1, mw, mh = moto
    mx2, my2 = mx1 + mw, my1 + mh
    x1, y1, x2, y2, n = mx1, my1, mx2, my2, 0
    for px1, py1, pw, ph in persons:
        px2, py2 = px1 + pw, py1 + ph
        ov = max(0.0, min(mx2, px2) - max(mx1, px1)) / max(pw, 1.0)
        if ov >= 0.5 and py1 < my1 and my1 <= py2 <= my2 + 0.1 * mh and ph < 3 * mh:
            x1, y1, x2 = min(x1, px1), min(y1, py1), max(x2, px2)
            n += 1
    return [x1, y1, x2, y2], n


def link(src: Path, dst: Path):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main():
    rng = random.Random(SEED)
    d = json.loads(ANN.read_text())
    info = {im["id"]: im for im in d["images"]}
    anns = defaultdict(list)
    for a in d["annotations"]:
        anns[a["image_id"]].append(a)

    def stats(i):
        A = anns[i]
        moto = [a for a in A if a["category_id"] == 4 and not a["iscrowd"]]
        cars = [a for a in A if a["category_id"] == 3 and not a["iscrowd"]]
        bus = [a for a in A if a["category_id"] == 6 and not a["iscrowd"]]
        persons = [a["bbox"] for a in A if a["category_id"] == 1 and not a["iscrowd"]]
        riders = sum(rider_union(m["bbox"], persons)[1] > 0 for m in moto)
        small_cars = sum(math.sqrt(a["bbox"][2] * a["bbox"][3]) < 32 for a in cars)
        return len(moto), riders, len(cars), small_cars, len(bus)

    S = {i: stats(i) for i in anns if any(a["category_id"] in (3, 4, 6) for a in anns[i])}
    ids = list(S)
    rng.shuffle(ids)
    moto = sorted([i for i in ids if S[i][0] >= 2], key=lambda i: (-(S[i][1] > 0), -S[i][1]))[:N_MOTO]
    # COCO chỉ có ~1.500 ảnh >= 2 xe máy → bù bằng ảnh có 1 xe máy CÓ người lái
    moto += [i for i in ids if S[i][0] == 1 and S[i][1] == 1][:N_MOTO - len(moto)]
    used = set(moto)
    car = sorted([i for i in ids if S[i][2] >= 4 and i not in used], key=lambda i: -S[i][3])[:N_CAR]
    used |= set(car)
    bus = [i for i in ids if S[i][4] >= 1 and i not in used][:N_BUS]
    sel = {"moto": moto, "car": car, "bus": bus}
    print("[1/5] chọn:", {k: len(v) for k, v in sel.items()},
          "| ảnh xe máy có người lái:", sum(S[i][1] > 0 for i in moto), flush=True)

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    need = [i for v in sel.values() for i in v if not (IMG_DIR / f"{i:012d}.jpg").exists()]
    if need:
        lst = IMG_DIR.parent / "_urls.txt"
        lst.write_text("\n".join(URL.format(i) for i in need))
        subprocess.run(f"xargs -P 16 -n 20 curl -sS -f --retry 3 --remote-name-all < {lst}", shell=True, cwd=IMG_DIR, check=False)
    miss = [i for v in sel.values() for i in v if not (IMG_DIR / f"{i:012d}.jpg").exists()]
    print(f"[2/5] tải ảnh xong, thiếu {len(miss)}", flush=True)

    if OUT.exists():
        shutil.rmtree(OUT)
    for sp in ("train", "val", "test", "test_vn_v10"):
        (OUT / "images" / sp).mkdir(parents=True)
        (OUT / "labels" / sp).mkdir(parents=True)
        for im in (BASE / "images" / sp).glob("*.jpg"):
            link(im, OUT / "images" / sp / im.name)
            shutil.copyfile(BASE / "labels" / sp / f"{im.stem}.txt", OUT / "labels" / sp / f"{im.stem}.txt")

    st = Counter()
    kept = defaultdict(list)
    for grp, v in sel.items():
        for i in v:
            p = IMG_DIR / f"{i:012d}.jpg"
            img = cv2.imread(str(p)) if p.exists() else None
            if img is None:
                st["skip_missing"] += 1
                continue
            h, w = img.shape[:2]
            A = anns[i]
            persons = [a["bbox"] for a in A if a["category_id"] == 1 and not a["iscrowd"]]
            rows, keep_boxes = [], []
            for a in A:
                c = a["category_id"]
                if c not in VEH:
                    continue
                x, y, bw, bh = a["bbox"]
                if a["iscrowd"] or c in MASK_CATS:
                    cv2.rectangle(img, (int(x), int(y)), (int(math.ceil(x + bw)), int(math.ceil(y + bh))), GRAY, -1)
                    st["masked_crowd" if a["iscrowd"] else "masked_truck"] += 1
                    continue
                if c == 4:
                    (x1, y1, x2, y2), n = rider_union(a["bbox"], persons)
                    st["moto_with_rider" if n else "moto_no_rider"] += 1
                else:
                    x1, y1, x2, y2 = x, y, x + bw, y + bh
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                keep_boxes.append((x1, y1, x2, y2))
                rows.append(f"{TO_V[c]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
                st[f"box_{NAMES[TO_V[c]]}"] += 1
            if not rows:
                st["skip_empty"] += 1
                continue
            # vùng tô xám có thể đè lên xe có nhãn → trả lại pixel trong hộp có nhãn
            orig = cv2.imread(str(p))
            for x1, y1, x2, y2 in keep_boxes:
                img[int(y1):int(math.ceil(y2)), int(x1):int(math.ceil(x2))] = \
                    orig[int(y1):int(math.ceil(y2)), int(x1):int(math.ceil(x2))]
            name = f"coco_{i:012d}"
            cv2.imwrite(str(OUT / "images" / "train" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (OUT / "labels" / "train" / f"{name}.txt").write_text("\n".join(rows))
            kept[grp].append(name)
    print("[3/5]", {k: len(v) for k, v in kept.items()}, dict(st), flush=True)

    tb = Counter()
    for lp in (OUT / "labels" / "train").glob("*.txt"):
        for line in lp.read_text().splitlines():
            if line.strip():
                tb[NAMES[int(line.split()[0])]] += 1
    counts = {sp: len(list((OUT / "images" / sp).glob("*.jpg"))) for sp in ("train", "val", "test", "test_vn_v10")}
    head = "nc: 4\nnames:\n  0: motorcycle\n  1: car\n  2: bus\n  3: truck\n"
    YAML.write_text("# V6split + 5.000 ảnh COCO train2017 (replay) — build_v6split_coco.py\n"
                    f"path: {OUT}\ntrain: images/train\nval:   images/val\ntest:  images/test\n\n" + head)
    REPORT.mkdir(parents=True, exist_ok=True)
    man = {"built_at": datetime.now().isoformat(timespec="seconds"), "base": str(BASE.relative_to(ROOT)),
           "counts": counts, "train_boxes": dict(tb), "coco_by_group": {k: len(v) for k, v in kept.items()},
           "coco_pct_of_train": round(sum(len(v) for v in kept.values()) / counts["train"], 4),
           "convert_stats": dict(st),
           "rules": {"motorcycle": "gộp hộp person GT của người lái", "truck": "tô xám (COCO gộp xe bán tải)",
                     "iscrowd": "tô xám"},
           "license": "COCO annotations CC BY 4.0; ảnh theo điều khoản Flickr",
           "coco_files": {k: v for k, v in kept.items()}}
    (REPORT / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    print("[4/5]", json.dumps({k: man[k] for k in ("counts", "train_boxes", "coco_pct_of_train")}), flush=True)

    col = [(0, 200, 0), (255, 140, 0), (0, 0, 255), (255, 0, 255)]
    for grp in kept:
        tiles = []
        for name in random.Random(1).sample(kept[grp], 12):
            img = cv2.imread(str(OUT / "images" / "train" / f"{name}.jpg"))
            h, w = img.shape[:2]
            for line in (OUT / "labels" / "train" / f"{name}.txt").read_text().splitlines():
                c, x, y, bw, bh = map(float, line.split())
                cv2.rectangle(img, (int((x - bw / 2) * w), int((y - bh / 2) * h)),
                              (int((x + bw / 2) * w), int((y + bh / 2) * h)), col[int(c)], 2)
            tiles.append(cv2.resize(img, (480, 360)))
        cv2.imwrite(str(REPORT / f"preview_{grp}.jpg"),
                    np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, 12, 4)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("[5/5] preview:", REPORT, flush=True)


if __name__ == "__main__":
    main()
