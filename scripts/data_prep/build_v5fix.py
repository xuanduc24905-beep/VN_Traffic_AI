"""Build V5fix: V5 train + v10 (da sua hop khong lo) + UA-DETRAC (bu o to / xe buyt).

Muc dich: sua detector bo sot o to xa va ve hop truck khong lo khi chay video ngoai
(xem memory project_external_video_gap). Val/test GIU NGUYEN cua V5+clean de so sanh duoc.

Thanh phan train:
  1. V5 train (2.833 anh) — giu nguyen, khong bu nhan.
  2. v10 (lay tu V5+clean, da bo xe ba gac / trung / thieu nhan):
       - bo anh co hop chiem > 35% dien tich anh (nguon cua hop truck khong lo)
       - lay toi da 1.500 anh, round-robin theo camera (uu tien da dang camera)
       - KHONG phong 640 -> 960: dat anh 640x640 vao khung 960x960 (nen xam 114),
         vi tri ngau nhien (seed) -> xe giu dung kich thuoc goc
  3. UA-DETRAC train (60 video, camera tren cao Trung Quoc), tu XML goc:
       - car, van -> car ; bus -> bus ; others -> to xam (khong co lop truck rieng)
       - to xam cac ignored_region (xe khong nhan)
       - bo khung con xe may (DETRAC khong gan nhan xe may) hoac >= 3 xe chua gan nhan,
         theo yolov8x.pt COCO
       - toi da 50 khung / video, rai deu theo thoi gian

Chay: python scripts/data_prep/build_v5fix.py
"""
from __future__ import annotations

import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "merged_v5plus_clean"
SRC_MANIFEST = ROOT / "reports" / "v5plus_clean" / "manifest_full.json"
OUT = ROOT / "data" / "merged_v5fix"
YAML = ROOT / "data" / "data_v5fix.yaml"
REPORT = ROOT / "reports" / "v5fix"
DETRAC = Path("/home/xuand/yolov8_finetune/data/raw/ua_detrac")
DETRAC_XML = DETRAC / "DETRAC-Train-Annotations-XML" / "DETRAC-Train-Annotations-XML"
DETRAC_IMG = DETRAC / "DETRAC-Images" / "DETRAC-Images"

SEED = 0
V10_MAX_BOX_AREA = 0.35
V10_N = 1500
V10_CANVAS = 960
DETRAC_PER_SEQ = 50
DETRAC_CANDIDATES = 90
DETRAC_MAP = {"car": 1, "van": 1, "bus": 2}  # others -> blackout
GRAY = (114, 114, 114)
FILTER_WEIGHTS = "yolov8x.pt"
FILTER_CONF_MOTO = 0.35
FILTER_CONF_VEH = 0.5
FILTER_MAX_UNLABELED = 2


def read_labels(p: Path) -> np.ndarray:
    if not p.exists() or p.stat().st_size == 0:
        return np.zeros((0, 5))
    return np.loadtxt(p, ndmin=2)


def write_labels(p: Path, rows) -> None:
    p.write_text("\n".join(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in rows))


def prepare_dirs():
    if OUT.exists():
        shutil.rmtree(OUT)
    for sp in ("train", "val", "test"):
        (OUT / "images" / sp).mkdir(parents=True)
        (OUT / "labels" / sp).mkdir(parents=True)


def link_split(split: str, keep=lambda name: True) -> list[str]:
    names = []
    for img in sorted((SRC / "images" / split).iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png") or not keep(img.name):
            continue
        (OUT / "images" / split / img.name).symlink_to(img.resolve())
        lab = SRC / "labels" / split / (img.stem + ".txt")
        shutil.copyfile(lab, OUT / "labels" / split / lab.name)
        names.append(img.name)
    return names


# ---------------------------------------------------------------- v10
def build_v10(rng: random.Random) -> dict:
    man = json.loads(SRC_MANIFEST.read_text())
    by_cam, dropped_giant = defaultdict(list), []
    for rec in man["v10_files"]:
        lab = read_labels(SRC / "labels" / "train" / (Path(rec["file"]).stem + ".txt"))
        if len(lab) == 0:
            continue
        if (lab[:, 3] * lab[:, 4]).max() > V10_MAX_BOX_AREA:
            dropped_giant.append(rec["file"])
            continue
        by_cam[rec["camera_id"]].append(rec)
    for cam in by_cam:
        rng.shuffle(by_cam[cam])
    chosen, cams = [], sorted(by_cam)
    while len(chosen) < V10_N and any(by_cam.values()):
        for cam in cams:
            if by_cam[cam] and len(chosen) < V10_N:
                chosen.append(by_cam[cam].pop())
    boxes = np.zeros(4, int)
    for rec in chosen:
        img = cv2.imread(str(SRC / "images" / "train" / rec["file"]))
        h, w = img.shape[:2]
        canvas = np.full((V10_CANVAS, V10_CANVAS, 3), GRAY, np.uint8)
        ox, oy = rng.randint(0, V10_CANVAS - w), rng.randint(0, V10_CANVAS - h)
        canvas[oy:oy + h, ox:ox + w] = img
        stem = Path(rec["file"]).stem
        cv2.imwrite(str(OUT / "images" / "train" / f"{stem}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lab = read_labels(SRC / "labels" / "train" / f"{stem}.txt")
        rows = [(c, (x * w + ox) / V10_CANVAS, (y * h + oy) / V10_CANVAS, bw * w / V10_CANVAS, bh * h / V10_CANVAS)
                for c, x, y, bw, bh in lab]
        write_labels(OUT / "labels" / "train" / f"{stem}.txt", rows)
        for r in rows:
            boxes[int(r[0])] += 1
    return {"n_source": len(man["v10_files"]), "dropped_giant_box": len(dropped_giant),
            "n_kept": len(chosen), "n_cameras": len({r["camera_id"] for r in chosen}),
            "boxes": dict(zip(["motorcycle", "car", "bus", "truck"], boxes.tolist())),
            "files": [r["file"] for r in chosen]}


# ---------------------------------------------------------------- DETRAC
def parse_detrac(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    ignored = [tuple(float(b.get(k)) for k in ("left", "top", "width", "height"))
               for b in root.find("ignored_region") or []]
    frames = {}
    for fr in root.findall("frame"):
        tg = []
        for t in fr.find("target_list"):
            b = t.find("box")
            box = tuple(float(b.get(k)) for k in ("left", "top", "width", "height"))
            tg.append((t.find("attribute").get("vehicle_type"), box))
        frames[int(fr.get("num"))] = tg
    return root.get("name"), ignored, frames


def iou_ioa(a, b):
    """a: (N,4) xyxy, b: (M,4) xyxy -> iou (N,M), ioa wrt a (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b))), np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    aa = (a[:, 2:] - a[:, :2]).prod(1)[:, None]
    ab = (b[:, 2:] - b[:, :2]).prod(1)[None, :]
    return inter / (aa + ab - inter + 1e-9), inter / (aa + 1e-9)


def build_detrac() -> dict:
    from ultralytics import YOLO

    det = YOLO(str(ROOT / FILTER_WEIGHTS))
    stats = defaultdict(int)
    boxes = np.zeros(4, int)
    kept_files = []
    for xml in sorted(DETRAC_XML.glob("*.xml")):
        seq, ignored, frames = parse_detrac(xml)
        nums = sorted(n for n, t in frames.items() if any(v in DETRAC_MAP for v, _ in t))
        if not nums:
            continue
        cand = [nums[i] for i in np.linspace(0, len(nums) - 1, min(DETRAC_CANDIDATES, len(nums))).astype(int)]
        cand = sorted(set(cand))
        imgs, metas = [], []
        for n in cand:
            img = cv2.imread(str(DETRAC_IMG / seq / f"img{n:05d}.jpg"))
            if img is None:
                stats["missing_image"] += 1
                continue
            blank = list(ignored) + [b for v, b in frames[n] if v not in DETRAC_MAP]
            for l, t, w, h in blank:
                cv2.rectangle(img, (int(l), int(t)), (int(l + w), int(t + h)), GRAY, -1)
            gt = [(DETRAC_MAP[v], b) for v, b in frames[n] if v in DETRAC_MAP]
            imgs.append(img)
            metas.append((n, gt, blank))
        survivors = []
        for i in range(0, len(imgs), 16):
            res = det.predict(imgs[i:i + 16], imgsz=960, conf=FILTER_CONF_MOTO, classes=[1, 2, 3, 5, 7],
                              verbose=False, half=True)
            for img, (n, gt, blank), r in zip(imgs[i:i + 16], metas[i:i + 16], res):
                d = r.boxes.xyxy.cpu().numpy()
                c = r.boxes.cls.cpu().numpy().astype(int)
                s = r.boxes.conf.cpu().numpy()
                g = np.array([[l, t, l + w, tp + h] for _, (l, t, w, h) in gt for tp in [t]]).reshape(-1, 4)
                bl = np.array([[l, t, l + w, t + h] for l, t, w, h in blank]).reshape(-1, 4)
                _, ioa_blank = iou_ioa(d, bl)
                in_blank = ioa_blank.max(1) > 0.5 if len(bl) else np.zeros(len(d), bool)
                iou_gt, ioa_gt = iou_ioa(d, g)
                matched = ((iou_gt.max(1) >= 0.3) | (ioa_gt.max(1) >= 0.6)) if len(g) else np.zeros(len(d), bool)
                moto = (np.isin(c, [1, 3])) & ~in_blank & ~matched
                unl = np.isin(c, [2, 5, 7]) & (s >= FILTER_CONF_VEH) & ~in_blank & ~matched
                if moto.any():
                    stats["drop_motorcycle"] += 1
                    continue
                if unl.sum() > FILTER_MAX_UNLABELED:
                    stats["drop_unlabeled_vehicle"] += 1
                    continue
                survivors.append((n, img, gt))
        pick = [survivors[i] for i in np.linspace(0, len(survivors) - 1,
                                                  min(DETRAC_PER_SEQ, len(survivors))).astype(int)] if survivors else []
        seen = set()
        for n, img, gt in pick:
            if n in seen:
                continue
            seen.add(n)
            H, W = img.shape[:2]
            stem = f"detrac_{seq}_img{n:05d}"
            cv2.imwrite(str(OUT / "images" / "train" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows = []
            for cls, (l, t, w, h) in gt:
                l, t = max(l, 0), max(t, 0)
                w, h = min(w, W - l), min(h, H - t)
                if w > 1 and h > 1:
                    rows.append((cls, (l + w / 2) / W, (t + h / 2) / H, w / W, h / H))
                    boxes[cls] += 1
            write_labels(OUT / "labels" / "train" / f"{stem}.txt", rows)
            kept_files.append(stem + ".jpg")
        stats["sequences"] += 1
        stats["candidates"] += len(cand)
        print(f"  {seq}: {len(cand)} ung vien -> {len(seen)} khung", flush=True)
    return {"n_kept": len(kept_files), "stats": dict(stats),
            "boxes": dict(zip(["motorcycle", "car", "bus", "truck"], boxes.tolist())),
            "files": kept_files}


def preview(files: list[str], out: Path, n=12, seed=1):
    rng = random.Random(seed)
    col = [(0, 255, 0), (255, 128, 0), (0, 0, 255), (255, 0, 255)]
    tiles = []
    for f in rng.sample(files, min(n, len(files))):
        img = cv2.imread(str(OUT / "images" / "train" / f))
        H, W = img.shape[:2]
        for c, x, y, w, h in read_labels(OUT / "labels" / "train" / (Path(f).stem + ".txt")):
            cv2.rectangle(img, (int((x - w / 2) * W), int((y - h / 2) * H)),
                          (int((x + w / 2) * W), int((y + h / 2) * H)), col[int(c)], 2)
        tiles.append(cv2.resize(img, (640, int(640 * H / W))))
    hmax = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])


def main():
    rng = random.Random(SEED)
    prepare_dirs()
    REPORT.mkdir(parents=True, exist_ok=True)
    print("[1/4] V5 train + val + test (symlink tu V5+clean)")
    v5 = link_split("train", keep=lambda n: not n.startswith("v10_"))
    val = link_split("val")
    test = link_split("test")
    print(f"      train V5 {len(v5)} | val {len(val)} | test {len(test)}")
    print("[2/4] v10: bo hop > 35%, round-robin camera, dat vao khung 960")
    v10 = build_v10(rng)
    print(f"      giu {v10['n_kept']} / {v10['n_source']} (bo {v10['dropped_giant_box']} anh co hop khong lo)")
    print("[3/4] UA-DETRAC train (60 video)")
    det = build_detrac()
    print(f"      giu {det['n_kept']} khung; {det['stats']}")
    print("[4/4] YAML + manifest + preview")
    YAML.write_text(
        "# V5fix — sinh boi scripts/data_prep/build_v5fix.py\n"
        "# Train = V5 train + v10 (bo hop khong lo, khung 960) + UA-DETRAC (car/van->car, bus)\n"
        "# Val/test = giong het V5+clean (val cam_07, test V5)\n"
        f"path: {OUT}\ntrain: images/train\nval:   images/val\ntest:  images/test\n\n"
        "nc: 4\nnames:\n  0: motorcycle\n  1: car\n  2: bus\n  3: truck\n")
    v5_boxes = np.zeros(4, int)
    for n in v5:
        for row in read_labels(OUT / "labels" / "train" / (Path(n).stem + ".txt")):
            v5_boxes[int(row[0])] += 1
    manifest = {
        "built_at": datetime.now().isoformat(timespec="seconds"), "builder": "scripts/data_prep/build_v5fix.py",
        "seed": SEED, "yaml": str(YAML.relative_to(ROOT)),
        "counts": {"v5_train": len(v5), "v10_train": v10["n_kept"], "detrac_train": det["n_kept"],
                   "train_total": len(v5) + v10["n_kept"] + det["n_kept"], "val": len(val), "test": len(test)},
        "boxes": {"v5": dict(zip(["motorcycle", "car", "bus", "truck"], v5_boxes.tolist())),
                  "v10": v10["boxes"], "detrac": det["boxes"]},
        "params": {"v10_max_box_area": V10_MAX_BOX_AREA, "v10_n": V10_N, "v10_canvas": V10_CANVAS,
                   "detrac_per_seq": DETRAC_PER_SEQ, "detrac_candidates": DETRAC_CANDIDATES,
                   "detrac_map": DETRAC_MAP, "filter_weights": FILTER_WEIGHTS,
                   "filter_conf_moto": FILTER_CONF_MOTO, "filter_conf_veh": FILTER_CONF_VEH,
                   "filter_max_unlabeled": FILTER_MAX_UNLABELED},
        "v10": {k: v for k, v in v10.items() if k != "files"}, "detrac": {k: v for k, v in det.items() if k != "files"},
        "val_files": val, "test_files": test, "v10_files": v10["files"], "detrac_files": det["files"],
    }
    (REPORT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    preview(v10["files"] and [Path(f).stem + ".jpg" for f in v10["files"]], REPORT / "preview_v10.jpg")
    preview(det["files"], REPORT / "preview_detrac.jpg")
    print(json.dumps(manifest["counts"]), json.dumps(manifest["boxes"]))


if __name__ == "__main__":
    main()
