"""Chia lại val/test THEO CAMERA cho project cuối kì (bộ V6split, dựa trên V6final_mio).

Vì sao: test cũ chỉ có SOICT cam_09/cam_10 (80% xe máy, gần như không có xe tải/xe buýt), val chỉ có SOICT
cam_07, nên best.pt được chọn theo đúng 1 camera SOICT. Bộ mới:

  train   = V6final_mio train  − ảnh v10 của 9 điểm quay rút ra, bù lại bằng ảnh v10 của các điểm khác
  val     = SOICT cam_07 (400 ảnh rải đều trong 1.000) + v10 4 điểm VAL_SITES
  test    = SOICT cam_09/cam_10 (Test A, giữ nguyên để so với bài NCKH)
  test_vn_v10 = v10 5 điểm TEST_SITES (test phụ cho xe tải/xe buýt; LƯU Ý: 95% ảnh v10 có khung/chữ
                vẽ sẵn của hệ thống camera gốc, và mỗi ảnh chỉ gán nhãn xe "sự kiện")

Chia theo điểm quay (v10: mọi camera cùng một điểm đi cùng nhau). Ảnh v10 val/test che xám chỗ yolov8x
thấy xe (conf ≥ 0.5) mà nhãn không có, như đã làm với train.
Video Đà Nẵng (raw_vn_cctv) đã thử làm test rồi BỎ: camera đặt thấp gần ngang tầm mắt, ngoài phạm vi
camera trên cao của project, đêm chói/mưa và thiếu nhãn nhiều.

Chạy: python scripts/data_prep/build_v6split.py
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "merged_v6final_mio"
V10_SRC = ROOT / "data" / "merged_v5plus_clean"
V10_MAN = ROOT / "reports" / "v5plus_clean" / "manifest_full.json"
OUT = ROOT / "data" / "merged_v6split"
REPORT = ROOT / "reports" / "v6split"

SEED = 0
TEST_SITES = ["nga3duong", "nga4hamthang", "comkimngan", "cafeminghouse", "hungvuong"]
VAL_SITES = ["hoaphu", "docvinhhanh", "cafestart", "nga3muitau"]
SOICT_VAL_N = 400
V10_MAX_BOX_AREA, V10_CANVAS = 0.35, 960
GRAY = (114, 114, 114)
TEACHER, TEACHER_CONF = "yolov8x.pt", 0.5
COCO2V = {3: 0, 2: 1, 5: 2, 7: 3}
NAMES = ["motorcycle", "car", "bus", "truck"]


def site_of(name: str) -> str | None:
    m = re.search(r"(?:gen|general|veh)[_-]([a-z0-9]+?)\d*[_-]", name)
    return m.group(1) if m else None


def link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def read_labels(p: Path) -> np.ndarray:
    if not p.exists() or p.stat().st_size == 0:
        return np.zeros((0, 5))
    return np.loadtxt(p, ndmin=2)


def write_labels(p: Path, rows) -> None:
    p.write_text("\n".join(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in rows))


def copy_pair(src_root: Path, split: str, stem_img: str, dst_split: str, new_name: str | None = None):
    img = src_root / "images" / split / stem_img
    name = new_name or img.name
    link(img, OUT / "images" / dst_split / name)
    shutil.copyfile(src_root / "labels" / split / (img.stem + ".txt"),
                    OUT / "labels" / dst_split / (Path(name).stem + ".txt"))


def iou_ioa(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b))), np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    aa = (a[:, 2:] - a[:, :2]).prod(1)[:, None]
    ab = (b[:, 2:] - b[:, :2]).prod(1)[None, :]
    return inter / (aa + ab - inter + 1e-9), inter / (aa + 1e-9)


def mask_unlabeled(det, img, rows, stats, key):
    """Che xám xe yolov8x thấy (conf ≥ 0.5) mà nhãn không có; trả lại pixel trong hộp thật."""
    h, w = img.shape[:2]
    g = np.array([[(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]
                  for _, x, y, bw, bh in rows]).reshape(-1, 4)
    r = det.predict(img, imgsz=960, conf=TEACHER_CONF, classes=list(COCO2V), verbose=False, half=True)[0]
    d = r.boxes.xyxy.cpu().numpy()
    iou, ioa = iou_ioa(d, g)
    miss = ~((iou.max(1) >= 0.3) | (ioa.max(1) >= 0.6)) if len(g) else np.ones(len(d), bool)
    if miss.any():
        orig = img.copy()
        for x1, y1, x2, y2 in d[miss]:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), GRAY, -1)
        for x1, y1, x2, y2 in g:
            x1, y1, x2, y2 = max(int(x1), 0), max(int(y1), 0), min(int(np.ceil(x2)), w), min(int(np.ceil(y2)), h)
            img[y1:y2, x1:x2] = orig[y1:y2, x1:x2]
        stats[f"{key}_masked_boxes"] += int(miss.sum())
        stats[f"{key}_masked_images"] += 1
    return img


def place_v10_on_canvas(rec, rng):
    img = cv2.imread(str(V10_SRC / "images" / "train" / rec["file"]))
    h, w = img.shape[:2]
    canvas = np.full((V10_CANVAS, V10_CANVAS, 3), GRAY, np.uint8)
    ox, oy = rng.randint(0, V10_CANVAS - w), rng.randint(0, V10_CANVAS - h)
    canvas[oy:oy + h, ox:ox + w] = img
    stem = Path(rec["file"]).stem
    cv2.imwrite(str(OUT / "images" / "train" / f"{stem}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    lab = read_labels(V10_SRC / "labels" / "train" / f"{stem}.txt")
    write_labels(OUT / "labels" / "train" / f"{stem}.txt",
                 [(c, (x * w + ox) / V10_CANVAS, (y * h + oy) / V10_CANVAS, bw * w / V10_CANVAS, bh * h / V10_CANVAS)
                  for c, x, y, bw, bh in lab])


def count_boxes(split):
    c = Counter()
    for lp in (OUT / "labels" / split).glob("*.txt"):
        for line in lp.read_text().splitlines():
            if line.strip():
                c[NAMES[int(line.split()[0])]] += 1
    return dict(c)


def main():
    from ultralytics import YOLO

    rng = random.Random(SEED)
    held = set(TEST_SITES) | set(VAL_SITES)
    splits = ("train", "val", "test", "test_vn_v10")
    if OUT.exists():
        shutil.rmtree(OUT)
    for sp in splits:
        (OUT / "images" / sp).mkdir(parents=True)
        (OUT / "labels" / sp).mkdir(parents=True)

    # ---- train: bỏ v10 của điểm rút ra, bù v10 điểm khác
    removed = Counter()
    kept_v10 = set()
    for img in (BASE / "images" / "train").glob("*.jpg"):
        if img.name.startswith(("v10_", "v10r_")):
            s = site_of(img.name)
            assert s, img.name
            if s in held:
                removed["v10r" if img.name.startswith("v10r_") else "v10"] += 1
                continue
            if img.name.startswith("v10_"):
                kept_v10.add(img.name)
        copy_pair(BASE, "train", img.name, "train")
    man = json.loads(V10_MAN.read_text())
    by_cam = defaultdict(list)
    site_recs = defaultdict(list)
    for rec in man["v10_files"]:
        s = site_of(rec["file"])
        site_recs[s].append(rec)
        if s in held or rec["file"] in kept_v10:
            continue
        lab = read_labels(V10_SRC / "labels" / "train" / (Path(rec["file"]).stem + ".txt"))
        if len(lab) == 0 or (lab[:, 3] * lab[:, 4]).max() > V10_MAX_BOX_AREA:
            continue
        by_cam[rec["camera_id"]].append(rec)
    for cam in by_cam:
        rng.shuffle(by_cam[cam])
    backfill, cams = [], sorted(by_cam)
    while len(backfill) < removed["v10"] and any(by_cam.values()):
        for cam in cams:
            if by_cam[cam] and len(backfill) < removed["v10"]:
                backfill.append(by_cam[cam].pop())
    for rec in backfill:
        place_v10_on_canvas(rec, rng)
    print(f"[1/3] train: bỏ {dict(removed)} ảnh v10 của điểm rút ra, bù {len(backfill)} ảnh v10 điểm khác", flush=True)

    # ---- test A + SOICT val
    for img in (BASE / "images" / "test").glob("*.jpg"):
        copy_pair(BASE, "test", img.name, "test")
    sv = sorted(p.name for p in (BASE / "images" / "val").glob("*.jpg"))
    sv = [sv[int(k * len(sv) / SOICT_VAL_N)] for k in range(SOICT_VAL_N)]
    for n in sv:
        copy_pair(BASE, "val", n, "val")

    # ---- v10 val/test (ảnh gốc 640, giữ cả hộp lớn vì đó là xe thật; che xám xe thiếu nhãn)
    det = YOLO(str(ROOT / TEACHER))
    v10_stats = Counter()
    v10_test, v10_val = [], []
    for s in TEST_SITES + VAL_SITES:
        for rec in site_recs[s]:
            dst = "test_vn_v10" if s in TEST_SITES else "val"
            stem = Path(rec["file"]).stem
            img = cv2.imread(str(V10_SRC / "images" / "train" / rec["file"]))
            rows = [tuple(r) for r in read_labels(V10_SRC / "labels" / "train" / f"{stem}.txt")]
            img = mask_unlabeled(det, img, rows, v10_stats, dst)
            cv2.imwrite(str(OUT / "images" / dst / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_labels(OUT / "labels" / dst / f"{stem}.txt", rows)
            (v10_test if s in TEST_SITES else v10_val).append(rec["file"])
    print(f"[2/3] v10 test {len(v10_test)} ảnh ({len(TEST_SITES)} điểm), val {len(v10_val)} ảnh ({len(VAL_SITES)} điểm)",
          dict(v10_stats), flush=True)

    # ---- kiểm tra rò rỉ: không điểm v10 / camera nào nằm ở 2 tập
    def sources(sp):
        out = set()
        for p in (OUT / "images" / sp).glob("*.jpg"):
            if p.name.startswith(("v10_", "v10r_")):
                out.add("v10:" + site_of(p.name))
            elif p.name.startswith("soict"):
                out.add(p.name.split("_")[0].replace("night", "") + ":" + re.search(r"cam_\d+", p.name).group(0)
                        + (":night" if "night" in p.name else ""))
        return out
    src = {sp: sources(sp) for sp in ("train", "val", "test", "test_vn_v10")}
    for a in src:
        for b in src:
            if a < b:
                assert not (src[a] & src[b]), (a, b, src[a] & src[b])
    names = {sp: {p.name for p in (OUT / "images" / sp).glob("*.jpg")} for sp in src}
    assert not (names["train"] & (names["val"] | names["test"] | names["test_vn_v10"]))

    yaml_head = "nc: 4\nnames:\n  0: motorcycle\n  1: car\n  2: bus\n  3: truck\n"
    for tag, test_split in (("", "test"), ("_vn_v10", "test_vn_v10")):
        (ROOT / "data" / f"data_v6split{tag}.yaml").write_text(
            f"# V6split — chia val/test theo camera (build_v6split.py). test = {test_split}\n"
            f"path: {OUT}\ntrain: images/train\nval:   images/val\ntest:  images/{test_split}\n\n" + yaml_head)

    REPORT.mkdir(parents=True, exist_ok=True)
    counts = {sp: len(list((OUT / "images" / sp).glob("*.jpg"))) for sp in splits}
    man_out = {"built_at": datetime.now().isoformat(timespec="seconds"), "base": str(BASE.relative_to(ROOT)),
               "counts": counts, "boxes": {sp: count_boxes(sp) for sp in splits},
               "test_sites_v10": TEST_SITES, "val_sites_v10": VAL_SITES,
               "danang": "đã bỏ (camera thấp, ngoài phạm vi, nhãn thiếu nhiều)",
               "v10_mask_stats": dict(v10_stats),
               "v10_note": "95% ảnh v10 có khung/chữ vẽ sẵn của hệ thống camera gốc; test_vn_v10 chỉ là test phụ",
               "train_removed_v10": dict(removed), "train_backfill_v10": len(backfill),
               "sources": {k: sorted(v) for k, v in src.items()},
               "label_note": "v10 val/test: che xám xe yolov8x thấy mà nhãn không có (conf 0.5)",
               "licenses": {"v10": "CC BY 4.0", "MIO-TCD": "CC BY-NC-SA 4.0"}}
    (REPORT / "manifest.json").write_text(json.dumps(man_out, indent=1, ensure_ascii=False))
    print("[3/3]", json.dumps({"counts": counts, "boxes": man_out["boxes"]}, ensure_ascii=False))

    col = [(0, 200, 0), (255, 140, 0), (0, 0, 255), (255, 0, 255)]
    for sp, n in (("test_vn_v10", 16),):
        fs = sorted((OUT / "images" / sp).glob("*.jpg"))
        tiles = []
        for p in random.Random(1).sample(fs, n):
            img = cv2.imread(str(p))
            h, w = img.shape[:2]
            for line in (OUT / "labels" / sp / (p.stem + ".txt")).read_text().splitlines():
                c, x, y, bw, bh = map(float, line.split())
                cv2.rectangle(img, (int((x - bw / 2) * w), int((y - bh / 2) * h)),
                              (int((x + bw / 2) * w), int((y + bh / 2) * h)), col[int(c)], 2)
            tiles.append(cv2.resize(img, (480, 270)))
        cv2.imwrite(str(REPORT / f"preview_{sp}.jpg"),
                    np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, n, 4)]), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
