"""T2 + NHÃN GIẢ từ yolov8x (chưng cất kiến thức COCO trên khung VN chưa dùng để train).

Ý tưởng: T2 giữ kiến thức COCO bằng L2-SP + 5.000 ảnh COCO, nhưng ảnh COCO khác xa camera VN. Ở đây lấy
khung VN CHƯA dùng để train, cho yolov8x (COCO, imgsz 1280) gán nhãn, rồi thêm vào bộ T2 → model học
"cái yolov8x thấy" ngay trên cảnh camera VN.

Nguồn khung (chỉ camera/điểm đang ở train; loại SOICT cam_07/09/10 = val/test, loại v10):
  - SOICT ngày cam_01/02/04/06 (data/raw_soict2024): bộ train đã lấy khung chỉ số ≡ 1 (mod 5); ở đây lấy
    khung ≡ 3 (mod 5) → cách khung đã dùng ≥ 2 khung.
  - SOICT đêm cam_03/05/08 (data/raw_soict_night): mỗi khung có tới 6 bản Roboflow; chỉ lấy khung chưa có
    bản nào trong train, và cách khung đã chọn trước đó ≥ 2 khung; mỗi khung lấy 1 bản (tên nhỏ nhất).
  - VNv3 (data/raw_vnv3): khung ≡ 3 (mod 5) (train đã lấy ≡ 1).
  - Camera Đà Nẵng (data/raw_vn_cctv, 0511.vn): đã bỏ khỏi test nên dùng được để train; lấy đều
    N_CCTV khung trên ~29.000 khung. Camera thấp hơn phạm vi chính → giới hạn số khung.
Kiểm tra trùng: dHash 64 bit với mọi ảnh val/test/test_vn_v10 của V6split, tập ngoài (ext_benchmark) và
khung video ngã tư dùng để chấm (deploy_frames); khoảng cách Hamming ≤ 6 → bỏ khung. Tên file đều mới
(tiền tố pl_).

Quy tắc nhãn (yolov8x, imgsz 1280, conf 0.3, NMS theo lớp, iou 0.7):
  - car, bus: conf ≥ NGƯỠNG (mặc định 0,6) → nhãn; 0,3 ≤ conf < NGƯỠNG → tô xám (114) vùng hộp.
  - motorcycle: như trên; hộp nhãn GỘP hộp person ngồi trên xe (cùng luật rider_union với
    build_v6split_coco.py; person conf ≥ 0,3) để khớp quy ước SOICT (hộp xe máy gồm người lái).
  - truck (mọi conf ≥ 0,3): tô xám — COCO gộp xe bán tải vào truck, SOICT gọi bán tải là car.
  - car/bus bị một hộp truck tự tin hơn trùng IoU ≥ 0,6 → coi là mập mờ, tô xám thay vì gán nhãn.
  - Sau khi tô xám, trả lại pixel trong các hộp có nhãn (vùng xám có thể đè lên xe có nhãn).
  - Khung không còn hộp nhãn nào → bỏ.
Giới hạn: yolov8x vẫn sót xe rất nhỏ/xa và xe bị che → các xe đó thành "nền" trong ảnh nhãn giả.
Kiểm tra bằng nhãn người gán sẵn có (scripts/data_prep/audit_pseudo_labels.py): ở SOICT, ngưỡng 0,6 chỉ bắt
~3% xe máy (và chỉ ~8% được tô xám), ~29% hộp "bus" là xe tải thật → bản nhãn giả thuần dạy model coi xe máy
nhỏ là nền. Vì vậy có thêm chế độ --mode ghep:
  - giữ toàn bộ nhãn người gán (Roboflow gốc, đổi về 4 lớp) của khung đó,
  - thêm hộp yolov8x conf ≥ NGƯỠNG (car/bus/xe máy gộp người lái) KHÔNG trùng hộp người gán nào
    (IoU < 0,3 và < 60% diện tích nằm trong 1 hộp người gán) → bù xe người gán bỏ sót,
  - tô xám hộp yolov8x không trùng có conf 0,3–NGƯỠNG và mọi truck yolov8x không trùng.
Mốc chấm "ngã tư" cũng là yolov8x → model học theo yolov8x sẽ được lợi ở thước đo đó; phải đọc cùng SOICT/v10.

Bộ kết quả = bộ T2 (data/merged_v6split_coco_tx2, hardlink cả ảnh lẫn cache .npy) + khung nhãn giả.
Dự đoán yolov8x được lưu cache (reports/v6split_pseudo/pred_cache.json) → dựng các ngưỡng khác không phải
chạy lại yolov8x.

Chạy: python scripts/data_prep/build_v6split_pseudo.py [--conf 0.6] [--mode gia|ghep]
  --mode ghep → thêm hậu tố _ghep vào tên bộ / yaml / báo cáo
  conf 0.6 → data/merged_v6split_pseudo,       data/data_v6split_pseudo.yaml
  conf 0.5 → data/merged_v6split_pseudo_c50,   data/data_v6split_pseudo_c50.yaml (tương tự 0.7 → _c70)
Báo cáo: reports/v6split_pseudo[_cXX]/manifest.json + preview_*.jpg
"""
from __future__ import annotations

import argparse
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
BASE = ROOT / "data" / "merged_v6split_coco_tx2"  # bộ T2
V6 = ROOT / "data" / "merged_v6split"
REPORT_ROOT = ROOT / "reports" / "v6split_pseudo"
CACHE = REPORT_ROOT / "pred_cache.json"
TEACHER, IMGSZ, LOW = "yolov8x.pt", 1280, 0.3
N_CCTV, SEED = 2000, 0
DAY_CAMS, NIGHT_CAMS = ("01", "02", "04", "06"), ("03", "05", "08")
HELD_CAMS = ("07", "09", "10")
GRAY = (114, 114, 114)
NAMES = ["motorcycle", "car", "bus", "truck"]
COCO = {"person": 0, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
TO_V = {3: 0, 2: 1, 5: 2}
SPLITS = ("train", "val", "test", "test_vn_v10")
# lớp Roboflow gốc → 4 lớp V (motorcycle, car, bus, truck)
REMAP = {"raw_soict2024": {0: 1, 1: 0, 2: 2, 3: 3}, "raw_soict_night": {0: 0, 1: 1, 2: 2, 3: 3},
         "raw_vnv3": {0: 1, 1: 0, 2: 3, 3: 2}, "raw_vn_cctv": {0: 2, 1: 1, 2: 0, 3: 3}}


def human_boxes(p: Path, w: int, h: int) -> np.ndarray:
    """Nhãn người gán gốc của khung (xyxy pixel, lớp V) — [n, 5] [cls, x1, y1, x2, y2]."""
    raw = p.parent.parent.parent.name
    lp = p.parent.parent / "labels" / f"{p.stem}.txt"
    rows = []
    if lp.exists():
        for line in lp.read_text().splitlines():
            v = line.split()
            if len(v) == 5 and int(v[0]) in REMAP[raw]:
                x, y, bw, bh = map(float, v[1:])
                rows.append([REMAP[raw][int(v[0])], (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
    return np.array(rows, dtype=float).reshape(-1, 5)


def frame_key(name: str) -> str:
    return re.split(r"_(?:jpg|png)", name)[0]


def link(src: Path, dst: Path):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def dhash(path: Path) -> int:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def rider_union(moto, persons):
    """moto, persons: [x1, y1, x2, y2]. Luật như build_v6split_coco.rider_union (đổi sang xyxy)."""
    mx1, my1, mx2, my2 = moto
    mh = my2 - my1
    x1, y1, x2, y2, n = mx1, my1, mx2, my2, 0
    for px1, py1, px2, py2 in persons:
        pw, ph = px2 - px1, py2 - py1
        ov = max(0.0, min(mx2, px2) - max(mx1, px1)) / max(pw, 1.0)
        if ov >= 0.5 and py1 < my1 and my1 <= py2 <= my2 + 0.1 * mh and ph < 3 * mh:
            x1, y1, x2 = min(x1, px1), min(y1, py1), max(x2, px2)
            n += 1
    return [x1, y1, x2, y2], n


def iou(a, b):
    lt = np.maximum(a[:2], b[:2])
    rb = np.minimum(a[2:], b[2:])
    inter = np.clip(rb - lt, 0, None).prod()
    return inter / ((a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - inter + 1e-9)


def select_frames() -> tuple[list[tuple[str, Path]], dict]:
    """Trả [(nguồn, đường dẫn ảnh)] và thống kê chọn khung."""
    used = set()
    for sp in SPLITS:
        used |= {frame_key(p.stem) for p in (V6 / "images" / sp).glob("*.jpg")}
    st = Counter()
    out = []
    # SOICT ngày
    for p in sorted((ROOT / "data" / "raw_soict2024").glob("*/images/*.jpg")):
        m = re.match(r"cam_(\d+)_(\d+)", p.name)
        cam, idx = m.group(1), int(m.group(2))
        assert cam in DAY_CAMS + HELD_CAMS, p.name
        if cam in DAY_CAMS and idx % 5 == 3 and f"soict24_{frame_key(p.name)}" not in used:
            out.append(("soict_day", p))
            st[f"soict_day_cam{cam}"] += 1
    # SOICT đêm: 1 bản / khung, khung chưa có bản nào trong train, cách khung đã chọn ≥ 2
    night = defaultdict(dict)
    for p in sorted((ROOT / "data" / "raw_soict_night").glob("*/images/*.jpg")):
        m = re.match(r"cam_(\d+)_(\d+)", p.name)
        night[m.group(1)].setdefault(int(m.group(2)), p)
    for cam in NIGHT_CAMS:
        last = -10
        for idx in sorted(night[cam]):
            p = night[cam][idx]
            if f"soictnight_{frame_key(p.name)}" in used or idx - last < 2:
                continue
            out.append(("soict_night", p))
            st[f"soict_night_cam{cam}"] += 1
            last = idx
    # VNv3
    for p in sorted((ROOT / "data" / "raw_vnv3").glob("*/images/*.jpg")):
        idx = int(re.match(r"(\d+)", p.name).group(1))
        if idx % 5 == 3 and f"vnv3_{frame_key(p.name)}" not in used:
            out.append(("vnv3", p))
            st["vnv3"] += 1
    # Camera Đà Nẵng: lấy đều theo số thứ tự khung
    cc = sorted((ROOT / "data" / "raw_vn_cctv").glob("*/images/*.jpg"), key=lambda p: int(p.name.split("_")[0]))
    for i in np.linspace(0, len(cc) - 1, N_CCTV).astype(int):
        out.append(("vn_cctv", cc[i]))
    st["vn_cctv"] = N_CCTV
    return out, dict(st)


def dedup(frames):
    """Bỏ khung gần trùng (dHash Hamming ≤ 6) với ảnh dùng để chấm điểm."""
    held = [p for sp in ("val", "test", "test_vn_v10") for p in (V6 / "images" / sp).glob("*.jpg")]
    held += list((ROOT / "data" / "ext_benchmark" / "images").glob("*.jpg"))
    held += list((ROOT / "data" / "deploy_frames").glob("*.jpg"))
    hh = np.array([dhash(p) for p in held], dtype=np.uint64)
    keep, dropped = [], []
    for src, p in frames:
        x = np.bitwise_xor(hh, np.uint64(dhash(p)))
        dist = np.array([bin(int(v)).count("1") for v in x])
        if dist.min() <= 6:
            dropped.append({"file": str(p.relative_to(ROOT)), "match": str(held[int(dist.argmin())].relative_to(ROOT)),
                            "hamming": int(dist.min())})
        else:
            keep.append((src, p))
    return keep, dropped, len(held)


def predict(frames) -> dict:
    """Chạy yolov8x một lần, lưu mọi hộp conf ≥ 0,3 (xyxy pixel, lớp COCO, conf)."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    need = [p for _, p in frames if str(p.relative_to(ROOT)) not in cache]
    if need:
        from ultralytics import YOLO
        m = YOLO(str(ROOT / TEACHER))
        for k in range(0, len(need), 16):
            chunk = need[k:k + 16]
            for p, r in zip(chunk, m.predict([str(x) for x in chunk], imgsz=IMGSZ, conf=LOW, iou=0.7, half=True,
                                             classes=list(COCO.values()), verbose=False)):
                cache[str(p.relative_to(ROOT))] = [[int(c), round(float(s), 4), *[round(v, 1) for v in b]] for c, s, b in
                                                    zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
            if k % 800 == 0:
                print(f"  yolov8x {k + len(chunk)}/{len(need)}", flush=True)
                CACHE.write_text(json.dumps(cache))
        CACHE.write_text(json.dumps(cache))
    return cache


def matches_human(box, hb: np.ndarray) -> bool:
    if not len(hb):
        return False
    b = np.array(box, dtype=float)
    lt = np.maximum(hb[:, 1:3], b[:2])
    rb = np.minimum(hb[:, 3:5], b[2:])
    inter = np.clip(rb - lt, 0, None).prod(1)
    ab = max((b[2:] - b[:2]).prod(), 1e-9)
    ah = (hb[:, 3:5] - hb[:, 1:3]).prod(1)
    return bool(((inter / (ab + ah - inter + 1e-9)).max() >= 0.3) or ((inter / ab).max() >= 0.6))


def make_label(img, dets, thr, st, hb: np.ndarray | None = None):
    """Áp quy tắc nhãn lên ảnh (tô xám tại chỗ). Trả list dòng nhãn YOLO và list hộp nhãn xyxy.
    hb (chế độ ghép): nhãn người gán; hộp yolov8x trùng hộp người gán thì bỏ qua (người gán thắng)."""
    h, w = img.shape[:2]
    persons = [d[2:] for d in dets if d[0] == COCO["person"]]
    trucks = [(d[1], np.array(d[2:])) for d in dets if d[0] == COCO["truck"]]
    gray, rows, keep = [], [], []
    if hb is not None:
        for c, *box in hb:
            x1, y1, x2, y2 = max(box[0], 0), max(box[1], 0), min(box[2], w), min(box[3], h)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            keep.append((x1, y1, x2, y2))
            rows.append(f"{int(c)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            st[f"human_{NAMES[int(c)]}"] += 1
    for c, s, *b in dets:
        if c == COCO["person"]:
            continue
        if hb is not None:
            bb = rider_union(b, persons)[0] if c == COCO["motorcycle"] else b
            if matches_human(bb, hb):
                continue
        if c == COCO["truck"]:
            gray.append(b)
            st["gray_truck"] += 1
            continue
        box = b
        if c == COCO["motorcycle"]:
            box, n = rider_union(b, persons)
            st["moto_with_rider" if n else "moto_no_rider"] += int(s >= thr)
        amb = c != COCO["motorcycle"] and any(ts > s and iou(np.array(b), tb) >= 0.6 for ts, tb in trucks)
        if s < thr or amb:
            gray.append(box)
            st["gray_ambiguous_truck" if (amb and s >= thr) else "gray_low_conf"] += 1
            continue
        x1, y1, x2, y2 = max(box[0], 0), max(box[1], 0), min(box[2], w), min(box[3], h)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        keep.append((x1, y1, x2, y2))
        rows.append(f"{TO_V[c]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        st[f"{'added' if hb is not None else 'box'}_{NAMES[TO_V[c]]}"] += 1
    if gray and rows:
        orig = img.copy()
        for x1, y1, x2, y2 in gray:
            cv2.rectangle(img, (int(x1), int(y1)), (int(np.ceil(x2)), int(np.ceil(y2))), GRAY, -1)
        for x1, y1, x2, y2 in keep:
            y1i, y2i, x1i, x2i = int(y1), int(np.ceil(y2)), int(x1), int(np.ceil(x2))
            img[y1i:y2i, x1i:x2i] = orig[y1i:y2i, x1i:x2i]
    return rows, keep


def main(thr: float, mode: str = "gia"):
    sfx = ("" if abs(thr - 0.6) < 1e-6 else f"_c{int(round(thr * 100))}") + ("_ghep" if mode == "ghep" else "")
    out = ROOT / "data" / f"merged_v6split_pseudo{sfx}"
    yaml = ROOT / "data" / f"data_v6split_pseudo{sfx}.yaml"
    report = ROOT / "reports" / f"v6split_pseudo{sfx}"
    report.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    assert BASE.exists(), "Chưa có bộ T2 (build_v6split_coco_truckx2.py)"

    frames, sel_stats = select_frames()
    print(f"[1/5] chọn {len(frames)} khung:", sel_stats, flush=True)
    frames, dropped, n_held = dedup(frames)
    print(f"[2/5] bỏ {len(dropped)} khung gần trùng với {n_held} ảnh chấm điểm", flush=True)
    cache = predict(frames)
    print("[3/5] đã có dự đoán yolov8x", flush=True)

    if out.exists():
        shutil.rmtree(out)
    for sp in SPLITS:
        (out / "images" / sp).mkdir(parents=True)
        (out / "labels" / sp).mkdir(parents=True)
        for im in (BASE / "images" / sp).iterdir():  # ảnh + cache .npy (giống hệt ảnh → dùng chung)
            if im.suffix in (".jpg", ".npy"):
                link(im, out / "images" / sp / im.name)
        for lp in (BASE / "labels" / sp).glob("*.txt"):
            shutil.copyfile(lp, out / "labels" / sp / lp.name)

    st, by_src, files = Counter(), Counter(), defaultdict(list)
    for src, p in frames:
        img = cv2.imread(str(p))
        hb = human_boxes(p, img.shape[1], img.shape[0]) if mode == "ghep" else None
        rows, _ = make_label(img, cache[str(p.relative_to(ROOT))], thr, st, hb)
        if not rows:
            st["skip_no_label"] += 1
            continue
        name = f"pl_{src}_{p.stem}"
        cv2.imwrite(str(out / "images" / "train" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / "labels" / "train" / f"{name}.txt").write_text("\n".join(rows))
        by_src[src] += 1
        files[src].append(name)
    print("[4/5]", dict(by_src), dict(st), flush=True)

    tb = Counter()
    for lp in (out / "labels" / "train").glob("*.txt"):
        for line in lp.read_text().splitlines():
            if line.strip():
                tb[NAMES[int(line.split()[0])]] += 1
    counts = {sp: len(list((out / "images" / sp).glob("*.jpg"))) for sp in SPLITS}
    yaml.write_text(f"# T2 + khung VN {'nhãn người gán + bù' if mode == 'ghep' else 'nhãn giả'} yolov8x "
                    f"(conf >= {thr}) — build_v6split_pseudo.py --mode {mode}\n"
                    f"path: {out}\ntrain: images/train\nval:   images/val\ntest:  images/test\n\n"
                    "nc: 4\nnames:\n  0: motorcycle\n  1: car\n  2: bus\n  3: truck\n")
    man = {"built_at": datetime.now().isoformat(timespec="seconds"), "base": str(BASE.relative_to(ROOT)),
           "mode": mode, "teacher": TEACHER, "imgsz": IMGSZ, "label_conf": thr, "gray_conf": [LOW, thr],
           "rules": {"car/bus/motorcycle": f"conf >= {thr} → nhãn; {LOW}–{thr} → tô xám",
                     "motorcycle": "gộp hộp person ngồi trên xe (rider_union)",
                     "truck": "tô xám mọi hộp (COCO gộp bán tải)",
                     "car/bus trùng truck tự tin hơn (IoU>=0.6)": "tô xám",
                     **({"ghep": "giữ nhãn người gán; chỉ áp các luật trên cho hộp yolov8x KHÔNG trùng hộp người "
                                 "gán (IoU < 0.3 và IoA < 0.6)"} if mode == "ghep" else {})},
           "selected": sel_stats, "dedup_dropped": dropped, "dedup_held_images": n_held,
           "pseudo_images_by_source": dict(by_src), "label_stats": dict(st),
           "counts": counts, "train_boxes": dict(tb),
           "pseudo_pct_of_train": round(sum(by_src.values()) / counts["train"], 4),
           "pseudo_files": dict(files)}
    (report / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))

    col = [(0, 200, 0), (255, 140, 0), (0, 0, 255), (255, 0, 255)]
    for src in files:
        tiles = []
        for name in random.Random(1).sample(files[src], min(12, len(files[src]))):
            img = cv2.imread(str(out / "images" / "train" / f"{name}.jpg"))
            h, w = img.shape[:2]
            for line in (out / "labels" / "train" / f"{name}.txt").read_text().splitlines():
                c, x, y, bw, bh = map(float, line.split())
                cv2.rectangle(img, (int((x - bw / 2) * w), int((y - bh / 2) * h)),
                              (int((x + bw / 2) * w), int((y + bh / 2) * h)), col[int(c)], 2)
            tiles.append(cv2.resize(img, (480, 300)))
        tiles += [np.zeros_like(tiles[0])] * (12 - len(tiles))
        cv2.imwrite(str(report / f"preview_{src}.jpg"),
                    np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, 12, 4)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("[5/5]", json.dumps({k: man[k] for k in ("counts", "train_boxes", "pseudo_pct_of_train")}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.6, help="ngưỡng conf để gán nhãn (dưới ngưỡng, trên 0,3 → tô xám)")
    ap.add_argument("--mode", choices=["gia", "ghep"], default="gia",
                    help="gia: chỉ nhãn giả yolov8x; ghep: nhãn người gán + bù hộp yolov8x không trùng")
    a = ap.parse_args()
    main(a.conf, a.mode)
