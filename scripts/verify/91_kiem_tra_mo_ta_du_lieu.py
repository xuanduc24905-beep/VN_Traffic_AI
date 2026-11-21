#!/usr/bin/env python3
"""Kiem tra cac cau mo ta du lieu va diem luu duoc sua trong ban v3 (dot sua truoc phan bien).

Ghi ra outputs/verify/kiem_tra_mo_ta_du_lieu.json:
  - buoc so thu tu khung trong ten tep cua tung camera SOICT (lay mau 5 khung hay lay het),
  - so video Can Tho theo tap,
  - so anh Vietnam Vehicles v10 chi co nhan xe ba gac (bi bo khi gop V5+),
  - so hop trung lap hoan toan trong tep nhan V5+,
  - epoch cua diem luu tot nhat (fitness = mAP@0.5:0.95) cua cac luot YOLO trong bai.
Chay: python3 scripts/verify/91_kiem_tra_mo_ta_du_lieu.py
"""
import collections
import csv
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V5P = ROOT / "data/merged_v5plus"
out = {}

# 1. Buoc so thu tu khung cua tung camera SOICT
steps = {}
for sp in ("train", "val", "test"):
    g = collections.defaultdict(list)
    for f in os.listdir(V5P / "images" / sp):
        m = re.match(r"(soict24_cam_\d+|soictnight_cam_\d+)_(\d+)_", f)
        if m and f.endswith(".jpg"):
            g[m.group(1)].append(int(m.group(2)))
    for cam, idx in sorted(g.items()):
        idx.sort()
        d = collections.Counter(b - a for a, b in zip(idx, idx[1:]))
        steps[f"{sp}/{cam}"] = {"n_images": len(idx), "first": idx[0], "last": idx[-1],
                                "step_counts": {str(k): v for k, v in sorted(d.items())}}
out["frame_index_steps"] = steps

# 2. Video Can Tho theo tap
cantho = {}
for sp in ("train", "val"):
    c = collections.Counter(re.sub(r"_mp4-\d+_jpg.*", "", f)[len("cantho_"):]
                            for f in os.listdir(V5P / "images" / sp) if f.startswith("cantho_") and f.endswith(".jpg"))
    cantho[sp] = dict(c)
out["cantho_videos"] = cantho
out["cantho_note"] = "4 video (CoopMart 3 khung gio va camDLHB) cung mot camera; xem anh cantho_4_video.jpg."

# 3. v10: anh chi co xe ba gac (lop 2 trong ban goc) va anh rong
v10 = {}
for sp in ("train", "valid", "test"):
    d = ROOT / "data/raw_vietnam_vehicles_v10" / sp / "labels"
    only, empty, tot = 0, 0, 0
    for f in os.listdir(d):
        cls = [l.split()[0] for l in open(d / f) if l.strip()]
        tot += 1
        if not cls:
            empty += 1
        elif set(cls) == {"2"}:
            only += 1
    v10[sp] = {"images": tot, "only_xe_ba_gac": only, "empty": empty}
v10["kept_in_v5plus"] = sum(1 for f in os.listdir(V5P / "images/train") if f.startswith("v10_") and f.endswith(".jpg"))
out["v10_drop"] = v10

# 4. Hop trung lap hoan toan trong tep nhan
dup = {}
for sp in ("train", "val", "test"):
    n_dup, cls_dup = 0, collections.Counter()
    for f in os.listdir(V5P / "labels" / sp):
        if not f.endswith(".txt"):
            continue
        lines = [tuple(round(float(x), 6) for x in l.split()) for l in open(V5P / "labels" / sp / f) if l.strip()]
        for k, v in collections.Counter(lines).items():
            if v > 1:
                n_dup += v - 1
                cls_dup[int(k[0])] += v - 1
    dup[sp] = {"duplicate_boxes": n_dup, "by_class": dict(cls_dup)}
out["duplicate_boxes"] = dup

# 5. Epoch cua diem luu tot nhat
best = {}
for r in ["exp_baseline", "p4_scale_from_baseline", "p4_imgsz1280_from_baseline", "r_p2", "r_progressive_s1",
          "r_progressive", "r_clspw_v2", "ctrl_cw0_s0", "ctrl_cw0_s1", "ctrl_cw0_s2",
          "ctrl_cw05_s0", "ctrl_cw05_s1", "ctrl_cw05_s2"]:
    rows = [{k.strip(): v for k, v in x.items()} for x in csv.DictReader(open(ROOT / "runs/detect" / r / "results.csv"))]
    m = [float(x["metrics/mAP50-95(B)"]) for x in rows]
    b = max(range(len(m)), key=m.__getitem__)
    best[r] = {"epochs_run": len(m), "best_epoch": b + 1, "val_mAP50_95_per_epoch": [round(v, 4) for v in m]}
out["best_epoch_fitness_mAP50_95"] = best

p = ROOT / "outputs/verify/kiem_tra_mo_ta_du_lieu.json"
p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(p)
for k in ("cantho_videos", "v10_drop", "duplicate_boxes"):
    print(k, out[k])
print({k: (v["best_epoch"], v["epochs_run"]) for k, v in best.items()})
