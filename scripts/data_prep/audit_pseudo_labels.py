"""Đo chất lượng nhãn giả yolov8x (build_v6split_pseudo.py) bằng nhãn NGƯỜI GÁN sẵn có của các khung đó.

Mọi nguồn khung nhãn giả đều có nhãn Roboflow gốc (SOICT, VNv3, camera Đà Nẵng) — chỉ dùng để KIỂM TRA,
không dùng để train trong bản nhãn giả. Với mỗi ngưỡng: so hộp nhãn giả (sau luật gộp người lái, chưa tô xám)
với hộp người gán (IoU ≥ 0,5):
  - recall theo lớp người gán: % hộp người gán được 1 hộp nhãn giả bắt trúng (không xét lớp)
  - precision: % hộp nhãn giả trùng 1 hộp người gán; ma trận nhầm lớp trên các cặp trùng
  - % hộp người gán bị tô xám (nằm trong vùng xám ≥ 50%) — xe bị "giấu" chứ không thành nền sai
Lưu ý: quy ước hộp có thể khác giữa các bộ (vd. xe máy Đà Nẵng có gồm người lái hay không) → số liệu chỉ
để so ngưỡng và thấy lỗi lớn (vd. xe tải VN bị gán bus).

Chạy: python scripts/data_prep/audit_pseudo_labels.py → reports/v6split_pseudo/audit.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data_prep"))
import build_v6split_pseudo as B  # noqa: E402

NAMES = ["motorcycle", "car", "bus", "truck"]
REMAP = B.REMAP
SRC_KEY = {"raw_soict2024": "soict_day", "raw_soict_night": "soict_night", "raw_vnv3": "vnv3", "raw_vn_cctv": "vn_cctv"}


def iou_m(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    i = np.clip(rb - lt, 0, None).prod(2)
    return i / ((a[:, 2:] - a[:, :2]).prod(1)[:, None] + (b[:, 2:] - b[:, :2]).prod(1)[None] - i + 1e-9)


def pseudo_boxes(dets, thr):
    """Áp đúng luật của build_v6split_pseudo: trả (hộp nhãn [cls, xyxy], hộp xám xyxy)."""
    persons = [d[2:] for d in dets if d[0] == B.COCO["person"]]
    trucks = [(d[1], np.array(d[2:])) for d in dets if d[0] == B.COCO["truck"]]
    lab, gray = [], []
    for c, s, *b in dets:
        if c == B.COCO["person"]:
            continue
        if c == B.COCO["truck"]:
            gray.append(b)
            continue
        box = B.rider_union(b, persons)[0] if c == B.COCO["motorcycle"] else b
        amb = c != B.COCO["motorcycle"] and any(ts > s and B.iou(np.array(b), tb) >= 0.6 for ts, tb in trucks)
        (gray if (s < thr or amb) else lab).append(box if (s < thr or amb) else [B.TO_V[c], *box])
    return lab, gray


def main():
    cache = json.loads(B.CACHE.read_text())
    out = {}
    for thr in (0.5, 0.6, 0.7):
        res = {}
        for raw, key in SRC_KEY.items():
            files = [k for k in cache if k.startswith(f"data/{raw}/")]
            rec_hit, rec_n, rec_gray = Counter(), Counter(), Counter()
            prec_hit, prec_n = Counter(), Counter()
            cm = np.zeros((4, 4), int)  # hàng: lớp người gán, cột: lớp nhãn giả
            for f in files:
                p = ROOT / f
                lp = p.parent.parent / "labels" / f"{p.stem}.txt"
                h, w = cv2.imread(str(p)).shape[:2]
                gt = []
                if lp.exists():
                    for line in lp.read_text().splitlines():
                        v = line.split()
                        if len(v) == 5 and int(v[0]) in REMAP[raw]:
                            c, x, y, bw, bh = int(v[0]), *map(float, v[1:])
                            gt.append([REMAP[raw][c], (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
                gt = np.array(gt).reshape(-1, 5)
                lab, gray = pseudo_boxes(cache[f], thr)
                lab = np.array(lab).reshape(-1, 5)
                io = iou_m(gt[:, 1:], lab[:, 1:])
                for i, g in enumerate(gt):
                    c = int(g[0])
                    rec_n[c] += 1
                    if len(lab) and io[i].max() >= 0.5:
                        rec_hit[c] += 1
                        cm[c, int(lab[io[i].argmax(), 0])] += 1
                    else:
                        gx = np.array(gray).reshape(-1, 4)
                        if len(gx):
                            lt = np.maximum(gx[:, :2], g[1:3]); rb = np.minimum(gx[:, 2:], g[3:])
                            cov = np.clip(rb - lt, 0, None).prod(1) / max((g[3] - g[1]) * (g[4] - g[2]), 1e-9)
                            rec_gray[c] += int(cov.max() >= 0.5)
                for j, l in enumerate(lab):
                    prec_n[int(l[0])] += 1
                    prec_hit[int(l[0])] += int(len(gt) > 0 and io[:, j].max() >= 0.5)
            r = lambda a, b: round(a / b, 3) if b else None  # noqa: E731
            res[key] = {"frames": len(files),
                        "recall": {NAMES[c]: r(rec_hit[c], rec_n[c]) for c in range(4)},
                        "gt_boxes": {NAMES[c]: rec_n[c] for c in range(4)},
                        "gt_grayed": {NAMES[c]: r(rec_gray[c], rec_n[c]) for c in range(4)},
                        "precision": {NAMES[c]: r(prec_hit[c], prec_n[c]) for c in range(4)},
                        "pseudo_boxes": {NAMES[c]: prec_n[c] for c in range(4)},
                        "confusion_gt_rows_pseudo_cols": cm.tolist()}
        out[str(thr)] = res
    (B.REPORT_ROOT / "audit.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for thr, res in out.items():
        print(f"== ngưỡng {thr}")
        for k, v in res.items():
            print(f"  {k:12s} recall {v['recall']} | xám {v['gt_grayed']}\n{'':15s}prec {v['precision']} | cm {v['confusion_gt_rows_pseudo_cols']}")


if __name__ == "__main__":
    main()
