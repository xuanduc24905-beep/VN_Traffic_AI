#!/usr/bin/env python3
"""Bootstrap khoang tin cay cho hieu so mAP@0,5 giua hai mo hinh (lay mau lai anh kiem thu, khong huan luyen lai).

Doc du doan da luu boi 80_unified_coco_eval.py. AP@0,5 tinh theo quy uoc COCO (ghep tham lam theo diem
trong tung anh va tung lop, noi suy 101 diem), duoc doi chieu voi pycocotools tren toan tap.
Cung in Precision/Recall tai nguong tin cay cho F1 trung binh cao nhat (cung dinh nghia voi Ultralytics).

Chay:  python scripts/verify/81_bootstrap_ci.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/xuand/vn_traffic_ai")
D = ROOT / "outputs/verify/unified_eval"
B = 1000
REC = np.linspace(0, 1, 101)


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-12)


def load_gt(sname):
    gt = json.loads((D / f"gt_{sname}.json").read_text())
    n = len(gt["images"])
    g = [[np.zeros((0, 4)) for _ in range(4)] for _ in range(n)]
    tmp = [[[] for _ in range(4)] for _ in range(n)]
    for a in gt["annotations"]:
        x, y, w, h = a["bbox"]
        tmp[a["image_id"]][a["category_id"] - 1].append([x, y, x + w, y + h])
    for i in range(n):
        for c in range(4):
            if tmp[i][c]:
                g[i][c] = np.array(tmp[i][c])
    return g


def match(gt, sname, name):
    """Tra ve per-image, per-class: (scores, tp flags), so GT."""
    z = np.load(D / f"pred_{sname}_{name}.npz")
    n = len(gt)
    S = [[None] * 4 for _ in range(n)]
    for i in range(n):
        p = z[f"b{i}"]
        for c in range(4):
            q = p[p[:, 5] == c]
            q = q[np.argsort(-q[:, 4], kind="mergesort")]
            g = gt[i][c]
            tp = np.zeros(len(q), bool)
            if len(q) and len(g):
                M = iou(q[:, :4], g)
                used = np.zeros(len(g), bool)
                for k in range(len(q)):
                    m = np.where(used, -1, M[k])
                    j = int(m.argmax())
                    if m[j] >= 0.5:
                        used[j] = True; tp[k] = True
            S[i][c] = (q[:, 4], tp)
    return S


def ap_from(scores, tps, ngt):
    if ngt == 0:
        return np.nan
    o = np.argsort(-scores, kind="mergesort")
    tp = tps[o]; ctp = np.cumsum(tp); cfp = np.cumsum(~tp)
    rc = ctp / ngt; pr = ctp / np.maximum(ctp + cfp, 1e-12)
    pr = np.maximum.accumulate(pr[::-1])[::-1] if len(pr) else pr
    idx = np.searchsorted(rc, REC, side="left")
    q = np.where(idx < len(pr), pr[np.minimum(idx, len(pr) - 1)], 0.0)
    return q.mean()


def mAP(S, gt, ids):
    aps = []
    for c in range(4):
        sc = np.concatenate([S[i][c][0] for i in ids]); tp = np.concatenate([S[i][c][1] for i in ids])
        ng = sum(len(gt[i][c]) for i in ids)
        aps.append(ap_from(sc, tp, ng))
    return np.nanmean(aps), aps


def pr_at_best_f1(S, gt):
    th = np.linspace(0, 1, 1000)
    P, R = [], []
    for c in range(4):
        sc = np.concatenate([S[i][c][0] for i in range(len(gt))]); tp = np.concatenate([S[i][c][1] for i in range(len(gt))])
        ng = sum(len(gt[i][c]) for i in range(len(gt)))
        o = np.argsort(-sc); sc, tp = sc[o], tp[o]
        ctp = np.cumsum(tp); n = np.arange(1, len(tp) + 1)
        k = np.searchsorted(-sc, -th, side="right")  # so du doan co diem >= th
        t = np.where(k > 0, ctp[np.maximum(k - 1, 0)], 0)
        P.append(np.where(k > 0, t / np.maximum(k, 1), 1.0)); R.append(t / max(ng, 1))
    P, R = np.array(P), np.array(R)
    f1 = (2 * P * R / (P + R + 1e-16)).mean(0)
    j = int(f1.argmax())
    return {"conf": float(th[j]), "P": float(P[:, j].mean()), "R": float(R[:, j].mean())}


def main():
    res = json.loads((D / "unified_eval.json").read_text())
    out = {}
    pairs = {"v5plus_test": [("r_clspw_v2", "exp_baseline"), ("r_clspw_v2", "ctrl_cw0_s0"),
                             ("ctrl_cw0_s0", "exp_baseline"), ("r_clspw_v2", "frcnn_300"),
                             ("exp_baseline", "frcnn_300")],
             "aerial_test": [("r_clspw_v2", "exp_baseline"), ("r_clspw_v2", "frcnn_300")]}
    rng = np.random.default_rng(0)
    for sname, prs in pairs.items():
        gt = load_gt(sname); n = len(gt)
        names = sorted({x for p in prs for x in p})
        S = {m: match(gt, sname, m) for m in names}
        full = {}
        for m in names:
            v, aps = mAP(S[m], gt, range(n))
            full[m] = {"mAP50_own": float(v), "pycoco_mAP50": res[f"{sname}/{m}"]["mAP50"],
                       "AP50_own": [float(a) for a in aps], **pr_at_best_f1(S[m], gt)}
        idx = [rng.integers(0, n, n) for _ in range(B)]
        boot = {m: np.array([mAP(S[m], gt, ii)[0] for ii in idx]) for m in names}
        diffs = {}
        for a, b in prs:
            d = boot[a] - boot[b]
            diffs[f"{a} - {b}"] = {"point": full[a]["mAP50_own"] - full[b]["mAP50_own"],
                                   "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                   "p_le_0": float((d <= 0).mean())}
        out[sname] = {"models": full, "diffs": diffs, "B": B}
        print(sname, json.dumps(out[sname], indent=1), flush=True)
    (D / "bootstrap_ci.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
