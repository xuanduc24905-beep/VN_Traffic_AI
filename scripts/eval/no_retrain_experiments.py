"""Thử các cách tăng độ chính xác KHÔNG cần train lại, trên cùng tập val.

Mỗi phương pháp chỉ đổi bước suy luận (weights giữ nguyên). Dự đoán được lưu
cache ở results/no_retrain/<method>.pkl, nên chạy lại chỉ tính lại chỉ số.

Phương pháp:
  base960    : suy luận chuẩn, imgsz 960 (giống lúc train V5+)
  sz1152     : tăng imgsz khi suy luận lên 1152
  sz1280     : tăng imgsz khi suy luận lên 1280
  tta960     : Test-Time Augmentation của Ultralytics (augment=True)
  clahe960   : tăng tương phản CLAHE (kênh L) trước khi suy luận — cho vùng bóng râm/tối
  slice960   : cắt ảnh thành ô 640 chồng lấn 25% (kiểu SAHI, tự cài đặt) + 1 lượt cả ảnh;
               chỉ giữ dự đoán từ ô cho vật nhỏ (< 96x96 px) và không chạm mép trong của ô
  wbf_v5p_v6 : gộp dự đoán V5+ và V6 bằng Weighted Boxes Fusion (tự cài đặt)
Ngưỡng conf riêng từng lớp: tính hậu kỳ trên dự đoán của base960 (xem --per-class).

Chỉ số:
  mAP50, mAP50-95, AP50 từng lớp: COCO 101 điểm, ghép tham lam theo conf, theo lớp.
  P/R/F1 ở ngưỡng vận hành: ngưỡng chọn trên nửa "tune", báo cáo trên nửa "eval"
  (chia ngẫu nhiên seed 0) để không chọn và đo trên cùng ảnh.
  Recall xe máy theo kích thước (small < 32², medium < 96², large) ở ngưỡng đó.

Dùng:
  python scripts/eval/no_retrain_experiments.py --methods base960 sz1280 --device cpu
  python scripts/eval/no_retrain_experiments.py --report-only
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.evaluation.det_analysis import IMG_EXTS, iou_matrix, load_gt, size_bucket  # noqa: E402

NAMES = {0: "motorcycle", 1: "car", 2: "bus", 3: "truck"}
W_V5P = REPO / "runs/detect/train_v8s_v5plus_960/weights/best.pt"
W_V6 = REPO / "runs/detect/train_v8s_v6_pseudo/weights/best.pt"
IOU_THRS = np.linspace(0.5, 0.95, 10)
CONF_FLOOR, NMS_IOU, MAX_DET = 0.001, 0.6, 300
ALL_METHODS = ["base960", "sz1152", "sz1280", "tta960", "clahe960", "slice960", "wbf_v5p_v6"]


# ---------------------------------------------------------------- suy luận
def _boxes(res):
    b = res.boxes
    return (b.xyxy.cpu().numpy().astype(np.float32), b.conf.cpu().numpy().astype(np.float32),
            b.cls.cpu().numpy().astype(int))


def nms_classwise(xyxy, conf, cls, iou_thr=NMS_IOU, max_det=MAX_DET):
    import torch
    from torchvision.ops import batched_nms
    if len(conf) == 0:
        return xyxy, conf, cls
    keep = batched_nms(torch.from_numpy(xyxy).float(), torch.from_numpy(conf).float(),
                       torch.from_numpy(cls).long(), iou_thr).numpy()[:max_det]
    return xyxy[keep], conf[keep], cls[keep]


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def tiles(w, h, size=640, overlap=0.25):
    if w <= size and h <= size:
        return []
    step = int(size * (1 - overlap))
    xs = list(range(0, max(w - size, 0) + 1, step))
    ys = list(range(0, max(h - size, 0) + 1, step))
    if xs[-1] != max(w - size, 0):
        xs.append(max(w - size, 0))
    if ys[-1] != max(h - size, 0):
        ys.append(max(h - size, 0))
    return [(x, y, min(x + size, w), min(y + size, h)) for y in ys for x in xs]


def predict_slice(model, img, device, small_max=96, margin=2):
    full = _boxes(model.predict(img, imgsz=960, conf=CONF_FLOOR, iou=NMS_IOU, max_det=MAX_DET,
                                device=device, verbose=False)[0])
    h, w = img.shape[:2]
    tl = tiles(w, h)
    parts = [full]
    if tl:
        crops = [img[y1:y2, x1:x2] for x1, y1, x2, y2 in tl]
        for (x1, y1, x2, y2), res in zip(tl, model.predict(crops, imgsz=960, conf=CONF_FLOOR,
                                                          iou=NMS_IOU, max_det=MAX_DET,
                                                          device=device, verbose=False)):
            b, c, k = _boxes(res)
            if not len(c):
                continue
            # bỏ box chạm mép TRONG của ô (bị cắt), giữ mép trùng mép ảnh
            inner = np.zeros(len(c), bool)
            if x1 > 0:
                inner |= b[:, 0] <= margin
            if y1 > 0:
                inner |= b[:, 1] <= margin
            if x2 < w:
                inner |= b[:, 2] >= (x2 - x1) - margin
            if y2 < h:
                inner |= b[:, 3] >= (y2 - y1) - margin
            b = b + np.array([x1, y1, x1, y1], np.float32)
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            keep = ~inner & (area < small_max ** 2)
            parts.append((b[keep], c[keep], k[keep]))
    xyxy = np.concatenate([p[0] for p in parts]).reshape(-1, 4)
    conf = np.concatenate([p[1] for p in parts])
    cls = np.concatenate([p[2] for p in parts])
    return nms_classwise(xyxy, conf, cls)


def wbf(lists, n_models, iou_thr=0.55):
    """Weighted Boxes Fusion (Solovyev et al., 2021), conf_type='avg'."""
    xyxy = np.concatenate([l[0] for l in lists]).reshape(-1, 4)
    conf = np.concatenate([l[1] for l in lists])
    cls = np.concatenate([l[2] for l in lists])
    out_b, out_s, out_c = [], [], []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        idx = idx[np.argsort(-conf[idx])]
        fused, members = [], []
        for i in idx:
            if fused:
                ious = iou_matrix(xyxy[i:i + 1], np.array(fused))[0]
                j = int(np.argmax(ious))
                if ious[j] > iou_thr:
                    members[j].append(i)
                    m = np.array(members[j])
                    wts = conf[m][:, None]
                    fused[j] = (xyxy[m] * wts).sum(0) / wts.sum()
                    continue
            fused.append(xyxy[i].copy())
            members.append([i])
        for fb, m in zip(fused, members):
            t = len(m)
            out_b.append(fb)
            out_s.append(conf[m].mean() * min(t, n_models) / n_models)
            out_c.append(c)
    if not out_b:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    out_b, out_s, out_c = np.array(out_b, np.float32), np.array(out_s, np.float32), np.array(out_c)
    o = np.argsort(-out_s)[:MAX_DET]
    return out_b[o], out_s[o], out_c[o]


def run_method(method, paths, device, cache_dir, extra=None):
    from ultralytics import YOLO
    out = cache_dir / f"{method}.pkl"
    if out.exists():
        return pickle.load(open(out, "rb"))
    extra = extra or {}
    m5 = YOLO(str(extra.get(method, W_V5P)))
    m6 = YOLO(str(W_V6)) if method == "wbf_v5p_v6" else None
    preds, t0 = {}, time.time()
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        kw = dict(conf=CONF_FLOOR, iou=NMS_IOU, max_det=MAX_DET, device=device, verbose=False)
        if method == "base960" or method in extra:
            r = _boxes(m5.predict(img, imgsz=960, **kw)[0])
        elif method == "sz1152":
            r = _boxes(m5.predict(img, imgsz=1152, **kw)[0])
        elif method == "sz1280":
            r = _boxes(m5.predict(img, imgsz=1280, **kw)[0])
        elif method == "tta960":
            r = _boxes(m5.predict(img, imgsz=960, augment=True, **kw)[0])
        elif method == "clahe960":
            r = _boxes(m5.predict(clahe(img), imgsz=960, **kw)[0])
        elif method == "slice960":
            r = predict_slice(m5, img, device)
        elif method == "wbf_v5p_v6":
            a = _boxes(m5.predict(img, imgsz=960, **kw)[0])
            b = _boxes(m6.predict(img, imgsz=960, **kw)[0])
            r = wbf([a, b], n_models=2)
        else:
            raise ValueError(method)
        preds[p.name] = r
        if (i + 1) % 200 == 0:
            print(f"  [{method}] {i + 1}/{len(paths)}  {(time.time() - t0) / (i + 1):.3f} s/ảnh", flush=True)
    sec = (time.time() - t0) / len(paths)
    data = {"preds": preds, "sec_per_img": sec, "device": device}
    pickle.dump(data, open(out, "wb"))
    print(f"[{method}] xong {len(paths)} ảnh, {sec:.3f} s/ảnh", flush=True)
    return data


# ---------------------------------------------------------------- chỉ số
def match_image(gt_cls, gt_box, p_box, p_conf, p_cls):
    """Ghép theo lớp, tham lam theo conf giảm dần, ở 10 ngưỡng IoU.
    Trả về tp [N,10] cho từng dự đoán và gt_conf [G] = conf của dự đoán đã ghép GT
    ở IoU 0.5 (0 nếu không ghép). Với ghép tham lam, lọc theo ngưỡng conf sau đó
    không làm thay đổi các cặp đã ghép của dự đoán còn lại."""
    tp = np.zeros((len(p_conf), len(IOU_THRS)), bool)
    gt_conf = np.zeros(len(gt_cls), np.float32)
    for c in np.unique(np.concatenate([gt_cls, p_cls])) if len(gt_cls) + len(p_cls) else []:
        gi = np.where(gt_cls == c)[0]
        pi = np.where(p_cls == c)[0]
        if not len(gi) or not len(pi):
            continue
        pi = pi[np.argsort(-p_conf[pi])]
        ious = iou_matrix(p_box[pi], gt_box[gi])
        for t, thr in enumerate(IOU_THRS):
            taken = np.zeros(len(gi), bool)
            for r, k in enumerate(pi):
                cand = np.where(~taken & (ious[r] >= thr))[0]
                if len(cand):
                    j = cand[np.argmax(ious[r, cand])]
                    taken[j] = True
                    tp[k, t] = True
                    if t == 0:
                        gt_conf[gi[j]] = p_conf[k]
    return tp, gt_conf


def ap101(tp, conf, n_gt):
    if n_gt == 0:
        return np.nan
    if len(conf) == 0:
        return 0.0
    o = np.argsort(-conf)
    tpc = np.cumsum(tp[o])
    fpc = np.cumsum(~tp[o])
    rec = tpc / n_gt
    prec = tpc / np.maximum(tpc + fpc, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    pts = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, pts, side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx]))


def build_table(preds, gts, names_subset):
    rows_p, rows_g = [], []
    for n in names_subset:
        gc, gb = gts[n]
        pb, pc, pk = preds[n]
        tp, gconf = match_image(gc, gb, pb, pc, pk)
        for k in range(len(pc)):
            rows_p.append((n, int(pk[k]), float(pc[k]), *tp[k]))
        for j in range(len(gc)):
            w, h = gb[j, 2] - gb[j, 0], gb[j, 3] - gb[j, 1]
            rows_g.append((n, int(gc[j]), size_bucket(w * h), float(gconf[j])))
    P = pd.DataFrame(rows_p, columns=["img", "cls", "conf"] + [f"tp{t}" for t in range(10)])
    G = pd.DataFrame(rows_g, columns=["img", "cls", "size", "mconf"])
    return P, G


def ap_metrics(P, G):
    out = {}
    aps50, aps = [], []
    for c, name in NAMES.items():
        p = P[P.cls == c]
        n = int((G.cls == c).sum())
        a = [ap101(p[f"tp{t}"].values, p.conf.values, n) for t in range(10)]
        out[f"AP50_{name}"] = a[0]
        aps50.append(a[0])
        aps.append(np.nanmean(a))
    out["mAP50"] = float(np.nanmean(aps50))
    out["mAP50-95"] = float(np.nanmean(aps))
    return out


def _area(b):
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


def ap_area_range(preds, gts, names_subset, cls, lo, hi):
    """AP50 của một lớp chỉ tính GT có diện tích trong [lo, hi) — kiểu COCO area range.
    GT ngoài khoảng bị bỏ qua (dự đoán ghép vào chúng không tính là sai); dự đoán
    không ghép được chỉ tính là FP nếu chính nó nằm trong khoảng. Không phụ thuộc ngưỡng conf."""
    tps, confs, n_valid = [], [], 0
    for n in names_subset:
        gc, gb = gts[n]
        pb, pc, pk = preds[n]
        gi = np.where(gc == cls)[0]
        gign = ~((_area(gb[gi]) >= lo) & (_area(gb[gi]) < hi)) if len(gi) else np.zeros(0, bool)
        n_valid += int((~gign).sum())
        pi = np.where(pk == cls)[0]
        if not len(pi):
            continue
        pi = pi[np.argsort(-pc[pi])]
        ious = iou_matrix(pb[pi], gb[gi]) if len(gi) else np.zeros((len(pi), 0))
        taken = np.zeros(len(gi), bool)
        parea = _area(pb[pi])
        for r, k in enumerate(pi):
            cand = np.where(~taken & (ious[r] >= 0.5))[0]
            if len(cand):
                ok = cand[~gign[cand]]
                j = ok[np.argmax(ious[r, ok])] if len(ok) else cand[np.argmax(ious[r, cand])]
                taken[j] = True
                if not gign[j]:
                    tps.append(True)
                    confs.append(pc[k])
            elif lo <= parea[r] < hi:
                tps.append(False)
                confs.append(pc[k])
    return ap101(np.array(tps, bool), np.array(confs, np.float32), n_valid)


def _ap101_w(tp, w, n_gt):
    """AP101 với trọng số ảnh (bootstrap); tp, w đã sắp theo conf giảm dần."""
    if n_gt <= 0:
        return np.nan
    tpc = np.cumsum(tp * w)
    fpc = np.cumsum((~tp) * w)
    rec = tpc / n_gt
    prec = tpc / np.maximum(tpc + fpc, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
    return float(np.mean(np.where(idx < len(prec), prec[np.minimum(idx, len(prec) - 1)], 0.0)))


def boot_arrays(P, G, img_index):
    """Chuẩn bị mảng theo lớp để tính mAP50 có trọng số ảnh nhanh."""
    out = {}
    for c in NAMES:
        p = P[P.cls == c].sort_values("conf", ascending=False)
        out[c] = (p.img.map(img_index).values, p.tp0.values.astype(bool),
                  G.img[G.cls == c].map(img_index).values)
    return out


def boot_map50(arr, w):
    aps = [_ap101_w(tp, w[pi], w[gi].sum()) for pi, tp, gi in arr.values()]
    return float(np.nanmean(aps))


def op_point(P, G, thr):
    """thr: float (chung) hoặc dict lớp -> ngưỡng."""
    t = (lambda c: thr[c]) if isinstance(thr, dict) else (lambda c: thr)
    keep = P.conf.values >= np.array([t(c) for c in P.cls.values])
    tp = int(P.tp0.values[keep].sum())
    npred = int(keep.sum())
    rec_mask = G.mconf.values >= np.array([t(c) for c in G.cls.values])
    rec_mask &= G.mconf.values > 0
    prec = tp / npred if npred else 0.0
    rec = rec_mask.sum() / len(G) if len(G) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    res = {"P": prec, "R": rec, "F1": f1}
    for s in ["small", "medium", "large"]:
        g = (G.cls.values == 0) & (G["size"].values == s)
        res[f"R_moto_{s}"] = rec_mask[g].mean() if g.any() else np.nan
        res[f"n_moto_{s}"] = int(g.sum())
    return res


def best_global_thr(P, G):
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    f1 = [op_point(P, G, t)["F1"] for t in grid]
    return float(grid[int(np.argmax(f1))])


def best_per_class_thr(P, G):
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    thr = {c: best_global_thr(P, G) for c in NAMES}
    for _ in range(2):                       # tối ưu lần lượt từng lớp (coordinate ascent)
        for c in NAMES:
            best = max(grid, key=lambda v: op_point(P, G, {**thr, c: float(v)})["F1"])
            thr[c] = float(best)
    return thr


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=None,
                    help="mặc định: 7 cách của V5+; nếu có --model-weights thì: base960 và các model đó")
    ap.add_argument("--model-weights", nargs="*", default=[], metavar="TÊN=ĐƯỜNG_DẪN",
                    help="đánh giá thêm mô hình khác bằng suy luận chuẩn imgsz 960, vd v6b=runs/detect/x/weights/best.pt")
    ap.add_argument("--data", type=Path, default=REPO / "data/merged_v6")
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--out", type=Path, default=REPO / "results/no_retrain")
    ap.add_argument("--report-only", action="store_true", help="chỉ tính chỉ số từ cache")
    ap.add_argument("--bootstrap", type=int, default=300,
                    help="số lần bootstrap ảnh cho khoảng tin cậy ΔmAP50 so với base960 (0 = tắt)")
    args = ap.parse_args()

    img_dir, lbl_dir = args.data / "images" / args.split, args.data / "labels" / args.split
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if args.max_images and args.max_images < len(paths):
        import random
        paths = sorted(random.Random(0).sample(paths, args.max_images))
    cache = args.out / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    gts, sizes = {}, {}
    for p in paths:
        h, w = cv2.imread(str(p)).shape[:2]
        gts[p.name] = load_gt(lbl_dir / (p.stem + ".txt"), w, h)
        sizes[p.name] = (w, h)

    names = [p.name for p in paths]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(names))
    tune = [names[i] for i in perm[: len(names) // 2]]
    evl = [names[i] for i in perm[len(names) // 2:]]

    img_index = {n: i for i, n in enumerate(names)}
    W = None
    if args.bootstrap:
        rng_b = np.random.default_rng(0)
        W = np.stack([np.bincount(rng_b.integers(0, len(names), len(names)), minlength=len(names))
                      for _ in range(args.bootstrap)]).astype(np.float64)
    base_boot = None
    extra = {}
    for kv in args.model_weights:
        k, _, v = kv.partition("=")
        if not k or not v or not Path(v).exists():
            ap.error(f"--model-weights cần dạng TÊN=ĐƯỜNG_DẪN tồn tại, nhận: {kv}")
        if k in ALL_METHODS:
            ap.error(f"tên {k} trùng tên một cách có sẵn, đổi tên khác")
        extra[k] = Path(v)
    if args.methods is None:
        args.methods = (["base960"] + list(extra)) if extra else list(ALL_METHODS)
    bad = [m for m in args.methods if m not in ALL_METHODS and m not in extra]
    if bad:
        ap.error(f"cách không hợp lệ: {bad}")
    order = ["base960"] + [m for m in args.methods if m != "base960"] \
        if "base960" in args.methods else list(args.methods)

    rows = []
    for m in order:
        if args.report_only and not (cache / f"{m}.pkl").exists():
            continue
        data = run_method(m, paths, args.device, cache, extra)
        preds = {k: v for k, v in data["preds"].items() if k in gts}
        P_all, G_all = build_table(preds, gts, names)
        P_t, G_t = P_all[P_all.img.isin(tune)], G_all[G_all.img.isin(tune)]
        P_e, G_e = P_all[P_all.img.isin(evl)], G_all[G_all.img.isin(evl)]
        thr = best_global_thr(P_t, G_t)
        row = {"method": m, "s/img": round(data["sec_per_img"], 3), **ap_metrics(P_all, G_all),
               "AP50_moto_small": ap_area_range(preds, gts, names, 0, 0, 32 ** 2),
               "AP50_moto_medium": ap_area_range(preds, gts, names, 0, 32 ** 2, 96 ** 2),
               "conf": thr, **op_point(P_e, G_e, thr)}
        if W is not None:
            arr = boot_arrays(P_all, G_all, img_index)
            vals = np.array([boot_map50(arr, w) for w in W])
            if m == "base960":
                base_boot = vals
            elif base_boot is not None:
                d = vals - base_boot
                row["ΔmAP50_CI95"] = f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
                row["P(Δ>0)"] = round(float((d > 0).mean()), 3)
        rows.append(row)
        if m == "base960":
            pct = best_per_class_thr(P_t, G_t)
            rows.append({"method": "base960+per_class_conf", "s/img": row["s/img"],
                         **ap_metrics(P_all, G_all),
                         "AP50_moto_small": row["AP50_moto_small"],
                         "AP50_moto_medium": row["AP50_moto_medium"],
                         "conf": "/".join(f"{NAMES[c][:4]}={v:.2f}" for c, v in pct.items()),
                         **op_point(P_e, G_e, pct)})
        print(f"[{m}] mAP50={row['mAP50']:.4f} F1={row['F1']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(4)
    df.to_csv(args.out / "summary.csv", index=False)
    if "base960" in set(df.method):
        b = df[df.method == "base960"].iloc[0]
        for col in ["mAP50", "mAP50-95", "F1", "AP50_moto_small"]:
            df[f"Δ{col}"] = (df[col] - b[col]).round(4)
    cols = ["method", "s/img", "mAP50", "mAP50-95", "AP50_motorcycle", "AP50_car", "AP50_bus",
            "AP50_truck", "AP50_moto_small", "AP50_moto_medium", "conf", "P", "R", "F1",
            "R_moto_small"]
    cols += [c for c in df.columns if c.startswith("Δ")] + ["P(Δ>0)"]
    t = df[[c for c in cols if c in df]].map(lambda v: "" if pd.isna(v) else str(v))
    md = ["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
    md += ["| " + " | ".join(r) + " |" for r in t.values.tolist()]
    (args.out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    print(df[[c for c in cols if c in df]].to_string(index=False))
    n_small = int(df["n_moto_small"].iloc[0]) if len(df) else 0
    print(f"\nTập: {len(names)} ảnh ({args.split}); nửa tune {len(tune)} / eval {len(evl)}; "
          f"xe máy small trong nửa eval: {n_small}")


if __name__ == "__main__":
    main()
