"""Phân tích lỗi detection theo kích thước đối tượng, nhầm lớp và ngưỡng conf.

Dùng cho notebook `notebooks/eval_v5plus_inference.ipynb`. Không phụ thuộc
ultralytics trừ hàm `collect_predictions` (nhận vào một đối tượng YOLO).

Quy ước:
- GT đọc từ file nhãn YOLO (class xc yc w h, chuẩn hoá) -> đổi sang xyxy pixel.
- Ghép cặp pred-GT tham lam theo conf giảm dần, IoU >= iou_thr, KHÔNG xét lớp.
  Ghép đúng lớp = TP, ghép sai lớp = confused, GT không ghép được = FN,
  pred không ghép được = FP.
- Kích thước theo chuẩn COCO tính trên diện tích pixel của ảnh gốc:
  small < 32^2, medium < 96^2, large còn lại.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SIZE_ORDER = ["small", "medium", "large"]


def load_gt(label_path: Path, img_w: int, img_h: int):
    """Đọc nhãn YOLO -> (cls [N], xyxy [N,4] pixel)."""
    cls, boxes = [], []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c = int(float(parts[0]))
            xc, yc, w, h = (float(v) for v in parts[1:5])
            boxes.append([(xc - w / 2) * img_w, (yc - h / 2) * img_h,
                          (xc + w / 2) * img_w, (yc + h / 2) * img_h])
            cls.append(c)
    return np.array(cls, dtype=int), np.array(boxes, dtype=float).reshape(-1, 4)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU từng cặp giữa a [N,4] và b [M,4] (xyxy)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def size_bucket(area_px: float) -> str:
    if area_px < 32 ** 2:
        return "small"
    if area_px < 96 ** 2:
        return "medium"
    return "large"


def match_image(gt_cls, gt_xyxy, pred_cls, pred_xyxy, pred_conf,
                conf_thr: float, iou_thr: float = 0.5):
    """Ghép dự đoán với GT cho 1 ảnh.

    Trả về (gt_rows, fp_rows):
      gt_rows: list dict {gt_cls, size, status in {tp, confused, fn}, pred_cls}
      fp_rows: list dict {pred_cls, conf}
    """
    keep = pred_conf >= conf_thr
    p_cls, p_box, p_conf = pred_cls[keep], pred_xyxy[keep], pred_conf[keep]
    order = np.argsort(-p_conf)
    p_cls, p_box, p_conf = p_cls[order], p_box[order], p_conf[order]

    ious = iou_matrix(p_box, gt_xyxy)
    gt_taken = np.zeros(len(gt_cls), dtype=bool)
    gt_pred = np.full(len(gt_cls), -1, dtype=int)
    fp_rows = []

    for i in range(len(p_cls)):
        cand = np.where(~gt_taken & (ious[i] >= iou_thr))[0] if len(gt_cls) else []
        if len(cand) == 0:
            fp_rows.append({"pred_cls": int(p_cls[i]), "conf": float(p_conf[i])})
            continue
        j = cand[np.argmax(ious[i, cand])]
        gt_taken[j] = True
        gt_pred[j] = p_cls[i]

    gt_rows = []
    for j in range(len(gt_cls)):
        w = gt_xyxy[j, 2] - gt_xyxy[j, 0]
        h = gt_xyxy[j, 3] - gt_xyxy[j, 1]
        if not gt_taken[j]:
            status = "fn"
        elif gt_pred[j] == gt_cls[j]:
            status = "tp"
        else:
            status = "confused"
        gt_rows.append({"gt_cls": int(gt_cls[j]), "size": size_bucket(w * h),
                        "status": status,
                        "pred_cls": int(gt_pred[j]) if gt_taken[j] else -1})
    return gt_rows, fp_rows


def collect_predictions(model, images_dir: Path, labels_dir: Path,
                        conf_floor: float = 0.05, iou_nms: float = 0.6,
                        imgsz: int = 960, device=None, batch: int = 16,
                        max_images: int | None = None):
    """Chạy model trên toàn bộ ảnh một lần, lưu dự đoán với conf thấp để quét ngưỡng.

    Trả về list record {path, w, h, gt_cls, gt_xyxy, pred_cls, pred_xyxy, pred_conf}.
    """
    paths = sorted(p for p in Path(images_dir).iterdir()
                   if p.suffix.lower() in IMG_EXTS)
    if max_images and max_images < len(paths):
        # Lấy mẫu ngẫu nhiên cố định (seed 0) để phủ đủ các camera/video.
        # Lấy N file đầu theo tên sẽ chỉ rơi vào 1-2 video (vd toàn Cần Thơ).
        import random
        paths = sorted(random.Random(0).sample(paths, max_images))
    records = []
    stream = model.predict([str(p) for p in paths], conf=conf_floor, iou=iou_nms,
                           imgsz=imgsz, device=device, batch=batch,
                           stream=True, verbose=False)
    for p, res in zip(paths, stream):
        h, w = res.orig_shape
        gt_cls, gt_xyxy = load_gt(Path(labels_dir) / (p.stem + ".txt"), w, h)
        records.append({
            "path": str(p), "w": w, "h": h,
            "gt_cls": gt_cls, "gt_xyxy": gt_xyxy,
            "pred_cls": res.boxes.cls.cpu().numpy().astype(int),
            "pred_xyxy": res.boxes.xyxy.cpu().numpy(),
            "pred_conf": res.boxes.conf.cpu().numpy(),
        })
    return records


def evaluate_records(records, conf_thr: float, iou_thr: float = 0.5):
    """Ghép toàn bộ record ở một ngưỡng conf. Trả về (df_gt, df_fp)."""
    gt_all, fp_all = [], []
    for r in records:
        g, f = match_image(r["gt_cls"], r["gt_xyxy"], r["pred_cls"],
                           r["pred_xyxy"], r["pred_conf"], conf_thr, iou_thr)
        for row in g:
            row["path"] = r["path"]
        for row in f:
            row["path"] = r["path"]
        gt_all += g
        fp_all += f
    return (pd.DataFrame(gt_all, columns=["path", "gt_cls", "size", "status", "pred_cls"]),
            pd.DataFrame(fp_all, columns=["path", "pred_cls", "conf"]))


def recall_by_size(df_gt: pd.DataFrame, names: dict) -> pd.DataFrame:
    """Recall (TP/GT) theo lớp x kích thước, kèm số GT."""
    d = df_gt.copy()
    d["class"] = d["gt_cls"].map(names)
    d["tp"] = (d["status"] == "tp").astype(int)
    out = d.groupby(["class", "size"]).agg(gt=("tp", "size"), tp=("tp", "sum"))
    out["recall"] = out["tp"] / out["gt"]
    return out.reset_index()


def confusion(df_gt: pd.DataFrame, df_fp: pd.DataFrame, n_classes: int) -> np.ndarray:
    """Ma trận (n+1)x(n+1): hàng = GT (cuối là background), cột = dự đoán
    (cuối là background/bỏ sót). [gt, n] = FN, [n, pred] = FP."""
    m = np.zeros((n_classes + 1, n_classes + 1), dtype=int)
    for g, p in zip(df_gt["gt_cls"], df_gt["pred_cls"]):
        m[g, p if p >= 0 else n_classes] += 1
    for p in df_fp["pred_cls"]:
        m[n_classes, p] += 1
    return m


def sweep_conf(records, thresholds, iou_thr: float = 0.5) -> pd.DataFrame:
    """Precision / Recall / F1 (toàn bộ lớp) theo ngưỡng conf."""
    rows = []
    for t in thresholds:
        g, f = evaluate_records(records, t, iou_thr)
        tp = int((g["status"] == "tp").sum())
        n_gt = len(g)
        n_pred = int((g["status"] != "fn").sum()) + len(f)
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_gt if n_gt else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"conf": float(t), "precision": prec, "recall": rec,
                     "f1": f1, "tp": tp, "fp": len(f) + int((g["status"] == "confused").sum()),
                     "fn": int((g["status"] == "fn").sum())})
    return pd.DataFrame(rows)


def worst_images(records, conf_thr: float, top_k: int = 6, iou_thr: float = 0.5):
    """Các ảnh có nhiều lỗi nhất (FN + nhầm lớp + FP)."""
    scored = []
    for r in records:
        g, f = match_image(r["gt_cls"], r["gt_xyxy"], r["pred_cls"],
                           r["pred_xyxy"], r["pred_conf"], conf_thr, iou_thr)
        errs = sum(1 for x in g if x["status"] != "tp") + len(f)
        scored.append((errs, r))
    scored.sort(key=lambda t: -t[0])
    return [r for e, r in scored[:top_k] if e > 0]
