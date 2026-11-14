"""(h) Kiểm tra ĐỘ ĐẦY ĐỦ nhãn của v10 (thiếu nhãn) + overlay vẽ sẵn lên ảnh.

Không phải check leak, mà là check chất lượng dữ liệu train. Hai phần:

A. Thiếu nhãn (tự động): chạy detector COCO đã huấn luyện sẵn (YOLOv8m, CHỈ inference,
   không train) trên toàn bộ ảnh v10. Lấy các detection lớp xe (car, motorcycle, bus, truck),
   conf ≥ 0.5, và coi là "xe chưa gán nhãn" nếu:
     - IoU với mọi box GT < 0.3, và
     - không nằm phần lớn trong một box GT (IoA < 0.6, để bỏ trường hợp detector cắt
       một phần xe tải/container thành nhiều box), và
     - diện tích ≥ 0.2% ảnh (bỏ xe rất xa, nhiều bộ dữ liệu cố ý không gán).
   ĐỐI CHỨNG: chạy y hệt trên val/test của V5 (gán nhãn đầy đủ theo video). Detector
   có dương tính giả, nên chỉ phần CHÊNH so với V5 mới là bằng chứng thiếu nhãn.

B. Overlay vẽ sẵn (bán thủ công): heuristic màu tự động không tách được (đã thử: bắt
   <3% ảnh v10, trong khi dương tính giả trên V5 là 5-8%) → dùng review bằng mắt một mẫu
   ngẫu nhiên cố định seed. `--make-sheets` sinh contact sheet; kết quả review ghi tay vào
   reports/v10_overlay_manual_review.json; script tính tỉ lệ + khoảng tin cậy Wilson 95%.

   Lưu ý: detector COCO bỏ sót nhiều xe máy nhìn từ trên cao → con số là CẬN DƯỚI.

Verdict:
- A: FAIL nếu ước lượng % XE chưa gán nhãn của v10 vượt V5 quá 5 điểm %.
  (Không dùng % ảnh: V5 ~19 box/ảnh, v10 ~1.3 → % ảnh thiên lệch theo mật độ xe.)
- B: FAIL nếu cận dưới CI 95% của tỉ lệ ảnh có overlay > 5%.

Usage:
    python scripts/data_prep/audit_v10/check_label_completeness_v10.py              # chạy A + tổng hợp B
    python scripts/data_prep/audit_v10/check_label_completeness_v10.py --make-sheets  # sinh sheet review B
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_common import IMG_EXTS, REPORTS_DIR, REPO_ROOT, V10_ROOT, write_json  # noqa: E402

OUT = REPORTS_DIR / "v10_label_completeness.json"
REVIEW = REPORTS_DIR / "v10_overlay_manual_review.json"
PREVIEW = REPORTS_DIR / "v10_missing_labels_preview.jpg"
V5_ROOT = REPO_ROOT / "data" / "merged_v5"
WEIGHTS = REPO_ROOT / "yolov8m.pt"
COCO_VEHICLE = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CONF, IOU_THR, IOA_THR, MIN_AREA = 0.5, 0.3, 0.6, 0.002
EXCESS_PP = 5.0
REVIEW_SEED, REVIEW_N = 2024, 60


# ───────────────────────────── A. thiếu nhãn ─────────────────────────────
def _v10_images() -> list[Path]:
    return sorted(p for s in ("train", "valid", "test") for p in (V10_ROOT / s / "images").iterdir()
                  if p.suffix.lower() in IMG_EXTS)


def _v5_eval_images() -> list[Path]:
    return sorted(p for s in ("val", "test") for p in (V5_ROOT / "images" / s).iterdir()
                  if p.suffix.lower() in IMG_EXTS)


def _label_of(img: Path) -> Path:
    # v10: <split>/images/x.jpg → <split>/labels/x.txt ; V5: images/<split>/x.jpg → labels/<split>/x.txt
    parts = list(img.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def _gt_xyxy(lbl: Path) -> np.ndarray:
    rows = []
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                _, x, y, w, h = map(float, p)
                rows.append((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
    return np.array(rows, dtype=np.float32).reshape(-1, 4)


def _iou_ioa(d: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """d (N,4), g (M,4) chuẩn hoá → IoU (N,M), IoA (N,M) = giao / diện tích detection."""
    if not len(d) or not len(g):
        return np.zeros((len(d), len(g))), np.zeros((len(d), len(g)))
    x1 = np.maximum(d[:, None, 0], g[None, :, 0]); y1 = np.maximum(d[:, None, 1], g[None, :, 1])
    x2 = np.minimum(d[:, None, 2], g[None, :, 2]); y2 = np.minimum(d[:, None, 3], g[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ad = ((d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1]))[:, None]
    ag = ((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]))[None, :]
    return inter / (ad + ag - inter + 1e-9), inter / (ad + 1e-9)


def _scan(model, images: list[Path], batch: int = 32) -> list[dict]:
    out = []
    for i in range(0, len(images), batch):
        chunk = images[i:i + batch]
        res = model.predict([str(p) for p in chunk], conf=CONF, classes=list(COCO_VEHICLE),
                            imgsz=640, verbose=False)
        for p, r in zip(chunk, res):
            det = r.boxes.xyxyn.cpu().numpy() if len(r.boxes) else np.zeros((0, 4), np.float32)
            cls = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) else np.zeros(0, int)
            gt = _gt_xyxy(_label_of(p))
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]) if len(det) else np.zeros(0)
            iou, ioa = _iou_ioa(det, gt)
            matched = (iou.max(1) >= IOU_THR) | (ioa.max(1) >= IOA_THR) if len(gt) else np.zeros(len(det), bool)
            miss = (~matched) & (area >= MIN_AREA)
            out.append({"path": str(p), "n_gt": len(gt), "n_det": len(det),
                        "missing": det[miss].round(4).tolist(),
                        "missing_cls": [COCO_VEHICLE[c] for c in cls[miss]]})
    return out


def _summ(rows: list[dict]) -> dict:
    n = len(rows)
    n_gt = sum(r["n_gt"] for r in rows)
    n_miss = sum(len(r["missing"]) for r in rows)
    k = sum(1 for r in rows if r["missing"])
    from collections import Counter
    return {
        "n_images": n, "n_gt_boxes": n_gt, "gt_per_image": round(n_gt / n, 2) if n else 0,
        "n_unlabeled_vehicle_dets": n_miss,
        "pct_images_with_unlabeled_vehicle": round(100 * k / n, 2) if n else 0,
        "unlabeled_per_image": round(n_miss / n, 3) if n else 0,
        "est_pct_vehicles_unlabeled": round(100 * n_miss / (n_gt + n_miss), 2) if (n_gt + n_miss) else 0,
        "unlabeled_by_coco_class": dict(Counter(c for r in rows for c in r["missing_cls"])),
    }


def _preview(rows: list[dict], out: Path, n: int = 12) -> None:
    from PIL import Image, ImageDraw
    rng = random.Random(7)
    cand = [r for r in rows if r["missing"]]
    pick = rng.sample(cand, min(n, len(cand)))
    S = 320
    sheet = Image.new("RGB", (4 * S, math.ceil(len(pick) / 4) * S), "white")
    for k, r in enumerate(pick):
        im = Image.open(r["path"]).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        for x1, y1, x2, y2 in _gt_xyxy(_label_of(Path(r["path"]))):
            d.rectangle([x1 * W, y1 * H, x2 * W, y2 * H], outline=(0, 255, 0), width=4)
        for x1, y1, x2, y2 in r["missing"]:
            d.rectangle([x1 * W, y1 * H, x2 * W, y2 * H], outline=(255, 0, 0), width=4)
        sheet.paste(im.resize((S, S)), ((k % 4) * S, (k // 4) * S))
    sheet.save(out, quality=88)
    print(f"[OK] Ghi preview {out}")


# ───────────────────────────── B. overlay ─────────────────────────────
def _review_sample() -> list[Path]:
    rng = random.Random(REVIEW_SEED)
    return rng.sample([str(p.relative_to(REPO_ROOT)) for p in _v10_images()], REVIEW_N)


def make_sheets() -> None:
    from PIL import Image, ImageDraw
    pick = _review_sample()
    S = 400
    for sh in range(math.ceil(REVIEW_N / 20)):
        sheet = Image.new("RGB", (4 * S, 5 * S), "white")
        for k in range(20):
            i = sh * 20 + k
            if i >= len(pick):
                break
            im = Image.open(REPO_ROOT / pick[i]).convert("RGB").resize((S, S))
            d = ImageDraw.Draw(im)
            d.rectangle([0, S - 34, 52, S], fill="yellow")
            d.text((6, S - 28), str(i), fill="black", font_size=24)
            sheet.paste(im, ((k % 4) * S, (k // 4) * S))
        p = REPORTS_DIR / f"v10_overlay_review_sheet_{sh}.jpg"
        sheet.save(p, quality=88)
        print(f"[OK] Ghi {p.relative_to(REPO_ROOT)} — ghi kết quả vào {REVIEW.relative_to(REPO_ROOT)}")


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def _overlay_summary() -> dict:
    if not REVIEW.exists():
        return {"verdict": "N/A", "note": f"Chưa có {REVIEW.name}; chạy --make-sheets rồi review"}
    rv = json.loads(REVIEW.read_text())
    sample = _review_sample()
    vals = rv["has_overlay"]
    k = sum(1 for v in vals.values() if v is True)
    n = len(vals)
    lo, hi = _wilson(k, n)
    return {
        "verdict": "FAIL" if lo > 0.05 else "PASS",
        "threshold": "cận dưới CI95 tỉ lệ ảnh có overlay ≤ 5%",
        "n_reviewed": n, "n_with_overlay": k,
        "n_unclear": sum(1 for v in vals.values() if v == "unclear"),
        "pct_with_overlay": round(100 * k / n, 1),
        "ci95_pct": [round(100 * lo, 1), round(100 * hi, 1)],
        "est_images_in_v10": [round(lo * len(_v10_images())), round(hi * len(_v10_images()))],
        "sample_files": {str(i): s for i, s in enumerate(sample)},
        "observations": rv.get("observations", []),
    }


def run(out: Path = OUT, preview: Path | None = PREVIEW, v5_limit: int | None = None) -> dict:
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    v10_rows = _scan(model, _v10_images())
    v5_imgs = _v5_eval_images()
    if v5_limit:
        v5_imgs = random.Random(0).sample(v5_imgs, min(v5_limit, len(v5_imgs)))
    v5_rows = _scan(model, v5_imgs)
    s10, s5 = _summ(v10_rows), _summ(v5_rows)
    # So theo % XE chưa gán nhãn (không theo % ảnh: V5 ~19 box/ảnh vs v10 ~1.3 → % ảnh thiên lệch theo mật độ)
    excess = round(s10["est_pct_vehicles_unlabeled"] - s5["est_pct_vehicles_unlabeled"], 2)
    if preview:
        _preview(v10_rows, preview)
    ov = _overlay_summary()
    a_verdict = "FAIL" if excess > EXCESS_PP else "PASS"
    result = {
        "check": "label_completeness",
        "threshold": (f"A: % xe chưa gán nhãn (v10) − (V5) ≤ {EXCESS_PP}pp; "
                      f"B: cận dưới CI95 tỉ lệ overlay ≤ 5%"),
        "verdict": "FAIL" if "FAIL" in (a_verdict, ov.get("verdict")) else "PASS",
        "missing_labels": {
            "verdict": a_verdict,
            "detector": f"{WEIGHTS.name} (COCO, chỉ inference)",
            "params": {"conf": CONF, "iou_match": IOU_THR, "ioa_inside_gt": IOA_THR, "min_area_frac": MIN_AREA,
                       "coco_classes": list(COCO_VEHICLE.values())},
            "v10": s10, "v5_eval_control": s5, "excess_pp_vs_v5": excess,
            "top_images": sorted(({"path": str(Path(r["path"]).relative_to(REPO_ROOT)), "n_gt": r["n_gt"],
                                   "n_unlabeled": len(r["missing"])} for r in v10_rows if r["missing"]),
                                 key=lambda e: -e["n_unlabeled"])[:50],
        },
        "burned_in_overlay": ov,
        "preview": str(preview.relative_to(REPO_ROOT)) if preview else None,
    }
    write_json(result, out)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--make-sheets", action="store_true", help="Sinh sheet review overlay rồi thoát")
    ap.add_argument("--v5-limit", type=int, default=None, help="Giới hạn số ảnh V5 đối chứng (mặc định: tất cả)")
    a = ap.parse_args()
    if a.make_sheets:
        make_sheets()
        return 0
    r = run(a.out, PREVIEW, a.v5_limit)
    m, o = r["missing_labels"], r["burned_in_overlay"]
    print(f"[{m['verdict']}] Thiếu nhãn (% xe): v10 {m['v10']['est_pct_vehicles_unlabeled']}% vs "
          f"V5 {m['v5_eval_control']['est_pct_vehicles_unlabeled']}% → chênh {m['excess_pp_vs_v5']}pp "
          f"(% ảnh: v10 {m['v10']['pct_images_with_unlabeled_vehicle']}%, V5 {m['v5_eval_control']['pct_images_with_unlabeled_vehicle']}%)")
    print(f"[{o.get('verdict')}] Overlay: {o.get('n_with_overlay')}/{o.get('n_reviewed')} = {o.get('pct_with_overlay')}% "
          f"(CI95 {o.get('ci95_pct')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
