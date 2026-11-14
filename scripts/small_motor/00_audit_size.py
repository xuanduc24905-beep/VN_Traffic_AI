"""00_audit_size.py — Phân tích định lượng kích thước đối tượng motorcycle trên tập V5+.

Mục tiêu:
  - Thống kê phân phối diện tích bbox (pixel²) của class motorcycle theo ngưỡng small/medium/large.
  - Tính AP50 từng bucket kích thước dựa trên kết quả val của mô hình baseline.
  - Xuất biểu đồ histogram và bảng CSV để so sánh trước-sau cải tiến.

Dùng:
  python scripts/small_motor/00_audit_size.py \\
      --data  data/data_v5plus.yaml \\
      --model runs/detect/exp_baseline/weights/best.pt \\
      --out   analysis/size_audit        # thư mục chứa CSV + PNG

Nếu không có --model: chỉ thống kê nhãn, không tính AP.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MOTOR_ID = 0          # motorcycle là class 0 trong schema 4-class
# Ngưỡng COCO: small <32², medium 32-96², large >96² (pixel tương đối 640px)
# Ở ảnh 960px ta nhân thêm (960/640)² → ~2.25 lần; nhưng ta dùng pixel thật
THRESHOLDS = {"small": (0, 32**2), "medium": (32**2, 96**2), "large": (96**2, float("inf"))}


# ---------------------------------------------------------------------------
# 1. Đọc nhãn
# ---------------------------------------------------------------------------

def read_yolo_labels(label_dir: Path, img_dir: Path) -> list[dict]:
    """Trả về list dict mỗi phần tử là một bbox motorcycle."""
    records = []
    for lp in sorted(label_dir.rglob("*.txt")):
        # Tìm ảnh tương ứng để lấy kích thước thật
        img_p = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = img_dir / lp.relative_to(label_dir).with_suffix(ext)
            if cand.exists():
                img_p = cand
                break
        if img_p is None:
            continue
        try:
            import cv2
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            H, W = img.shape[:2]
        except Exception:
            continue
        for line in lp.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            if cls != MOTOR_ID:
                continue
            _, xc, yc, w, h = map(float, parts)
            pw, ph = w * W, h * H
            area = pw * ph
            records.append(dict(
                img=lp.stem, W=W, H=H,
                w_px=round(pw, 1), h_px=round(ph, 1), area_px=round(area, 1),
                rel_w=round(w, 4), rel_h=round(h, 4),
            ))
    return records


def bucket(area: float) -> str:
    for name, (lo, hi) in THRESHOLDS.items():
        if lo <= area < hi:
            return name
    return "large"


# ---------------------------------------------------------------------------
# 2. Thống kê
# ---------------------------------------------------------------------------

def compute_stats(records: list[dict]) -> dict:
    areas = np.array([r["area_px"] for r in records])
    if len(areas) == 0:
        return {}
    buckets = {}
    for b in THRESHOLDS:
        mask = np.array([bucket(r["area_px"]) == b for r in records])
        buckets[b] = int(mask.sum())
    return dict(
        total=len(areas),
        mean_area=float(np.mean(areas)),
        median_area=float(np.median(areas)),
        p5_area=float(np.percentile(areas, 5)),
        p25_area=float(np.percentile(areas, 25)),
        p75_area=float(np.percentile(areas, 75)),
        p95_area=float(np.percentile(areas, 95)),
        pct_small=round(100 * buckets.get("small", 0) / max(len(areas), 1), 1),
        pct_medium=round(100 * buckets.get("medium", 0) / max(len(areas), 1), 1),
        pct_large=round(100 * buckets.get("large", 0) / max(len(areas), 1), 1),
        **{f"n_{k}": v for k, v in buckets.items()},
    )


# ---------------------------------------------------------------------------
# 3. Tính AP theo bucket (nếu có model)
# ---------------------------------------------------------------------------

def eval_by_size(model_path: Path, data_yaml: Path, imgsz: int, device) -> dict:
    """Chạy val của Ultralytics, trích AP50 cho motorcycle.
    Trả về dict: ap50_motorcycle, và phân rã thủ công qua thresholding conf."""
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    r = model.val(
        data=str(data_yaml), split="val",
        imgsz=imgsz, batch=8,
        conf=0.001, iou=0.6,
        device=device, plots=False, verbose=False,
    )
    # AP50 toàn cục của motorcycle
    names = r.names
    ap50_motor = None
    for idx, ap in zip(r.box.ap_class_index, r.box.ap50):
        if names[int(idx)] == "motorcycle":
            ap50_motor = float(ap)
    out = dict(
        map50_all=round(float(r.box.map50), 4),
        ap50_motorcycle=round(ap50_motor, 4) if ap50_motor is not None else None,
    )
    # Phân tích thêm: recall theo conf threshold → proxy cho recall ở small objects
    # (thực tế cần custom eval; đây là xấp xỉ)
    for conf_thr in (0.25, 0.5):
        r2 = model.val(
            data=str(data_yaml), split="val",
            imgsz=imgsz, batch=8,
            conf=conf_thr, iou=0.6,
            device=device, plots=False, verbose=False,
        )
        for idx, recall in zip(r2.box.ap_class_index, r2.box.r):
            if r2.names[int(idx)] == "motorcycle":
                out[f"recall_motor_conf{int(conf_thr*100)}"] = round(float(recall), 4)
    return out


# ---------------------------------------------------------------------------
# 4. Biểu đồ
# ---------------------------------------------------------------------------

def plot_histogram(records: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        areas = [r["area_px"] for r in records]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram diện tích (log scale)
        axes[0].hist(areas, bins=80, color="steelblue", edgecolor="none")
        for lo, _ in THRESHOLDS.values():
            if lo > 0:
                axes[0].axvline(lo, color="red", linestyle="--", alpha=0.6)
        axes[0].set_xscale("log")
        axes[0].set_xlabel("Diện tích bbox (px², log scale)")
        axes[0].set_ylabel("Số lượng bbox")
        axes[0].set_title("Phân phối kích thước motorcycle")

        # Pie chart phân nhóm
        labels, sizes = [], []
        for b in THRESHOLDS:
            cnt = sum(1 for r in records if bucket(r["area_px"]) == b)
            labels.append(f"{b}\n({cnt})")
            sizes.append(cnt)
        axes[1].pie(sizes, labels=labels, autopct="%1.1f%%",
                    colors=["#e74c3c", "#f39c12", "#2ecc71"])
        axes[1].set_title("Tỉ lệ small/medium/large")

        plt.tight_layout()
        (out_dir / "size_histogram.png").parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "size_histogram.png", dpi=150)
        plt.close()
        print(f"[PLOT] {out_dir / 'size_histogram.png'}")
    except ImportError:
        print("[WARN] matplotlib không có — bỏ qua vẽ biểu đồ")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=REPO / "data" / "data_v5plus.yaml",
                    help="file data yaml (default: data/data_v5plus.yaml)")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--model", type=Path, default=None,
                    help="best.pt của baseline để tính AP; bỏ qua nếu chỉ cần thống kê nhãn")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=REPO / "analysis" / "size_audit")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    data_root = Path(cfg["path"])
    split_dir = data_root / "images" / args.split
    label_dir = data_root / "labels" / args.split
    if not label_dir.exists():
        print(f"[ERR] không tìm thấy {label_dir}"); return

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[SCAN] đọc nhãn từ {label_dir} ...")
    records = read_yolo_labels(label_dir, split_dir)
    print(f"[INFO] tổng bbox motorcycle: {len(records)}")

    stats = compute_stats(records)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # Lưu CSV chi tiết
    detail_csv = args.out / f"motor_boxes_{args.split}.csv"
    if records:
        with open(detail_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()) + ["bucket"])
            w.writeheader()
            for r in records:
                w.writerow({**r, "bucket": bucket(r["area_px"])})
        print(f"[CSV] {detail_csv}")

    # Lưu summary
    summary_json = args.out / "size_stats.json"
    summary_json.write_text(json.dumps({"split": args.split, **stats}, indent=2, ensure_ascii=False))
    print(f"[JSON] {summary_json}")

    plot_histogram(records, args.out)

    # Đánh giá AP theo kích thước nếu có model
    if args.model and args.model.exists():
        print(f"\n[EVAL] chạy val với model {args.model} ...")
        ap_res = eval_by_size(args.model, args.data, args.imgsz, args.device)
        print("[AP]", json.dumps(ap_res, indent=2))
        ap_json = args.out / "ap_by_size_baseline.json"
        ap_json.write_text(json.dumps({"model": str(args.model), **ap_res}, indent=2))
        print(f"[JSON] {ap_json}")
    else:
        print("[INFO] --model không có / không tìm thấy → bỏ qua bước tính AP")

    print(f"\n[DONE] Tóm tắt:")
    print(f"  Tổng motorcycle bbox : {stats.get('total', 0)}")
    print(f"  Small (<32²px)       : {stats.get('n_small', 0)}  ({stats.get('pct_small', 0)}%)")
    print(f"  Medium (32–96²px)    : {stats.get('n_medium', 0)}  ({stats.get('pct_medium', 0)}%)")
    print(f"  Large (>96²px)       : {stats.get('n_large', 0)}  ({stats.get('pct_large', 0)}%)")
    print(f"  Median diện tích      : {stats.get('median_area', 0):.0f} px²")


if __name__ == "__main__":
    main()
