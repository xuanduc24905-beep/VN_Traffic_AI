"""eval_compare.py — So sánh tất cả model đã train trong pipeline small-motor.

Đọc best.pt của mỗi run, chạy val trên cùng split, xuất bảng CSV + biểu đồ bar.

Dùng:
  # So sánh tất cả run motor_* tự động:
  python scripts/small_motor/eval_compare.py --data data/data_v5plus.yaml --device 0

  # Chỉ định model cụ thể:
  python scripts/small_motor/eval_compare.py \\
      --models baseline=runs/detect/motor_baseline/weights/best.pt \\
               p2_head=runs/detect/motor_p2_head/weights/best.pt \\
               tile_mixed=runs/detect/motor_tile_mixed/weights/best.pt \\
      --data data/data_v5plus.yaml --device 0

  # Thêm --test để đánh giá trên test set (chỉ 1 lần sau khi chốt model):
  python scripts/small_motor/eval_compare.py --data data/data_v5plus.yaml --test --device 0

Kết quả:
  analysis/model_compare/compare_val.csv
  analysis/model_compare/compare_val.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MOTOR_CLASS = "motorcycle"
MOTOR_ID = 0


def find_motor_runs(runs_dir: Path) -> dict[str, Path]:
    """Tự động tìm tất cả motor_* runs có best.pt."""
    result = {}
    if not runs_dir.exists():
        return result
    for run_dir in sorted(runs_dir.iterdir()):
        if run_dir.is_dir() and run_dir.name.startswith("motor_"):
            best = run_dir / "weights" / "best.pt"
            if best.exists():
                # Tên ngắn gọn: bỏ prefix 'motor_'
                short_name = run_dir.name[len("motor_"):]
                result[short_name] = best
    return result


def evaluate_model(
    name: str,
    weights: Path,
    data_yaml: Path,
    split: str,
    imgsz: int,
    device,
) -> dict:
    """Chạy val, trả về dict kết quả."""
    from ultralytics import YOLO
    print(f"\n[EVAL] {name} ({weights.name}) on {split} ...")
    model = YOLO(str(weights))
    r = model.val(
        data=str(data_yaml), split=split,
        imgsz=imgsz, batch=8,
        conf=0.001, iou=0.6,
        device=device, plots=False, verbose=False,
    )
    res = dict(
        name=name,
        split=split,
        weights=str(weights),
        map50=round(float(r.box.map50), 4),
        map50_95=round(float(r.box.map), 4),
        precision=round(float(r.box.mp), 4),
        recall=round(float(r.box.mr), 4),
    )
    for idx, ap50 in zip(r.box.ap_class_index, r.box.ap50):
        cls = r.names[int(idx)]
        res[f"ap50_{cls}"] = round(float(ap50), 4)
    for idx, recall in zip(r.box.ap_class_index, r.box.r):
        cls = r.names[int(idx)]
        if cls == MOTOR_CLASS:
            res[f"recall_{cls}"] = round(float(recall), 4)
    return res


def save_csv(results: list[dict], out_path: Path) -> None:
    if not results:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"[CSV] {out_path}")


def plot_compare(results: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        names = [r["name"] for r in results]
        map50 = [r.get("map50", 0) for r in results]
        ap50_motor = [r.get("ap50_motorcycle", 0) for r in results]
        recall_motor = [r.get("recall_motorcycle", 0) for r in results]

        x = np.arange(len(names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(max(10, len(names) * 2), 5))
        b1 = ax.bar(x - width, map50, width, label="mAP50 (all)", color="#3498db")
        b2 = ax.bar(x, ap50_motor, width, label="AP50 motorcycle", color="#e74c3c")
        b3 = ax.bar(x + width, recall_motor, width, label="Recall motorcycle", color="#2ecc71")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("So sánh model: mAP50, AP50-motorcycle, Recall-motorcycle")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        for bars in (b1, b2, b3):
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.annotate(f"{h:.3f}",
                                xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=7)

        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[PLOT] {out_path}")
    except ImportError:
        print("[WARN] matplotlib không có — bỏ qua vẽ biểu đồ")


def print_table(results: list[dict]) -> None:
    """In bảng so sánh ra terminal."""
    if not results:
        return
    cols = ["name", "map50", "ap50_motorcycle", "recall_motorcycle", "map50_95"]
    header = f"{'MODEL':20s}  {'mAP50':8s}  {'AP50_moto':10s}  {'R_moto':8s}  {'mAP50-95':10s}"
    print(f"\n{'='*65}")
    print("BẢNG SO SÁNH MODEL (val)")
    print(header)
    print("-" * 65)
    # Sắp xếp theo AP50 motorcycle giảm dần
    for r in sorted(results, key=lambda x: x.get("ap50_motorcycle", 0), reverse=True):
        print(f"{r['name']:20s}  {r.get('map50','—'):8}  "
              f"{r.get('ap50_motorcycle','—'):10}  "
              f"{r.get('recall_motorcycle','—'):8}  "
              f"{r.get('map50_95','—'):10}")
    print(f"{'='*65}\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data", type=Path, default=Path("data/data_v5plus.yaml"),
                    help="Data yaml để eval (default: data/data_v5plus.yaml)")
    ap.add_argument("--models", nargs="+", default=None,
                    metavar="NAME=PATH",
                    help="Chỉ định model: 'tên=path/best.pt'. Bỏ qua → tự tìm motor_* runs.")
    ap.add_argument("--runs-dir", type=Path, default=None,
                    help="Thư mục chứa các run (default: runs/detect)")
    ap.add_argument("--imgsz", type=int, default=960,
                    help="Kích thước ảnh eval (default: 960)")
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="Split để eval (default: val)")
    ap.add_argument("--test", action="store_true",
                    help="Đánh giá trên test split (chỉ dùng 1 lần sau khi chốt model)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Thư mục output (default: analysis/model_compare)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    split = "test" if args.test else args.split
    runs_dir = args.runs_dir or (REPO / "runs" / "detect")
    out_dir = args.out or (REPO / "analysis" / "model_compare")
    data_yaml = args.data if args.data.is_absolute() else REPO / args.data
    if not data_yaml.exists():
        data_yaml = REPO / "data" / args.data
    if not data_yaml.exists():
        ap.error(f"Không tìm thấy data yaml: {args.data}")

    # Tập hợp model cần eval
    models: dict[str, Path] = {}
    if args.models:
        for spec in args.models:
            name, _, path_str = spec.partition("=")
            if not name or not path_str:
                ap.error(f"Dạng sai: '{spec}'. Cần 'tên=path/best.pt'")
            p = Path(path_str)
            if not p.is_absolute():
                p = REPO / p
            if not p.exists():
                print(f"[WARN] Không tìm thấy {p}, bỏ qua.")
                continue
            models[name.strip()] = p
    else:
        models = find_motor_runs(runs_dir)
        if not models:
            print(f"[ERR] Không tìm thấy run nào trong {runs_dir}.")
            print("  Chạy train_p2_motor.py hoặc tile_motor_train.py trước.")
            return 1
        print(f"[AUTO] Tìm thấy {len(models)} runs: {list(models.keys())}")

    if args.test:
        print("\n[CANH BÁO] Đang đánh giá trên TEST set.")
        print("  Chỉ làm điều này SAU KHI đã chốt model qua val.")
        print("  Kết quả test dùng để báo cáo cuối, không dùng để chọn model.\n")

    results = []
    for name, weights in models.items():
        try:
            res = evaluate_model(name, weights, data_yaml, split, args.imgsz, args.device)
            results.append(res)
        except Exception as e:
            print(f"[ERR] {name}: {e}")

    if not results:
        print("[ERR] Không có kết quả nào.")
        return 1

    print_table(results)

    csv_name = f"compare_{split}.csv"
    png_name = f"compare_{split}.png"
    save_csv(results, out_dir / csv_name)
    plot_compare(results, out_dir / png_name)

    # Tìm model tốt nhất theo AP50 motorcycle
    best = max(results, key=lambda x: x.get("ap50_motorcycle", 0))
    print(f"[BEST] {best['name']}: AP50_moto={best.get('ap50_motorcycle','—')}, "
          f"mAP50={best.get('map50','—')}, Recall_moto={best.get('recall_motorcycle','—')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
