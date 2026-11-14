"""train_p2_motor.py — So sánh P2 head vs baseline trên cùng tập data_v5plus.

Baseline đã có sẵn (exp_baseline, val AP50_moto=0.885). Script này train thêm
các biến thể P2, cùng tập + cùng hyperparams, để biết P2 có bắt xe máy tốt hơn không.

Thí nghiệm kiểm soát (mỗi run CHỈ đổi 1 biến):

  baseline  — YOLOv8s chuẩn, imgsz=960           ← ĐÃ CÓ (exp_baseline)
  p2_head   — YOLOv8s-P2, imgsz=960               ← chỉ đổi kiến trúc
  p2_aug    — YOLOv8s-P2 + aug tối ưu cho vật nhỏ ← kiến trúc + aug
  p2_1280   — YOLOv8s-P2, imgsz=1280              ← kiến trúc + resolution

Kết quả baseline đã có (val):
  mAP50=0.8593  |  AP50_motorcycle=0.885  |  Recall=0.8067

Dùng nhanh:
  # Chạy tuần tự cả 3 biến thể P2:
  python scripts/small_motor/train_p2_motor.py --all --device 0

  # Chạy từng cái:
  python scripts/small_motor/train_p2_motor.py --exp p2_head --device 0
  python scripts/small_motor/train_p2_motor.py --exp p2_aug  --device 0
  python scripts/small_motor/train_p2_motor.py --exp p2_1280 --device 0

  # Chạy thử nhanh (3 epoch, 10% data):
  python scripts/small_motor/train_p2_motor.py --exp p2_head --epochs 3 --fraction 0.1 --device cpu

  # Chạy tiếp sau khi bị ngắt:
  python scripts/small_motor/train_p2_motor.py --exp p2_head --resume

  # Chỉ xem kết quả baseline + các run đã có:
  python scripts/small_motor/train_p2_motor.py --results

Kết quả:
  runs/detect/motor_p2_head/weights/best.pt
  runs/detect/motor_p2_aug/weights/best.pt
  runs/detect/motor_p2_1280/weights/best.pt
  runs/detect/motor_p2_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# ─── Data yaml thực tế đã dùng cho baseline ──────────────────────────────────
# File này Ultralytics tự tạo khi chạy v8_experiments.py, trỏ vào merged_v5plus:
#   path: /home/xuand/vn_traffic_ai/data/merged_v5plus
#   train: images/train   (7237 ảnh)
#   val:   images/val     (1466 ảnh)
#   test:  images/test
DATA_YAML = REPO / "runs" / "detect" / "_data_v5plus_local.yaml"
DATA_YAML_FALLBACK = REPO / "data" / "data_v5plus.yaml"

# Trọng số baseline đã train
BASELINE_WEIGHTS = REPO / "runs" / "detect" / "exp_baseline" / "weights" / "best.pt"

# Kết quả baseline đã biết (từ exp_summary.csv)
BASELINE_VAL = dict(
    map50=0.8593,
    ap50_motorcycle=0.8850,
    ap50_car=0.8525,
    ap50_bus=0.7963,
    ap50_truck=0.9035,
    recall=0.8067,
    precision=0.8199,
)

# ─── Cấu hình nền V5+ (khớp hoàn toàn với exp_baseline) ─────────────────────
BASE = dict(
    epochs=40, batch=16, imgsz=960,
    lr0=0.005, lrf=0.01, cos_lr=True,
    patience=15, optimizer="SGD", seed=42,
    workers=4, amp=True, save_period=5,
    mixup=0.1, copy_paste=0.0,
    mosaic=1.0, close_mosaic=10,
    scale=0.5, translate=0.1, fliplr=0.5,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
)

# Aug bổ sung cho vật nhỏ (xe máy xa)
_AUG_SMALL = dict(
    scale=0.7,       # jitter lớn → nhiều xe nhỏ trong mosaic
    translate=0.15,
    hsv_v=0.5,       # giả lập ánh sáng ngược / tối (CCTV ban đêm)
    hsv_s=0.75,
)

# ─── Định nghĩa thí nghiệm ───────────────────────────────────────────────────
# arch=None      → YOLO("yolov8s.pt")
# arch="*.yaml"  → YOLO("yolov8s-p2.yaml").load("yolov8s.pt")
EXPS: dict[str, dict] = {
    "p2_head": dict(
        desc="YOLOv8s-P2 (stride-4 head), imgsz=960. Chỉ đổi kiến trúc so với baseline.",
        arch="yolov8s-p2.yaml",
        over={},
    ),
    "p2_aug": dict(
        desc="YOLOv8s-P2 + aug tối ưu cho vật nhỏ (scale=0.7, hsv_v=0.5).",
        arch="yolov8s-p2.yaml",
        over=_AUG_SMALL,
    ),
    "p2_1280": dict(
        desc="YOLOv8s-P2, imgsz=1280, batch=8. Resolution cao hơn + P2.",
        arch="yolov8s-p2.yaml",
        over=dict(imgsz=1280, batch=8, **_AUG_SMALL),
    ),
}

MOTOR = "motorcycle"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_data_yaml() -> Path:
    if DATA_YAML.exists():
        return DATA_YAML
    if DATA_YAML_FALLBACK.exists():
        return DATA_YAML_FALLBACK
    raise FileNotFoundError(
        f"Không tìm thấy data yaml.\n"
        f"  Thử 1: {DATA_YAML}\n"
        f"  Thử 2: {DATA_YAML_FALLBACK}\n"
        f"Chạy scripts/training/v8_experiments.py --preset baseline --dry-run một lần để tạo file local yaml."
    )


def build_model(arch: str | None, pretrained: str = "yolov8s.pt"):
    from ultralytics import YOLO
    if arch is None:
        return YOLO(pretrained)
    if arch.endswith(".yaml"):
        return YOLO(arch).load(pretrained)
    return YOLO(arch)


def run_val(weights: Path, data_yaml: Path, imgsz: int, device) -> dict:
    from ultralytics import YOLO
    r = YOLO(str(weights)).val(
        data=str(data_yaml), split="val",
        imgsz=imgsz, batch=8,
        conf=0.001, iou=0.6,
        device=device, plots=False, verbose=False,
    )
    out = dict(
        map50=round(float(r.box.map50), 4),
        map50_95=round(float(r.box.map), 4),
        precision=round(float(r.box.mp), 4),
        recall=round(float(r.box.mr), 4),
    )
    for idx, ap50 in zip(r.box.ap_class_index, r.box.ap50):
        out[f"ap50_{r.names[int(idx)]}"] = round(float(ap50), 4)
    for idx, rec in zip(r.box.ap_class_index, r.box.r):
        if r.names[int(idx)] == MOTOR:
            out[f"recall_{MOTOR}"] = round(float(rec), 4)
    return out


def save_to_csv(runs_dir: Path, row: dict) -> None:
    path = runs_dir / "motor_p2_summary.csv"
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def print_comparison(results: dict[str, dict]) -> None:
    """In bảng so sánh với baseline."""
    all_rows = {"baseline (đã có)": BASELINE_VAL, **results}
    print(f"\n{'='*72}")
    print(f"{'MODEL':22s}  {'mAP50':7s}  {'AP50_moto':10s}  {'Recall_moto':12s}  {'Δ AP50_moto':11s}")
    print(f"{'-'*72}")
    base_ap = BASELINE_VAL["ap50_motorcycle"]
    for name, r in all_rows.items():
        ap = r.get("ap50_motorcycle", r.get("ap50_motor", "—"))
        delta = f"{float(ap)-base_ap:+.4f}" if isinstance(ap, float) and name != "baseline (đã có)" else "   —"
        marker = " ◄ BEST" if isinstance(ap, float) and ap == max(
            v.get("ap50_motorcycle", 0) for v in all_rows.values() if isinstance(v.get("ap50_motorcycle"), float)
        ) else ""
        print(f"{name:22s}  {r.get('map50','—'):7}  {ap:10}  "
              f"{r.get('recall_motorcycle', r.get('recall','—')):12}  {delta}{marker}")
    print(f"{'='*72}")


# ─── Chạy 1 thí nghiệm ───────────────────────────────────────────────────────

def run_exp(exp_name: str, args, data_yaml: Path) -> dict | None:
    exp = EXPS[exp_name]
    runs_dir = REPO / "runs" / "detect"
    run_name = f"motor_{exp_name}"
    best = runs_dir / run_name / "weights" / "best.pt"
    last = runs_dir / run_name / "weights" / "last.pt"

    cfg = dict(BASE)
    cfg.update(exp["over"])
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch:
        cfg["batch"] = args.batch
    if args.fraction:
        cfg["fraction"] = args.fraction
    if args.seed is not None:
        cfg["seed"] = args.seed

    cfg.update(
        data=str(data_yaml),
        project=str(runs_dir),
        name=run_name,
        exist_ok=True,
        verbose=True,
        device=args.device,
    )
    imgsz = cfg["imgsz"]

    # In plan
    print(f"\n{'#'*65}")
    print(f"# EXP: {run_name}")
    print(f"# {exp['desc']}")
    print(f"# ARCH: {'yolov8s.pt (chuẩn)' if exp['arch'] is None else exp['arch']}")
    changed = {k: v for k, v in exp["over"].items()}
    if changed:
        print(f"# KHÁC baseline: {json.dumps(changed, ensure_ascii=False)}")
    print(f"# Data: {data_yaml}  ({7237} train / {1466} val, imgsz={imgsz})")
    print(f"{'#'*65}\n")

    if args.dry_run:
        print("[DRY-RUN] Bỏ qua huấn luyện.")
        return None

    # Huấn luyện
    if args.resume:
        if not last.exists():
            print(f"[ERR] Không có {last}. Chạy lần đầu trước.")
            return None
        from ultralytics import YOLO
        YOLO(str(last)).train(resume=True)
    else:
        model = build_model(exp["arch"], pretrained=args.pretrained)
        model.train(**cfg)

    if not best.exists():
        print(f"[WARN] Không tìm thấy {best}.")
        return None

    # Đánh giá val
    print(f"\n[EVAL VAL] {run_name} ...")
    res = run_val(best, data_yaml, imgsz, args.device)

    # So sánh ngay với baseline
    base_ap = BASELINE_VAL["ap50_motorcycle"]
    cur_ap = res.get("ap50_motorcycle", 0)
    delta = cur_ap - base_ap
    sign = "▲" if delta > 0 else "▼"
    print(f"\n[KẾT QUẢ] {run_name}")
    print(f"  mAP50          : {res.get('map50')}  (baseline: {BASELINE_VAL['map50']})")
    print(f"  AP50_motorcycle: {cur_ap}  (baseline: {base_ap})  {sign} {abs(delta):.4f}")
    print(f"  Recall_moto    : {res.get('recall_motorcycle','—')}  (baseline: {BASELINE_VAL['recall']})")

    row = dict(
        exp=exp_name, run=run_name,
        arch=exp["arch"] or "yolov8s.pt",
        imgsz=imgsz, epochs=cfg["epochs"],
        **res,
        delta_ap50_moto=round(delta, 4),
    )
    save_to_csv(runs_dir, row)
    print(f"[CSV] {runs_dir / 'motor_p2_summary.csv'}")
    print(f"[BEST] {best}\n")
    return res


# ─── Chỉ xem kết quả đã có ───────────────────────────────────────────────────

def show_results() -> None:
    """Đọc motor_p2_summary.csv và in bảng."""
    csv_path = REPO / "runs" / "detect" / "motor_p2_summary.csv"
    results = {}
    if csv_path.exists():
        import csv as _csv
        with open(csv_path, encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                name = row.get("run", row.get("exp", "?"))
                results[name] = {k: _safe_float(v) for k, v in row.items()}
        print(f"[CSV] đọc từ {csv_path}")
    else:
        print(f"[INFO] Chưa có {csv_path}. Chạy --exp hoặc --all trước.")
    print_comparison(results)


def _safe_float(v: str):
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--exp", choices=list(EXPS),
                       help="Chạy 1 thí nghiệm: p2_head | p2_aug | p2_1280")
    group.add_argument("--all", action="store_true",
                       help="Chạy tuần tự: p2_head → p2_aug → p2_1280")
    group.add_argument("--results", action="store_true",
                       help="Chỉ xem bảng kết quả đã có, không train")
    group.add_argument("--list", action="store_true",
                       help="Liệt kê thí nghiệm")

    ap.add_argument("--pretrained", default="yolov8s.pt",
                    help="Trọng số COCO pretrained (default: yolov8s.pt)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch",  type=int, default=None)
    ap.add_argument("--fraction", type=float, default=None,
                    help="Dùng x%% dữ liệu để chạy thử (vd: 0.1)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None,
                    help="GPU id hoặc 'cpu' (vd: 0  |  0,1  |  cpu)")
    ap.add_argument("--resume", action="store_true",
                    help="Tiếp tục từ last.pt (chỉ dùng với --exp)")
    ap.add_argument("--dry-run", action="store_true",
                    help="In cấu hình, không huấn luyện")
    args = ap.parse_args()

    if args.list:
        print(f"\n{'EXP':10s}  {'ARCH':20s}  MÔ TẢ")
        print("-" * 75)
        print(f"{'baseline':10s}  {'yolov8s.pt':20s}  Đã train (exp_baseline). AP50_moto=0.885")
        for name, cfg in EXPS.items():
            arch = cfg["arch"] or "yolov8s.pt"
            print(f"{name:10s}  {arch:20s}  {cfg['desc']}")
        return 0

    if args.results:
        show_results()
        return 0

    # Lấy data yaml
    try:
        data_yaml = get_data_yaml()
    except FileNotFoundError as e:
        print(f"[ERR] {e}")
        return 1
    print(f"[DATA] {data_yaml}")

    exps_to_run = list(EXPS.keys()) if args.all else [args.exp]
    all_results: dict[str, dict] = {}

    for exp_name in exps_to_run:
        result = run_exp(exp_name, args, data_yaml)
        if result:
            all_results[exp_name] = result

    # Bảng tổng kết
    if all_results:
        print_comparison(all_results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
