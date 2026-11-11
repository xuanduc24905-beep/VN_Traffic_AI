"""Thí nghiệm huấn luyện cải tiến trên nền V5+ (mỗi preset chỉ đổi 1 nhóm tham số).

Mục tiêu: bắt tốt hơn xe ở xa (nhỏ), xe bị che khuất, xe trong bóng râm/tối,
và class hiếm (bus, truck). Tất cả preset kế thừa cấu hình V5+ rồi ghi đè
đúng phần cần thử, để so sánh với `baseline` là công bằng (ablation).

Dùng:
    python scripts/training/v8_experiments.py --list
    python scripts/training/v8_experiments.py --preset freeze10 --dry-run
    python scripts/training/v8_experiments.py --preset aug_small --epochs 40
    python scripts/training/v8_experiments.py --preset baseline --fraction 0.1 --epochs 3   # chạy thử nhanh

Nguyên tắc đánh giá (khớp góp ý của cô):
- Sau mỗi lần train chỉ đánh giá trên VAL, ghi vào runs/detect/exp_summary.csv.
- Chọn preset tốt nhất bằng VAL. Chỉ chạy `--final --preset <tên>` (đánh giá TEST)
  MỘT LẦN sau khi đã chốt.

Lưu ý quan trọng về copy_paste:
- V5+ đặt copy_paste=0.3, nhưng trong Ultralytics copy_paste chỉ hoạt động khi
  nhãn có segmentation (polygon). Nhãn của dự án là bbox, nên tham số này
  không có tác dụng. Preset `baseline` giữ nguyên để tái lập V5+, các preset
  khác tắt hẳn (copy_paste=0) cho khỏi hiểu nhầm.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# ----- Cấu hình nền = V5+ (scripts/training/v5_plus_960.py) -----
BASE = dict(
    epochs=40, batch=16, imgsz=960,
    lr0=0.005, lrf=0.01, cos_lr=True,
    patience=15, optimizer="SGD", seed=42,
    workers=8, amp=True, save_period=5,
    mixup=0.1, copy_paste=0.3,   # copy_paste không tác dụng với nhãn bbox (xem docstring)
)

# ----- Preset: (mô tả, ghi đè tham số, kiến trúc) -----
# Kiến trúc: "yolov8s.pt" = chuẩn; "yolov8s-p2.yaml" = thêm head P2 (stride 4) cho vật nhỏ.
PRESETS: dict[str, dict] = {
    "baseline": dict(
        desc="Tái lập V5+ (yolov8s, imgsz 960).",
        arch="yolov8s.pt", over={}),
    "freeze10": dict(
        desc="Đóng băng 10 lớp đầu (backbone). Giảm overfit khi data nhỏ, nhưng có thể "
             "hạn chế thích nghi miền mới.",
        arch="yolov8s.pt", over=dict(freeze=10, copy_paste=0.0)),
    "aug_small": dict(
        desc="Xe ở xa: scale jitter rộng + mosaic đủ lâu, giảm mosaic ở cuối.",
        arch="yolov8s.pt",
        over=dict(scale=0.7, mosaic=1.0, translate=0.15, close_mosaic=10, copy_paste=0.0)),
    "aug_light": dict(
        desc="Bóng râm / ban đêm / ngược sáng: tăng nhiễu độ sáng (hsv_v) và độ bão hoà.",
        arch="yolov8s.pt",
        over=dict(hsv_v=0.6, hsv_s=0.8, hsv_h=0.02, copy_paste=0.0)),
    "aug_occl": dict(
        desc="Xe bị che khuất: mixup mạnh hơn để học khi box chồng lấp, mosaic tắt muộn.",
        arch="yolov8s.pt",
        over=dict(mixup=0.25, mosaic=1.0, close_mosaic=15, copy_paste=0.0)),
    "loss_cls": dict(
        desc="Class hiếm (bus, truck): tăng trọng số cls loss (0.5 -> 1.0).",
        arch="yolov8s.pt", over=dict(cls=1.0, copy_paste=0.0)),
    "imgsz1280": dict(
        desc="Tăng độ phân giải 960 -> 1280 (batch 8 cho vừa VRAM).",
        arch="yolov8s.pt", over=dict(imgsz=1280, batch=8, copy_paste=0.0)),
    "p2_head": dict(
        desc="Kiến trúc: thêm head P2 (stride 4) cho vật rất nhỏ. Chậm hơn, nặng hơn.",
        arch="yolov8s-p2.yaml", over=dict(copy_paste=0.0)),
    "combo": dict(
        desc="Gộp aug_small + aug_light + loss_cls (chỉ thử SAU khi từng phần đã có lợi).",
        arch="yolov8s.pt",
        over=dict(scale=0.7, mosaic=1.0, translate=0.15, close_mosaic=10,
                  hsv_v=0.6, hsv_s=0.8, hsv_h=0.02, cls=1.0, copy_paste=0.0)),
}


def local_data_yaml(data_yaml: Path, root: Path, out_dir: Path) -> Path:
    """data_v5.yaml ghi đường dẫn tuyệt đối của máy train cũ. Ghi một bản local
    trỏ vào <root>/data/<tên thư mục> nếu đường dẫn gốc không tồn tại."""
    import yaml
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    p = Path(cfg["path"])
    if not p.exists():
        p = root / "data" / p.name
    cfg["path"] = str(p)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"_{data_yaml.stem}_local.yaml"
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return out


def build_cfg(preset: str, args, data_yaml: Path, runs_dir: Path) -> dict:
    cfg = dict(BASE)
    cfg.update(PRESETS[preset]["over"])
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch:
        cfg["batch"] = args.batch
    if args.fraction:
        cfg["fraction"] = args.fraction
    cfg.update(
        data=str(data_yaml), project=str(runs_dir), name=f"exp_{preset}",
        exist_ok=True, verbose=True, device=args.device,
    )
    return cfg


def append_summary(runs_dir: Path, row: dict) -> None:
    path = runs_dir / "exp_summary.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def evaluate(weights: Path, data_yaml: Path, split: str, imgsz: int, device) -> dict:
    from ultralytics import YOLO
    m = YOLO(str(weights))
    r = m.val(data=str(data_yaml), split=split, imgsz=imgsz, batch=8,
              conf=0.001, iou=0.6, device=device, plots=False, verbose=False)
    names = r.names
    out = dict(split=split, precision=round(float(r.box.mp), 4),
               recall=round(float(r.box.mr), 4),
               map50=round(float(r.box.map50), 4), map50_95=round(float(r.box.map), 4))
    for idx, ap in zip(r.box.ap_class_index, r.box.ap50):
        out[f"ap50_{names[int(idx)]}"] = round(float(ap), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="liệt kê preset")
    ap.add_argument("--preset", choices=list(PRESETS))
    ap.add_argument("--root", type=Path, default=REPO, help="thư mục gốc dự án")
    ap.add_argument("--data", type=Path, default=None, help="mặc định data/data_v5.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--fraction", type=float, default=None,
                    help="dùng một phần tập train để chạy thử (vd 0.1)")
    ap.add_argument("--device", default=None, help="vd 0 hoặc cpu")
    ap.add_argument("--dry-run", action="store_true", help="chỉ in cấu hình")
    ap.add_argument("--final", action="store_true",
                    help="đánh giá TEST cho preset đã train (chỉ làm 1 lần khi chốt)")
    args = ap.parse_args()

    if args.list:
        for k, v in PRESETS.items():
            print(f"{k:10s} {v['arch']:16s} {v['desc']}")
        return 0
    if not args.preset:
        ap.error("cần --preset hoặc --list")

    root = args.root.resolve()
    runs_dir = root / "runs" / "detect"
    data_yaml = args.data or (root / "data" / "data_v5.yaml")
    local_yaml = local_data_yaml(data_yaml, root, runs_dir)
    cfg = build_cfg(args.preset, args, local_yaml, runs_dir)
    arch = PRESETS[args.preset]["arch"]
    best = runs_dir / f"exp_{args.preset}" / "weights" / "best.pt"

    if args.final:
        res = evaluate(best, local_yaml, "test", cfg["imgsz"], args.device)
        append_summary(runs_dir, dict(preset=args.preset, **res))
        print("[FINAL TEST]", res)
        return 0

    print(f"[PRESET] {args.preset}: {PRESETS[args.preset]['desc']}")
    print(f"[ARCH]   {arch}")
    diff = {k: v for k, v in cfg.items() if BASE.get(k) != v and k in
            set(BASE) | set(PRESETS[args.preset]['over'])}
    print(f"[KHÁC V5+] {diff}")
    if args.dry_run:
        print("[CFG]", cfg)
        return 0

    from ultralytics import YOLO
    if arch.endswith(".yaml"):
        model = YOLO(arch).load("yolov8s.pt")   # cấu trúc mới + nạp trọng số khớp từ COCO
    else:
        model = YOLO(arch)
    model.train(**cfg)

    res = evaluate(best, local_yaml, "val", cfg["imgsz"], args.device)
    append_summary(runs_dir, dict(preset=args.preset, **res))
    print("[VAL]", res)
    print(f"[DONE] best: {best}  | tổng hợp: {runs_dir / 'exp_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
