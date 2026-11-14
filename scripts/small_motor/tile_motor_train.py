"""tile_motor_train.py — Cắt ảnh theo vùng ưu tiên motorcycle nhỏ + huấn luyện.

Dùng cùng tập data_v5plus (merged_v5plus, 7237 train / 1466 val, imgsz=960)
với baseline để so sánh fair.

Chiến lược tile:
  - Cắt ảnh train thành ô 640×640 chồng lấn 25%.
  - Ô có ≥1 bbox motorcycle → giữ 100%.
  - Ô không có motorcycle   → giữ theo --keep-no-motor (mặc định 20%).
  - Val/test KHÔNG cắt → đánh giá fair trên ảnh đầy đủ (imgsz=960).

So sánh 3 mode:
  full   → chỉ ảnh đầy đủ gốc (đối chứng tile)
  tile   → chỉ ô cắt
  mixed  → ảnh đầy đủ + ô cắt (mặc định, thường tốt nhất)

Kết quả baseline đã có (val):
  mAP50=0.8593  |  AP50_motorcycle=0.885  |  Recall=0.8067

Dùng:
  # Chạy mixed mode (khuyến nghị):
  python scripts/small_motor/tile_motor_train.py --mode mixed --device 0

  # So sánh cả 3 mode:
  python scripts/small_motor/tile_motor_train.py --compare --device 0

  # Kết hợp P2 head + tile (mạnh nhất):
  python scripts/small_motor/tile_motor_train.py --mode mixed --p2 --device 0

  # Chạy thử nhanh (3 epoch):
  python scripts/small_motor/tile_motor_train.py --mode mixed --epochs 3 --fraction 0.1 --device cpu

  # Chỉ cắt ảnh, không train:
  python scripts/small_motor/tile_motor_train.py --mode mixed --no-train

Kết quả:
  data/merged_v5plus_tiles_<mode>/     ← dataset đã cắt
  runs/detect/motor_tile_<mode>/       ← model
  runs/detect/motor_tile_summary.csv   ← bảng so sánh
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]

# ─── Nguồn dữ liệu (cùng tập với baseline) ───────────────────────────────────
SRC_DEFAULT   = REPO / "data" / "merged_v5plus"          # 7237 train / 1466 val
DATA_YAML_LOCAL = REPO / "runs" / "detect" / "_data_v5plus_local.yaml"

# Kết quả baseline đã biết — dùng để in bảng so sánh
BASELINE_VAL = dict(
    map50=0.8593,
    ap50_motorcycle=0.8850,
    recall=0.8067,
)

# ─── Cấu hình nền V5+ — khớp exp_baseline ─────────────────────────────────────
# imgsz=960: train trên tile 640px nhưng Ultralytics resize lên 960.
# Xe máy trong tile → chiếm pixel lớn hơn → model học feature ở resolution tốt hơn.
BASE_TRAIN = dict(
    epochs=40, batch=16, imgsz=960,
    lr0=0.005, lrf=0.01, cos_lr=True,
    patience=15, optimizer="SGD", seed=42,
    workers=4, amp=True, save_period=5,
    mixup=0.1, copy_paste=0.0,
    mosaic=1.0, close_mosaic=10,
    scale=0.5, translate=0.1, fliplr=0.5,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
)

MOTOR_ID = 0        # motorcycle = class 0
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─── Tiện ích đọc/ghi ────────────────────────────────────────────────────────

def list_images(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS)


def read_yolo(lp: Path) -> np.ndarray:
    if not lp.exists():
        return np.zeros((0, 5), dtype=np.float64)
    rows = []
    for line in lp.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            rows.append([float(x) for x in parts])
    return np.array(rows, dtype=np.float64).reshape(-1, 5)


def write_yolo(lp: Path, rows: np.ndarray) -> None:
    lp.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}" for r in rows]
    lp.write_text("\n".join(lines), encoding="utf-8")


def label_path_for(img: Path) -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return img.parent.parent / "labels" / (img.stem + ".txt")


# ─── Tạo lưới tile ───────────────────────────────────────────────────────────

def make_grid(W: int, H: int, tile: int, overlap: float) -> list[tuple[int,int,int,int]]:
    """Trả về list (x0,y0,x1,y1). Rỗng nếu ảnh ≤ tile."""
    if W <= tile and H <= tile:
        return []
    step = max(int(tile * (1 - overlap)), 1)
    xs = sorted(set(max(0, x) for x in list(range(0, W - tile, step)) + [W - tile]))
    ys = sorted(set(max(0, y) for y in list(range(0, H - tile, step)) + [H - tile]))
    return [(x, y, min(x+tile, W), min(y+tile, H)) for y in ys for x in xs]


# ─── Cắt nhãn vào tile ───────────────────────────────────────────────────────

def clip_to_tile(
    rows: np.ndarray, W: int, H: int,
    tb: tuple[int,int,int,int],
    min_visible: float = 0.5, min_px: int = 4,
) -> np.ndarray:
    if len(rows) == 0:
        return np.zeros((0, 5), dtype=np.float64)
    x0, y0, x1, y1 = tb
    tw, th = x1-x0, y1-y0
    out = []
    for cls, xc, yc, w, h in rows:
        bx1, by1 = (xc-w/2)*W, (yc-h/2)*H
        bx2, by2 = (xc+w/2)*W, (yc+h/2)*H
        ix1 = max(bx1, x0); iy1 = max(by1, y0)
        ix2 = min(bx2, x1); iy2 = min(by2, y1)
        iw, ih = ix2-ix1, iy2-iy1
        if iw < min_px or ih < min_px:
            continue
        if (iw*ih) / max((bx2-bx1)*(by2-by1), 1e-9) < min_visible:
            continue
        out.append([cls, ((ix1+ix2)/2-x0)/tw, ((iy1+iy2)/2-y0)/th, iw/tw, ih/th])
    return np.array(out, dtype=np.float64).reshape(-1, 5)


# ─── Thống kê ─────────────────────────────────────────────────────────────────

def motor_area_stats(label_dir: Path, ref_size: int = 960) -> dict:
    """% bbox motorcycle nhỏ (<32²px tương đối imgsz ref_size)."""
    areas = []
    for lp in sorted(label_dir.rglob("*.txt")):
        for cls, xc, yc, w, h in read_yolo(lp):
            if int(cls) == MOTOR_ID:
                areas.append(w * ref_size * h * ref_size)
    if not areas:
        return {}
    a = np.array(areas)
    return dict(
        n_motor_boxes=len(a),
        median_area_px2=round(float(np.median(a)), 1),
        pct_small=round(100*int((a < 32**2).sum())/len(a), 1),
    )


# ─── Bước 1: Build tile dataset ───────────────────────────────────────────────

def build_tiles(
    src: Path, out: Path,
    tile: int, overlap: float,
    min_visible: float, keep_no_motor: float,
    no_full: bool, rng: random.Random,
) -> dict:
    src_img = src / "images" / "train"
    out_img = out / "images" / "train"
    out_lbl = out / "labels" / "train"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    s = dict(n_tiled=0, n_small=0, tiles_motor=0, tiles_no_motor_kept=0,
             tiles_no_motor_dropped=0, tiles_total=0)

    images = list_images(src_img)
    print(f"[TILE] Xử lý {len(images)} ảnh từ {src_img} ...")
    for img_p in images:
        rows = read_yolo(label_path_for(img_p))
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        H, W = img.shape[:2]
        grid = make_grid(W, H, tile, overlap)

        if not grid:   # ảnh nhỏ hơn tile
            s["n_small"] += 1
            if no_full:
                dst = out_img / img_p.name
                if not dst.exists():
                    shutil.copy2(img_p, dst)
                write_yolo(out_lbl / (img_p.stem + ".txt"), rows)
            continue

        s["n_tiled"] += 1
        for k, tb in enumerate(grid):
            tile_rows = clip_to_tile(rows, W, H, tb, min_visible)
            has_motor = int((tile_rows[:, 0] == MOTOR_ID).sum()) > 0 if len(tile_rows) else False

            if has_motor:
                keep = True
                s["tiles_motor"] += 1
            else:
                keep = rng.random() < keep_no_motor
                if keep:
                    s["tiles_no_motor_kept"] += 1
                else:
                    s["tiles_no_motor_dropped"] += 1

            if not keep:
                continue

            x0, y0, x1, y1 = tb
            name = f"{img_p.stem}_t{k:03d}"
            cv2.imwrite(str(out_img / (name+".jpg")), img[y0:y1, x0:x1],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_yolo(out_lbl / (name+".txt"), tile_rows)
            s["tiles_total"] += 1

    # Thống kê area
    s["after_tile"] = motor_area_stats(out_lbl)
    s["before_tile"] = motor_area_stats(src / "labels" / "train")

    # Ghi yaml — train = [src_full?, out_tile]
    train_dirs = []
    if not no_full:
        train_dirs.append(str(src / "images" / "train"))
    train_dirs.append(str(out / "images" / "train"))

    nc, names = 4, {0: "motorcycle", 1: "car", 2: "bus", 3: "truck"}
    tiles_yaml = out / "tiles.yaml"
    yaml_data: dict = dict(train=train_dirs, val=str(src/"images"/"val"), nc=nc, names=names)
    test_dir = src / "images" / "test"
    if test_dir.exists():
        yaml_data["test"] = str(test_dir)
    tiles_yaml.write_text(yaml.dump(yaml_data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    (out / "tile_manifest.json").write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")

    # In tóm tắt
    pct_before = s["before_tile"].get("pct_small", "—")
    pct_after  = s["after_tile"].get("pct_small", "—")
    print(f"[TILE STATS]")
    print(f"  Ảnh được cắt            : {s['n_tiled']}")
    print(f"  Ảnh nhỏ (không cắt)     : {s['n_small']}")
    print(f"  Tile có motorcycle       : {s['tiles_motor']}")
    print(f"  Tile giữ (no-motor)      : {s['tiles_no_motor_kept']}")
    print(f"  Tile bỏ (no-motor)       : {s['tiles_no_motor_dropped']}")
    print(f"  Tổng tile ghi vào disk   : {s['tiles_total']}")
    print(f"  % motor nhỏ trước cắt   : {pct_before}%  → sau cắt: {pct_after}%")
    print(f"  tiles.yaml               : {tiles_yaml}\n")
    return s


# ─── Bước 2: Huấn luyện ──────────────────────────────────────────────────────

def run_training(tiles_yaml: Path, run_name: str, use_p2: bool, args) -> dict | None:
    runs_dir = REPO / "runs" / "detect"
    runs_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(BASE_TRAIN)
    if args.epochs:   cfg["epochs"]   = args.epochs
    if args.batch:    cfg["batch"]    = args.batch
    if args.fraction: cfg["fraction"] = args.fraction
    cfg.update(
        data=str(tiles_yaml),
        project=str(runs_dir),
        name=run_name,
        exist_ok=True,
        verbose=True,
        device=args.device,
    )

    arch = "yolov8s-p2.yaml" if use_p2 else None
    print(f"[TRAIN] {run_name}")
    print(f"  arch   : {'yolov8s-p2.yaml (P2 head)' if use_p2 else 'yolov8s.pt'}")
    print(f"  data   : {tiles_yaml}")
    print(f"  imgsz  : {cfg['imgsz']}  |  epochs: {cfg['epochs']}  |  batch: {cfg['batch']}")

    if args.dry_run:
        print("[DRY-RUN] Bỏ qua huấn luyện.")
        return None

    from ultralytics import YOLO
    model = YOLO(arch).load(args.pretrained) if use_p2 else YOLO(args.pretrained)
    model.train(**cfg)

    best = runs_dir / run_name / "weights" / "best.pt"
    if not best.exists():
        print(f"[WARN] Không tìm thấy {best}.")
        return None

    # Eval trên val (ảnh đầy đủ, imgsz=960)
    print(f"\n[EVAL VAL] {run_name} (imgsz=960, ảnh đầy đủ) ...")
    r = YOLO(str(best)).val(
        data=str(tiles_yaml), split="val",
        imgsz=960, batch=8,
        conf=0.001, iou=0.6,
        device=args.device, plots=False, verbose=False,
    )
    res = dict(
        map50=round(float(r.box.map50), 4),
        map50_95=round(float(r.box.map), 4),
    )
    for idx, ap50 in zip(r.box.ap_class_index, r.box.ap50):
        res[f"ap50_{r.names[int(idx)]}"] = round(float(ap50), 4)
    for idx, rec in zip(r.box.ap_class_index, r.box.r):
        if r.names[int(idx)] == "motorcycle":
            res["recall_motorcycle"] = round(float(rec), 4)

    # So sánh với baseline
    base_ap = BASELINE_VAL["ap50_motorcycle"]
    cur_ap  = res.get("ap50_motorcycle", 0)
    delta   = cur_ap - base_ap
    sign    = "▲" if delta > 0 else "▼"
    print(f"\n[KẾT QUẢ] {run_name}")
    print(f"  mAP50          : {res['map50']}  (baseline: {BASELINE_VAL['map50']})")
    print(f"  AP50_motorcycle: {cur_ap}  (baseline: {base_ap})  {sign} {abs(delta):.4f}")
    print(f"  Recall_moto    : {res.get('recall_motorcycle','—')}  (baseline: {BASELINE_VAL['recall']})")
    print(f"[BEST] {best}")

    return res


def save_tile_summary(runs_dir: Path, row: dict) -> None:
    path = runs_dir / "motor_tile_summary.csv"
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[CSV] {path}")


def print_tile_comparison(results: dict[str, dict]) -> None:
    all_rows = {"baseline (đã có)": BASELINE_VAL, **results}
    base_ap = BASELINE_VAL["ap50_motorcycle"]
    print(f"\n{'='*72}")
    print(f"{'MODEL':26s}  {'mAP50':7s}  {'AP50_moto':10s}  {'Recall_moto':12s}  {'Δ AP50_moto':11s}")
    print(f"{'-'*72}")
    for name, r in all_rows.items():
        ap = r.get("ap50_motorcycle", r.get("ap50_motor", "—"))
        delta = f"{float(ap)-base_ap:+.4f}" if isinstance(ap, float) and name != "baseline (đã có)" else "   —"
        print(f"{name:26s}  {r.get('map50','—'):7}  {ap:10}  "
              f"{r.get('recall_motorcycle', r.get('recall','—')):12}  {delta}")
    print(f"{'='*72}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--src", type=Path, default=SRC_DEFAULT,
                    help=f"Dataset gốc (default: {SRC_DEFAULT.name})")
    ap.add_argument("--mode", choices=["full", "tile", "mixed"], default="mixed",
                    help="Chế độ dữ liệu train (default: mixed)")
    ap.add_argument("--compare", action="store_true",
                    help="Chạy cả 3 mode tuần tự để so sánh")
    ap.add_argument("--tile",  type=int,   default=640,  help="Kích thước ô (px)")
    ap.add_argument("--overlap", type=float, default=0.25, help="Độ chồng lấn (0–1)")
    ap.add_argument("--min-visible", type=float, default=0.5,
                    help="Tỉ lệ bbox phải nằm trong ô để giữ")
    ap.add_argument("--keep-no-motor", type=float, default=0.2,
                    help="Xác suất giữ ô không có motorcycle (default: 0.2)")
    ap.add_argument("--p2", action="store_true",
                    help="Dùng YOLOv8s-P2 (kết hợp P2 head + tile)")
    ap.add_argument("--pretrained", default="yolov8s.pt")
    ap.add_argument("--epochs",   type=int,   default=None)
    ap.add_argument("--batch",    type=int,   default=None)
    ap.add_argument("--fraction", type=float, default=None)
    ap.add_argument("--seed",     type=int,   default=42)
    ap.add_argument("--device",   default=None)
    ap.add_argument("--no-train", action="store_true",
                    help="Chỉ cắt ảnh, không train")
    ap.add_argument("--dry-run",  action="store_true",
                    help="In kế hoạch, không thực thi")
    args = ap.parse_args()

    src = args.src if args.src.is_absolute() else REPO / args.src
    if not src.exists():
        ap.error(f"Không tìm thấy src: {src}")
    if not (src / "images" / "train").exists():
        ap.error(f"Không tìm thấy {src}/images/train")

    rng = random.Random(args.seed)
    modes = ["full", "tile", "mixed"] if args.compare else [args.mode]
    all_results: dict[str, dict] = {}

    for mode in modes:
        print(f"\n{'#'*65}")
        print(f"# MODE: {mode.upper()}{'  +  P2 HEAD' if args.p2 else ''}")
        print(f"# Data: {src.name}  ({7237} train / {1466} val)")
        print(f"# Baseline (đối chứng): mAP50={BASELINE_VAL['map50']}, "
              f"AP50_moto={BASELINE_VAL['ap50_motorcycle']}")
        print(f"{'#'*65}\n")

        p2_tag = "_p2" if args.p2 else ""
        run_name  = f"motor_tile_{mode}{p2_tag}"
        out_dir   = REPO / "data" / f"{src.name}_tiles_{mode}{p2_tag}"

        if mode == "full":
            # Mode full: dùng thẳng src, tạo yaml trỏ vào đó
            tiles_yaml = out_dir / "tiles_full.yaml"
            out_dir.mkdir(parents=True, exist_ok=True)
            nc, names = 4, {0: "motorcycle", 1: "car", 2: "bus", 3: "truck"}
            yaml_data: dict = dict(
                train=str(src/"images"/"train"),
                val=str(src/"images"/"val"),
                nc=nc, names=names,
            )
            test_dir = src / "images" / "test"
            if test_dir.exists():
                yaml_data["test"] = str(test_dir)
            tiles_yaml.write_text(yaml.dump(yaml_data, allow_unicode=True), encoding="utf-8")
            print(f"[FULL] Dùng ảnh đầy đủ gốc: {src}")
        else:
            if args.dry_run:
                print(f"[DRY-RUN] Sẽ cắt: {src} → {out_dir}")
            else:
                build_tiles(
                    src=src, out=out_dir,
                    tile=args.tile, overlap=args.overlap,
                    min_visible=args.min_visible,
                    keep_no_motor=args.keep_no_motor,
                    no_full=(mode == "tile"),
                    rng=rng,
                )
            tiles_yaml = out_dir / "tiles.yaml"

        if args.no_train:
            print(f"[NO-TRAIN] tiles.yaml: {tiles_yaml}")
            continue

        result = run_training(tiles_yaml, run_name, use_p2=args.p2, args=args)
        if result:
            all_results[run_name] = result
            row = dict(
                mode=mode, arch="p2" if args.p2 else "v8s",
                tile=args.tile, overlap=args.overlap,
                keep_no_motor=args.keep_no_motor,
                run=run_name,
                **result,
                delta_ap50_moto=round(result.get("ap50_motorcycle",0) - BASELINE_VAL["ap50_motorcycle"], 4),
            )
            save_tile_summary(REPO / "runs" / "detect", row)

    if all_results:
        print_tile_comparison(all_results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
