"""Train + đánh giá các biến thể V5+clean theo MỘT giao thức cố định (chỉ đổi dữ liệu).

Vì sao cần script này: so sánh cũ V5 (0.639) → V5+ warm (0.747 mAP50) KHÔNG quy được cho dữ
liệu v10, vì 2 run khác nhau đồng thời ở: imgsz 640→960, tổng epoch 40→80 (warm-start),
augmentation (mixup/copy_paste/degrees), lr0 0.01→0.001 + cos_lr. Ở đây mọi biến thể dùng:
  • cùng init: COCO yolov8s.pt (không warm-start từ checkpoint chọn bằng val có leak)
  • cùng recipe, cùng số epoch, KHÔNG early-stop (patience = epochs) → cùng ngân sách epoch
  • chọn best.pt bằng val sạch (soict24_cam_07); test V5 chạy MỘT lần sau khi train xong
  → chênh lệch giữa biến thể chủ yếu do dữ liệu (cộng nhiễu seed — xem --seeds).

Biến thể (build bởi scripts/data_prep/build_v5plus_clean.py):
  v5base     V5 train                       (baseline)
  full       V5 train + v10 đã lọc          (bản chính)
  nooverlay  V5 train + v10 không overlay   (ablation)
  ovmatched  V5 train + v10 có overlay, cùng cỡ & camera với nooverlay (đối chứng)

Usage (bạn tự chạy):
    conda activate yolov8_ft
    python scripts/training/v5plus_clean_train.py --variant v5base --seed 42
    python scripts/training/v5plus_clean_train.py --variant full   --seed 42
    python scripts/training/v5plus_clean_train.py --variant nooverlay --seed 42
    python scripts/training/v5plus_clean_train.py --variant ovmatched --seed 42
    python scripts/training/v5plus_clean_train.py --summarize          # bảng kết quả (mean ± std nếu nhiều seed)

Chạy lại cùng lệnh sẽ RESUME nếu run dở, SKIP train nếu đã xong (vẫn đánh giá test nếu chưa có).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
RESULTS_DIR = PROJECT_ROOT / "reports" / "v5plus_clean" / "results"
COCO_INIT = PROJECT_ROOT / "yolov8s.pt"
NAMES = ["motorcycle", "car", "bus", "truck"]

YAMLS = {
    "v5base": PROJECT_ROOT / "data" / "data_v5_cleanval.yaml",
    "full": PROJECT_ROOT / "data" / "data_v5plus_clean.yaml",
    "nooverlay": PROJECT_ROOT / "data" / "data_v5plus_clean_nooverlay.yaml",
    "ovmatched": PROJECT_ROOT / "data" / "data_v5plus_clean_ovmatched.yaml",
}

# Recipe chung — KHÔNG chỉnh riêng cho biến thể nào. Đổi recipe = đổi cho tất cả & train lại tất cả.
RECIPE = dict(
    epochs=60,
    imgsz=960,
    batch=16,
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    cos_lr=True,
    close_mosaic=10,
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.3,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=5.0, translate=0.1, scale=0.5,
    fliplr=0.5, flipud=0.0,
    amp=True,
    cache="disk",
    workers=8,
    deterministic=True,
    save_period=10,
    exist_ok=True,
    verbose=True,
)


def run_name(variant: str, seed: int) -> str:
    return f"train_v8s_v5pc_{variant}_s{seed}"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def train(variant: str, seed: int) -> Path:
    import pandas as pd
    from ultralytics import YOLO

    data = YAMLS[variant]
    assert data.exists(), f"Thiếu {data}. Chạy scripts/data_prep/build_v5plus_clean.py trước."
    name = run_name(variant, seed)
    run_dir = RUNS_DIR / name
    last_pt, best_pt, csv = run_dir / "weights" / "last.pt", run_dir / "weights" / "best.pt", run_dir / "results.csv"

    done = 0
    if csv.exists():
        try:
            done = len(pd.read_csv(csv))
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] {e}")
    if done >= RECIPE["epochs"] and best_pt.exists():
        print(f"[SKIP] {name} đã train xong ({done} epoch)")
        return best_pt
    if last_pt.exists() and done > 0:
        print(f"[RESUME] {name} từ epoch {done}/{RECIPE['epochs']}")
        YOLO(str(last_pt)).train(resume=True)
        return best_pt

    assert COCO_INIT.exists(), f"Thiếu {COCO_INIT}"
    print(f"[FRESH] {name}: init {COCO_INIT.name}, data {data.name}")
    YOLO(str(COCO_INIT)).train(data=str(data), seed=seed, patience=RECIPE["epochs"],
                               project=str(RUNS_DIR), name=name, **RECIPE)
    return best_pt


def evaluate_test(variant: str, seed: int, best_pt: Path) -> dict:
    """Đánh giá test V5 — gọi MỘT lần cho mỗi run sau khi train xong."""
    from ultralytics import YOLO

    out = RESULTS_DIR / f"{run_name(variant, seed)}.json"
    if out.exists():
        print(f"[SKIP] Đã có kết quả test: {out.relative_to(PROJECT_ROOT)}")
        return json.loads(out.read_text())
    data = YAMLS[variant]
    m = YOLO(str(best_pt)).val(data=str(data), split="test", imgsz=RECIPE["imgsz"], batch=RECIPE["batch"],
                               conf=0.001, iou=0.7, plots=False,
                               project=str(RUNS_DIR), name=f"{run_name(variant, seed)}_test", exist_ok=True)
    ap50 = {NAMES[c]: float(m.box.ap50[i]) for i, c in enumerate(m.box.ap_class_index)}
    manifest = PROJECT_ROOT / "reports" / "v5plus_clean" / f"manifest_{variant}.json"
    res = {
        "variant": variant, "seed": seed, "run": run_name(variant, seed),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"), "git_head": _git_head(),
        "weights": str(best_pt.relative_to(PROJECT_ROOT)), "data_yaml": str(data.relative_to(PROJECT_ROOT)),
        "dataset_manifest_sha256_16": _sha(manifest) if manifest.exists() else None,
        "recipe": RECIPE, "init": COCO_INIT.name,
        "test": {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map), "AP50": ap50,
                 "precision": float(m.box.mp), "recall": float(m.box.mr)},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"[TEST] {res['run']}: mAP50={res['test']['mAP50']:.4f} mAP50-95={res['test']['mAP50_95']:.4f}")
    return res


def summarize() -> int:
    import statistics as st
    rows: dict[str, list[dict]] = {}
    for p in sorted(RESULTS_DIR.glob("*.json")):
        r = json.loads(p.read_text())
        rows.setdefault(r["variant"], []).append(r)
    if not rows:
        print("Chưa có kết quả.")
        return 1
    recipes = {json.dumps(r["recipe"], sort_keys=True) for rs in rows.values() for r in rs}
    if len(recipes) > 1:
        print("[CẢNH BÁO] Các run dùng recipe KHÁC nhau → không so sánh trực tiếp được.")

    def fmt(xs):
        return f"{xs[0]:.4f}" if len(xs) == 1 else f"{st.mean(xs):.4f} ± {st.stdev(xs):.4f}"
    lines = ["| Biến thể | n seed | mAP50 | mAP50-95 | " + " | ".join(f"AP50 {n}" for n in NAMES) + " |",
             "|---|---|---|---|" + "---|" * len(NAMES)]
    for v in YAMLS:
        rs = rows.get(v)
        if not rs:
            continue
        lines.append(f"| `{v}` | {len(rs)} | {fmt([r['test']['mAP50'] for r in rs])} | "
                     f"{fmt([r['test']['mAP50_95'] for r in rs])} | "
                     + " | ".join(fmt([r['test']['AP50'].get(n, 0.0) for r in rs]) for n in NAMES) + " |")
    table = "\n".join(lines)
    print(table)
    out = RESULTS_DIR.parent / "results_summary.md"
    out.write_text("# Kết quả V5+clean trên test V5 (cùng recipe, chỉ đổi dữ liệu)\n\n"
                   "_Sinh bởi scripts/training/v5plus_clean_train.py --summarize. "
                   "Chênh lệch < ~1 điểm với 1 seed chưa đủ để kết luận._\n\n" + table + "\n")
    print(f"[OK] {out.relative_to(PROJECT_ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(YAMLS))
    ap.add_argument("--seed", type=int, nargs="+", default=[42], help="Một hoặc nhiều seed, vd --seed 42 0 1")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    if not a.variant:
        ap.error("cần --variant hoặc --summarize")
    for seed in a.seed:
        best = train(a.variant, seed)
        evaluate_test(a.variant, seed, best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
