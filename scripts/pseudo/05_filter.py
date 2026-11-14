"""Filter pseudo-labels V6 theo per-class threshold + size filter.

Logic lọc:
  - Per-class confidence threshold (motorcycle cao vì class khó)
  - Loại box diện tích quá nhỏ (<20x20 px sau upscale)
  - Loại ảnh không còn box sau filter
  - Copy ảnh + label qua data/pseudo_v6/accepted/

Chạy:
  python scripts/pseudo/05_filter.py
"""

import argparse
import shutil
from pathlib import Path
from PIL import Image

# ─── Config ────────────────────────────────────────────────
PREDICT_DIR = Path("data/pseudo_v6/round1/predict")
LABELS_IN   = PREDICT_DIR / "labels"
FRAMES_DIR  = Path("data/raw/frames/hcm")
OUT_DIR     = Path("data/pseudo_v6/accepted")

# Per-class confidence threshold (class_id, name, threshold)
CLASS_CONFIG = {
    0: ("motorcycle", 0.60),   # class khó nhất -> threshold cao
    1: ("car",        0.45),
    2: ("bus",        0.50),
    3: ("truck",      0.50),
}

# Size filter: box area tối thiểu (fraction of image, vì imgsz không cố định)
# Ảnh 512x288 -> 20x20 px = 400 / (512*288) = 0.0027
MIN_BOX_AREA_FRAC = 0.0025

# Cấu hình ảnh: tối thiểu 1 box qualified -> giữ ảnh
MIN_BOXES_PER_IMAGE = 1


def read_label(path: Path) -> list[tuple[int, float, float, float, float, float]]:
    """Trả list [(cls, xc, yc, w, h, conf)]"""
    boxes = []
    for line in path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 6:
            continue
        cls = int(parts[0])
        xc, yc, w, h, conf = map(float, parts[1:])
        boxes.append((cls, xc, yc, w, h, conf))
    return boxes


def filter_boxes(boxes, frames_lookup: dict[str, Path], label_path: Path):
    """Áp dụng per-class threshold + size filter."""
    kept, rejected_reasons = [], {"conf": 0, "size": 0, "unknown_class": 0}
    for cls, xc, yc, w, h, conf in boxes:
        if cls not in CLASS_CONFIG:
            rejected_reasons["unknown_class"] += 1
            continue
        name, thr = CLASS_CONFIG[cls]
        if conf < thr:
            rejected_reasons["conf"] += 1
            continue
        if w * h < MIN_BOX_AREA_FRAC:
            rejected_reasons["size"] += 1
            continue
        kept.append((cls, xc, yc, w, h, conf))
    return kept, rejected_reasons


def find_frame_path(label_name: str, frames_lookup: dict[str, Path]) -> Path | None:
    """Tìm ảnh gốc từ tên label (.txt -> .jpg trong recursive glob)."""
    jpg_name = label_name.replace(".txt", ".jpg")
    return frames_lookup.get(jpg_name)


def main(dry_run: bool):
    assert LABELS_IN.exists(), f"Không thấy labels: {LABELS_IN}"
    assert FRAMES_DIR.exists(), f"Không thấy frames: {FRAMES_DIR}"

    # Lookup: filename -> full path (vì ảnh nằm trong subfolder theo ward)
    frames_lookup = {p.name: p for p in FRAMES_DIR.rglob("*.jpg")}
    print(f"[load] {len(frames_lookup)} ảnh frames, {sum(1 for _ in LABELS_IN.glob('*.txt'))} label files")

    # Stats
    total_imgs_in = 0
    total_imgs_kept = 0
    total_boxes_in = 0
    total_boxes_kept = 0
    rejected_all = {"conf": 0, "size": 0, "unknown_class": 0}
    class_kept = {cls: 0 for cls in CLASS_CONFIG}

    if not dry_run:
        (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)

    for label_path in sorted(LABELS_IN.glob("*.txt")):
        total_imgs_in += 1
        boxes = read_label(label_path)
        total_boxes_in += len(boxes)

        kept, rej = filter_boxes(boxes, frames_lookup, label_path)
        for k, v in rej.items():
            rejected_all[k] += v

        if len(kept) < MIN_BOXES_PER_IMAGE:
            continue

        # Image qualifies
        total_imgs_kept += 1
        total_boxes_kept += len(kept)
        for cls, *_ in kept:
            class_kept[cls] += 1

        if dry_run:
            continue

        # Copy image + write filtered label
        src_img = find_frame_path(label_path.name, frames_lookup)
        if src_img is None:
            continue

        dst_img = OUT_DIR / "images" / src_img.name
        dst_lbl = OUT_DIR / "labels" / label_path.name

        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # Ghi label KHÔNG có confidence (đúng YOLO training format)
        with open(dst_lbl, "w") as f:
            for cls, xc, yc, w, h, _conf in kept:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    # Report
    mode = "[DRY-RUN]" if dry_run else "[APPLIED]"
    print(f"\n{'='*55}")
    print(f"{mode} Filter pseudo-labels round 1")
    print(f"{'='*55}")
    print(f"  Ảnh in → kept  : {total_imgs_in} → {total_imgs_kept} "
          f"({100*total_imgs_kept/max(1,total_imgs_in):.1f}%)")
    print(f"  Box in → kept  : {total_boxes_in} → {total_boxes_kept} "
          f"({100*total_boxes_kept/max(1,total_boxes_in):.1f}%)")
    print(f"\n  Box bị loại:")
    print(f"    confidence  : {rejected_all['conf']}")
    print(f"    quá nhỏ     : {rejected_all['size']}")
    print(f"    class lạ    : {rejected_all['unknown_class']}")
    print(f"\n  Phân bố class giữ lại:")
    for cls, (name, thr) in CLASS_CONFIG.items():
        print(f"    {name:12s} (thr={thr}): {class_kept[cls]}")

    if not dry_run:
        print(f"\n  Output: {OUT_DIR}/")
        print(f"    images/ : {len(list((OUT_DIR / 'images').glob('*.jpg')))} ảnh")
        print(f"    labels/ : {len(list((OUT_DIR / 'labels').glob('*.txt')))} label")
    else:
        print("\n  (dry-run — chạy với --apply để copy file thật)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    main(dry_run=not args.apply)
