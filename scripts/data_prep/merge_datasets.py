"""Gộp nhiều dataset về 1 tập chung 5 lớp và split train/val/test.

Dataset gốc cần đặt sẵn trong data/raw/{ua_detrac,roboflow_vn,visdrone}/ với
cấu trúc chuẩn YOLO (images/ + labels/). Script sẽ:
  1. Đọc class_mapping.json để remap class id gốc id chuẩn 5 lớp.
  2. Copy ảnh + rewrite label file với id mới.
  3. Split ngẫu nhiên 70/15/15 (train/val/test), stratified per-source.

Chạy: python scripts/merge_datasets.py
"""
import argparse
import json
import random
import shutil
from pathlib import Path


CANONICAL = ["motorcycle", "car", "bus", "truck", "bicycle"]
CANON_ID = {name: i for i, name in enumerate(CANONICAL)}


def remap_label_file(src_path: Path, dst_path: Path, mapping: dict):
    """Đọc .txt YOLO, remap class id theo mapping ('0' -> 'car' -> 0 mới)."""
    kept = []
    with open(src_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            old_cls = parts[0]
            new_name = mapping.get(old_cls)
            if new_name is None:
                continue
            new_id = CANON_ID[new_name]
            kept.append(" ".join([str(new_id)] + parts[1:]))
    if not kept:
        return False
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w") as f:
        f.write("\n".join(kept) + "\n")
    return True


def process_source(src_root: Path, source_name: str, out_root: Path,
                   mapping_all: dict, split_ratio=(0.7, 0.15, 0.15), seed=42):
    """Đọc mọi cặp (image, label) trong source, remap và ghi ra out với split."""
    mapping = mapping_all[source_name]
    img_dir = src_root / "images"
    lbl_dir = src_root / "labels"
    if not img_dir.exists():
        print(f"[!] Bỏ qua {source_name}: không có {img_dir}")
        return 0

    pairs = []
    for img_path in sorted(img_dir.rglob("*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl_path = lbl_dir / img_path.relative_to(img_dir).with_suffix(".txt")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))

    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    total_kept = 0
    for split, items in splits.items():
        for i, (img, lbl) in enumerate(items):
            new_stem = f"{source_name}_{split}_{i:06d}"
            out_img = out_root / "images" / split / f"{new_stem}{img.suffix.lower()}"
            out_lbl = out_root / "labels" / split / f"{new_stem}.txt"
            ok = remap_label_file(lbl, out_lbl, mapping)
            if not ok:
                continue
            out_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, out_img)
            total_kept += 1
    print(f"[] {source_name}: giữ {total_kept}/{n} ảnh sau remap.")
    return total_kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/merged")
    ap.add_argument("--mapping", default="configs/class_mapping.json")
    ap.add_argument("--sources", nargs="+",
                    default=["ua_detrac", "roboflow_vn", "visdrone"])
    args = ap.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.mapping) as f:
        mapping_all = json.load(f)

    total = 0
    for src in args.sources:
        src_root = raw / src
        if not src_root.exists():
            print(f"[!] Không tìm thấy {src_root} — bỏ qua.")
            continue
        total += process_source(src_root, src, out, mapping_all)

    print(f"\n[] Tổng số ảnh trong dataset gộp: {total}")
    print(f"[] Output: {out}")
    print(f"[i] Cập nhật data/data.yaml nếu path khác.")


if __name__ == "__main__":
    main()
