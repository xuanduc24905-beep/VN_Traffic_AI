"""Import UA-DETRAC từ project cũ (yolov8_finetune) sang project mới.

Chiến lược: KHÔNG copy 60GB ảnh — symlink JPG gốc từ raw/DETRAC-Images, chỉ
copy + remap file label .txt (dung lượng nhỏ). Giữ nguyên split train/val/test
đã có sẵn.

Class remap (DETRAC 4-class canonical 4-class):
    0 car 1 car
    1 bus 2 bus
    2 van 3 truck (van gần truck nhất về hình dáng)
    3 others null (bỏ)
Schema canonical 4-class: {0: motorcycle, 1: car, 2: bus, 3: truck} — bỏ bicycle.

Chạy: python scripts/import_ua_detrac.py
"""
import argparse
import json
import re
from pathlib import Path


OLD_LABELS = Path("/home/xuand/yolov8_finetune/data/detrac/labels")
OLD_RAW_IMAGES = Path(
    "/home/xuand/yolov8_finetune/data/raw/ua_detrac/DETRAC-Images/DETRAC-Images"
)
CANONICAL = ["motorcycle", "car", "bus", "truck"]
CANON_ID = {name: i for i, name in enumerate(CANONICAL)}

# Regex tách tên: MVI_20011_img00001.txt ("MVI_20011", "img00001")
NAME_RE = re.compile(r"^(MVI_\d+)_(img\d+)$")


def remap_label_lines(lines: list[str], mapping: dict) -> list[str]:
    out = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        new_name = mapping.get(parts[0])
        if new_name is None:
            continue
        parts[0] = str(CANON_ID[new_name])
        out.append(" ".join(parts))
    return out


def process_split(split: str, out_root: Path, mapping: dict, dry_run: bool):
    src_lbl_dir = OLD_LABELS / split
    if not src_lbl_dir.exists():
        print(f"[!] Không có split {split} ở {src_lbl_dir}")
        return 0, 0

    out_img_dir = out_root / "images" / split
    out_lbl_dir = out_root / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_kept = 0
    n_missing_img = 0

    label_files = sorted(src_lbl_dir.glob("*.txt"))
    for lbl_path in label_files:
        n_total += 1
        m = NAME_RE.match(lbl_path.stem)
        if not m:
            continue
        seq, frame = m.group(1), m.group(2)
        src_jpg = OLD_RAW_IMAGES / seq / f"{frame}.jpg"
        if not src_jpg.exists():
            n_missing_img += 1
            continue

        with open(lbl_path) as f:
            new_lines = remap_label_lines(f.readlines(), mapping)
        if not new_lines:
            continue

        dst_img = out_img_dir / f"detrac_{lbl_path.stem}.jpg"
        dst_lbl = out_lbl_dir / f"detrac_{lbl_path.stem}.txt"

        if not dry_run:
            if dst_img.is_symlink() or dst_img.exists():
                dst_img.unlink()
            dst_img.symlink_to(src_jpg)
            with open(dst_lbl, "w") as f:
                f.write("\n".join(new_lines) + "\n")

        n_kept += 1

        if n_kept % 5000 == 0:
            print(f" ... {split}: đã xử lý {n_kept}/{n_total}")

    print(f"[] {split}: giữ {n_kept}/{n_total} ảnh (thiếu JPG: {n_missing_img})")
    return n_kept, n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/merged")
    ap.add_argument("--mapping", default="configs/class_mapping.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="đếm số ảnh sẽ import mà không tạo symlink")
    args = ap.parse_args()

    with open(args.mapping) as f:
        mapping = json.load(f)["ua_detrac"]

    print("=== IMPORT UA-DETRAC data/merged ===")
    print(f" Old labels : {OLD_LABELS}")
    print(f" Old images : {OLD_RAW_IMAGES}")
    print(f" Output : {args.out}")
    print(f" Mapping : {mapping}")
    print()

    if not OLD_LABELS.exists():
        print(f"[X] Không tìm thấy {OLD_LABELS}")
        return 1
    if not OLD_RAW_IMAGES.exists():
        print(f"[X] Không tìm thấy {OLD_RAW_IMAGES}")
        return 1

    out_root = Path(args.out)
    grand_kept = 0
    for split in ["train", "val", "test"]:
        k, _ = process_split(split, out_root, mapping, args.dry_run)
        grand_kept += k

    print()
    print(f"[] TỔNG: {grand_kept} ảnh đã import (symlink JPG + label remapped)")
    if args.dry_run:
        print("[i] Dry-run — chưa tạo file thật. Bỏ --dry-run để chạy thật.")
    else:
        print(f"[i] Data sẵn ở {out_root}/{{images,labels}}/{{train,val,test}}/")
        print(f"[i] File data/data.yaml đã trỏ về path này rồi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
