"""Import Vietnam Cần Thơ vehicle dataset (Roboflow yolov8 format) merged/.

Class remap (VN Cần Thơ 4-class canonical 4-class):
    0 bus 2 bus
    1 car 1 car
    2 motorbike 0 motorcycle (đây là giá trị chính — bù thiếu motorcycle của UA-DETRAC)
    3 truck 3 truck

Split mapping: Roboflow dùng "valid" thay "val" chuyển về "val" cho khớp UA-DETRAC.

File output có prefix `cantho_` để phân biệt với DETRAC (không đụng nhau).

Chạy: python scripts/import_cantho_vn.py
"""
import argparse
import json
import shutil
from pathlib import Path


SRC_ROOT = Path("data/raw_vn_cantho")
CANONICAL = ["motorcycle", "car", "bus", "truck"]
CANON_ID = {name: i for i, name in enumerate(CANONICAL)}

# Roboflow đôi khi dùng valid, đôi khi val
SPLIT_MAP = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def remap_label_lines(lines, mapping):
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


def process_split(src_split_dir, dst_split, out_root, mapping, dry_run):
    src_img_dir = src_split_dir / "images"
    src_lbl_dir = src_split_dir / "labels"

    if not src_img_dir.exists() or not src_lbl_dir.exists():
        print(f"[!] Bỏ qua {src_split_dir} — thiếu images/ hoặc labels/")
        return 0

    out_img_dir = out_root / "images" / dst_split
    out_lbl_dir = out_root / "labels" / dst_split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for lbl_path in sorted(src_lbl_dir.glob("*.txt")):
        stem = lbl_path.stem
        # Tìm ảnh (jpg, jpeg, png)
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            cand = src_img_dir / f"{stem}{ext}"
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue

        with open(lbl_path) as f:
            new_lines = remap_label_lines(f.readlines(), mapping)
        if not new_lines:
            continue

        dst_img = out_img_dir / f"cantho_{stem}{img_path.suffix}"
        dst_lbl = out_lbl_dir / f"cantho_{stem}.txt"

        if not dry_run:
            if dst_img.is_symlink() or dst_img.exists():
                dst_img.unlink()
            dst_img.symlink_to(img_path.resolve())
            with open(dst_lbl, "w") as f:
                f.write("\n".join(new_lines) + "\n")
        n += 1

    print(f"[] {src_split_dir.name} {dst_split}: {n} ảnh")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC_ROOT),
                    help="Thư mục extract của zip Cần Thơ (có train/valid/test)")
    ap.add_argument("--out", default="data/merged")
    ap.add_argument("--mapping", default="configs/class_mapping.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"[X] Không tìm thấy {src}")
        return 1

    with open(args.mapping) as f:
        mapping = json.load(f)["cantho_vn"]

    print("=== IMPORT CanTho VN data/merged ===")
    print(f" Source : {src}")
    print(f" Output : {args.out}")
    print(f" Mapping : {mapping}")
    print()

    out_root = Path(args.out)
    total = 0
    for src_split in ["train", "valid", "test"]:
        src_dir = src / src_split
        if not src_dir.exists():
            print(f"[!] Không có {src_dir}")
            continue
        total += process_split(src_dir, SPLIT_MAP[src_split], out_root,
                               mapping, args.dry_run)

    print()
    print(f"[] TỔNG: {total} ảnh Cần Thơ đã import")
    if args.dry_run:
        print("[i] Dry-run — chưa tạo file thật.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
