"""Import Vietnamese vehicle v3 (Roboflow) → merge vào data/merged/{train}.

Remap class id:
  new 0 car        -> old 1
  new 1 motorcycle -> old 0
  new 2 truck      -> old 3
  new 3 (unused)   -> old 2   (an toàn nếu Roboflow slot bus)

Chỉ merge split `train` để giữ val/test cũ nguyên vẹn cho so sánh mAP.
Prefix `vnv3_` để tránh trùng tên file.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path("/home/xuand/vn_traffic_ai/data")
ZIP = ROOT / "Vietnamese vehicle.v3-2023-02-01-5-31pm.yolov8.zip"
EXTRACT_DIR = ROOT / "raw_vnv3"
MERGED = ROOT / "merged"
PREFIX = "vnv3_"

REMAP = {0: 1, 1: 0, 2: 3, 3: 2}


def extract() -> None:
    if EXTRACT_DIR.exists():
        print(f"[skip extract] {EXTRACT_DIR} đã tồn tại")
        return
    print(f"[extract] {ZIP.name} -> {EXTRACT_DIR}")
    EXTRACT_DIR.mkdir(parents=True)
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(EXTRACT_DIR)


def remap_label(src: Path, dst: Path) -> int:
    """Đọc file label, remap class id, ghi ra dst. Trả về số bbox."""
    lines_out = []
    kept = 0
    for line in src.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        old_cls = int(parts[0])
        if old_cls not in REMAP:
            print(f"  ! class {old_cls} lạ trong {src.name}, skip line")
            continue
        parts[0] = str(REMAP[old_cls])
        lines_out.append(" ".join(parts))
        kept += 1
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
    return kept


def merge_split(split_src: str) -> None:
    """split_src trong bộ mới: 'train' | 'valid' | 'test'. Luôn merge vào 'train' của bộ cũ."""
    img_src = EXTRACT_DIR / split_src / "images"
    lbl_src = EXTRACT_DIR / split_src / "labels"
    img_dst = MERGED / "images" / "train"
    lbl_dst = MERGED / "labels" / "train"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    n_img, n_lbl, n_box = 0, 0, 0
    for img_path in sorted(img_src.iterdir()):
        stem = img_path.stem
        new_name = PREFIX + img_path.name
        shutil.copy2(img_path, img_dst / new_name)
        n_img += 1

        lbl_path = lbl_src / f"{stem}.txt"
        if lbl_path.exists():
            n_box += remap_label(lbl_path, lbl_dst / f"{PREFIX}{stem}.txt")
            n_lbl += 1
    print(f"[merge {split_src}] {n_img} ảnh, {n_lbl} label, {n_box} bbox")


def invalidate_cache() -> None:
    for cache in MERGED.glob("labels/*.cache"):
        print(f"[del cache] {cache}")
        cache.unlink()


def main() -> None:
    extract()
    for split in ("train", "valid", "test"):
        if (EXTRACT_DIR / split).exists():
            merge_split(split)
    invalidate_cache()
    print("Done.")


if __name__ == "__main__":
    main()
