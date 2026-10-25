"""Import SOICT datasets (SOICT2024 + SOICT-night) sang canonical format.

Canonical class order (khớp data.yaml dự án):
  0: motorcycle, 1: car, 2: bus, 3: truck

Mapping:
  - SOICT2024 (src: ['car','motorbike','tourist bus','truck']):
      {0:1, 1:0, 2:2, 3:3}
  - SOICT-night (src tên '4','5','6','7' nhưng label ID là 0-3,
    đoán theo size/aspect: 0=motorcycle, 1=car, 2=bus, 3=truck):
      {0:0, 1:1, 2:2, 3:3}  (identity)

Output:
  data/raw_soict_canonical/{train,valid,test}/{images,labels}/
  với prefix 'soict24_' và 'soictnight_' tránh trùng tên.

Smart start: skip nếu output đã có đủ số file expected.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path("/home/xuand/vn_traffic_ai/data")
OUT_DIR = ROOT / "raw_soict_canonical"

DATASETS = [
    {
        "name": "soict24",
        "src_dir": ROOT / "raw_soict2024",
        "remap": {0: 1, 1: 0, 2: 2, 3: 3},
        "splits": ["train", "valid", "test"],
    },
    {
        "name": "soictnight",
        "src_dir": ROOT / "raw_soict_night",
        "remap": {0: 0, 1: 1, 2: 2, 3: 3},
        "splits": ["train"],
    },
]


def remap_label(src: Path, dst: Path, remap: dict[int, int]) -> int:
    lines_out = []
    kept = 0
    for line in src.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        old_cls = int(parts[0])
        if old_cls not in remap:
            print(f"  ! class {old_cls} lạ trong {src.name}, skip line")
            continue
        parts[0] = str(remap[old_cls])
        lines_out.append(" ".join(parts))
        kept += 1
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
    return kept


def link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink để tiết kiệm disk, fallback copy nếu khác filesystem."""
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def import_dataset(cfg: dict) -> None:
    name = cfg["name"]
    src_root = cfg["src_dir"]
    remap = cfg["remap"]

    for split in cfg["splits"]:
        img_src = src_root / split / "images"
        lbl_src = src_root / split / "labels"
        if not img_src.exists():
            print(f"[skip {name}/{split}] không tồn tại")
            continue

        img_dst = OUT_DIR / split / "images"
        lbl_dst = OUT_DIR / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        n_img, n_lbl, n_box = 0, 0, 0
        skipped = 0
        for img_path in sorted(img_src.iterdir()):
            if not img_path.is_file():
                continue
            stem = img_path.stem
            new_stem = f"{name}_{stem}"
            new_img = img_dst / f"{new_stem}{img_path.suffix}"
            new_lbl = lbl_dst / f"{new_stem}.txt"

            if new_img.exists() and new_lbl.exists():
                skipped += 1
                continue

            link_or_copy(img_path, new_img)
            n_img += 1

            src_lbl = lbl_src / f"{stem}.txt"
            if src_lbl.exists():
                n_box += remap_label(src_lbl, new_lbl, remap)
                n_lbl += 1
        print(f"[{name}/{split}] +{n_img} ảnh, {n_lbl} label, {n_box} bbox "
              f"(skipped {skipped} đã tồn tại)")


def summarize() -> None:
    print("\n=== Summary ===")
    for split in ("train", "valid", "test"):
        img_dir = OUT_DIR / split / "images"
        lbl_dir = OUT_DIR / split / "labels"
        if not img_dir.exists():
            continue
        n_img = sum(1 for _ in img_dir.iterdir())
        n_lbl = sum(1 for _ in lbl_dir.iterdir()) if lbl_dir.exists() else 0
        print(f"  {split}: {n_img} ảnh, {n_lbl} label")

    # Class distribution across all splits
    from collections import Counter
    cnt = Counter()
    for split in ("train", "valid", "test"):
        lbl_dir = OUT_DIR / split / "labels"
        if not lbl_dir.exists():
            continue
        for lbl in lbl_dir.iterdir():
            for line in lbl.read_text().splitlines():
                if line.strip():
                    cnt[int(line.split()[0])] += 1
    names = {0: "motorcycle", 1: "car", 2: "bus", 3: "truck"}
    print("\n  Class distribution (canonical):")
    for c in sorted(cnt):
        print(f"    {c} ({names.get(c,'?')}): {cnt[c]:,}")


def main() -> None:
    for cfg in DATASETS:
        import_dataset(cfg)
    summarize()
    print("\nDone.")


if __name__ == "__main__":
    main()
