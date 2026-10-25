"""Build data/merged_v3/ — VN-only (bỏ UA-DETRAC để tránh TQ lấn át lớp motorcycle).

Composition:
  train = SOICT2024/train (4,804) + SOICT-night/train (13,834)
        + VNv3/train (1,235) + Cantho/train (678)                 ≈ 20.5k
  val   = SOICT2024/valid (1,407) + VNv3/valid (156) + Cantho/valid (216)  ≈ 1.8k
  test  = SOICT2024/test (681) + VNv3/test (156) + Cantho/test (216)       ≈ 1.0k

Class canonical: {0: motorcycle, 1: car, 2: bus, 3: truck}
Remap:
  - raw_soict_canonical: đã canonical sẵn (từ import_soict.py), copy nguyên.
  - raw_vnv3 (src: 0=car, 1=motorcycle, 2=truck, 3=bus): {0:1, 1:0, 2:3, 3:2}
  - raw_vn_cantho (src: 0=bus, 1=car, 2=motorcycle, 3=truck): {0:2, 1:1, 2:0, 3:3}

Hardlink zero-copy, lọc .jpg (bỏ .npy cache). Smart start: skip nếu đã có.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path("/home/xuand/vn_traffic_ai/data")
OUT = ROOT / "merged_v3"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SOURCES = [
    {
        "name": "soict",
        "src_root": ROOT / "raw_soict_canonical",
        "prefix": "",            # file đã có prefix soict24_ / soictnight_
        "remap": None,            # đã canonical
        "split_map": {"train": "train", "valid": "val", "test": "test"},
    },
    {
        "name": "vnv3",
        "src_root": ROOT / "raw_vnv3",
        "prefix": "vnv3_",
        "remap": {0: 1, 1: 0, 2: 3, 3: 2},
        "split_map": {"train": "train", "valid": "val", "test": "test"},
    },
    {
        "name": "cantho",
        "src_root": ROOT / "raw_vn_cantho",
        "prefix": "cantho_",
        "remap": {0: 2, 1: 1, 2: 0, 3: 3},
        "split_map": {"train": "train", "valid": "val", "test": "test"},
    },
]


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def ensure_dirs() -> None:
    for sp in ("train", "val", "test"):
        (OUT / "images" / sp).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)


def remap_label(src: Path, dst: Path, remap: dict[int, int] | None) -> None:
    if remap is None:
        if dst.exists():
            return
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
        return
    if dst.exists():
        return
    lines_out = []
    for line in src.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        old = int(parts[0])
        if old not in remap:
            print(f"  ! class {old} lạ trong {src.name}, skip line")
            continue
        parts[0] = str(remap[old])
        lines_out.append(" ".join(parts))
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))


def link_image(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def import_source(cfg: dict) -> dict[str, int]:
    name = cfg["name"]
    counts = {}
    for src_split, dst_split in cfg["split_map"].items():
        img_dir = cfg["src_root"] / src_split / "images"
        lbl_dir = cfg["src_root"] / src_split / "labels"
        if not img_dir.exists():
            continue
        img_dst = OUT / "images" / dst_split
        lbl_dst = OUT / "labels" / dst_split
        n = 0
        for img in img_dir.iterdir():
            if not is_image(img):
                continue
            new_name = f"{cfg['prefix']}{img.name}"
            new_stem = Path(new_name).stem
            dst_img = img_dst / new_name
            dst_lbl = lbl_dst / f"{new_stem}.txt"
            src_lbl = lbl_dir / f"{img.stem}.txt"

            if dst_img.exists() and dst_lbl.exists():
                continue
            link_image(img, dst_img)
            if src_lbl.exists():
                remap_label(src_lbl, dst_lbl, cfg["remap"])
            n += 1
        print(f"  [{name}] {src_split} → {dst_split}: +{n} ảnh")
        counts[dst_split] = counts.get(dst_split, 0) + n
    return counts


def write_yaml() -> None:
    yaml_path = ROOT / "data_v3.yaml"
    content = f"""# Ultralytics YOLO config — vòng 3 (VN-only, no DETRAC)
# train: SOICT (18.6k) + VNv3/train (1.2k) + Cantho/train (0.7k) ≈ 20.5k
# val:   SOICT/valid (1.4k) + VNv3/valid (156) + Cantho/valid (216) ≈ 1.8k
# test:  SOICT/test (0.7k) + VNv3/test (156) + Cantho/test (216) ≈ 1.0k
# Schema: {{motorcycle, car, bus, truck}}

path: {OUT}
train: images/train
val:   images/val
test:  images/test

nc: 4
names:
  0: motorcycle
  1: car
  2: bus
  3: truck
"""
    yaml_path.write_text(content)
    print(f"[yaml] {yaml_path}")


def invalidate_cache() -> None:
    for cache in OUT.glob("labels/*.cache"):
        print(f"[del cache] {cache}")
        cache.unlink()


def summarize() -> None:
    from collections import Counter
    print("\n=== merged_v3 summary (VN-only) ===")
    names = {0: "motorcycle", 1: "car", 2: "bus", 3: "truck"}
    for split in ("train", "val", "test"):
        img_dir = OUT / "images" / split
        lbl_dir = OUT / "labels" / split
        n_img = sum(1 for p in img_dir.iterdir() if is_image(p))
        src_cnt = Counter()
        for p in img_dir.iterdir():
            if not is_image(p):
                continue
            for pref in ("soict24_", "soictnight_", "vnv3_", "cantho_"):
                if p.name.startswith(pref):
                    src_cnt[pref.rstrip("_")] += 1
                    break
        cnt = Counter()
        for lbl in lbl_dir.iterdir():
            if lbl.suffix != ".txt":
                continue
            for line in lbl.read_text().splitlines():
                if line.strip():
                    cnt[int(line.split()[0])] += 1
        print(f"\n  {split}: {n_img:,} ảnh")
        print(f"    nguồn: {dict(src_cnt)}")
        print(f"    bbox:  {sum(cnt.values()):,}")
        for c in sorted(cnt):
            print(f"      {c} ({names.get(c,'?')}): {cnt[c]:,}")


def main() -> None:
    ensure_dirs()
    for cfg in SOURCES:
        print(f"\n[source] {cfg['name']}")
        import_source(cfg)
    invalidate_cache()
    write_yaml()
    summarize()
    print("\nDone.")


if __name__ == "__main__":
    main()
