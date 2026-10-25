"""Build data/merged_v4/ — V3 + thêm vn_cctv_traffic (CCTV Đà Nẵng, 29k ảnh).

Composition:
  train = merged_v3/train (20.5k) + vn_cctv/train (26.3k) + vn_cctv/test (2.9k)
        ≈ 49.7k ảnh (hơn V3 2.4×)
  val   = merged_v3/val (1.8k) — giữ nguyên để so sánh V3 vs V4
  test  = merged_v3/test (1.1k) — giữ nguyên

Class canonical: {0: motorcycle, 1: car, 2: bus, 3: truck}
vn_cctv src classes: ['bus', 'car', 'motorbike', 'truck']
  → remap {0:2, 1:1, 2:0, 3:3}

Dùng hardlink. Smart start: skip nếu đã có.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path("/home/xuand/vn_traffic_ai/data")
MERGED_V3 = ROOT / "merged_v3"
VN_CCTV = ROOT / "raw_vn_cctv"
OUT = ROOT / "merged_v4"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VN_CCTV_REMAP = {0: 2, 1: 1, 2: 0, 3: 3}


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def ensure_dirs() -> None:
    for sp in ("train", "val", "test"):
        (OUT / "images" / sp).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)


def link_file(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def remap_label(src: Path, dst: Path, remap: dict[int, int]) -> None:
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
            continue
        parts[0] = str(remap[old])
        lines_out.append(" ".join(parts))
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))


def copy_from_merged_v3() -> dict[str, int]:
    """Hardlink toàn bộ merged_v3 vào merged_v4."""
    counts = {"train": 0, "val": 0, "test": 0}
    for split in ("train", "val", "test"):
        img_dir = MERGED_V3 / "images" / split
        lbl_dir = MERGED_V3 / "labels" / split
        for img in img_dir.iterdir():
            if not is_image(img):
                continue
            dst_img = OUT / "images" / split / img.name
            dst_lbl = OUT / "labels" / split / f"{img.stem}.txt"
            link_file(img, dst_img)
            src_lbl = lbl_dir / f"{img.stem}.txt"
            if src_lbl.exists():
                link_file(src_lbl, dst_lbl)
            counts[split] += 1
    print(f"[v3 → v4] train={counts['train']}, val={counts['val']}, test={counts['test']}")
    return counts


def copy_vn_cctv() -> int:
    """vn_cctv train + test → merged_v4/train với remap + prefix."""
    prefix = "vncctv_"
    img_dst = OUT / "images" / "train"
    lbl_dst = OUT / "labels" / "train"
    n = 0
    for src_split in ("train", "test"):
        img_dir = VN_CCTV / src_split / "images"
        lbl_dir = VN_CCTV / src_split / "labels"
        if not img_dir.exists():
            continue
        split_n = 0
        for img in img_dir.iterdir():
            if not is_image(img):
                continue
            new_name = f"{prefix}{src_split}_{img.name}"
            new_stem = Path(new_name).stem
            dst_img = img_dst / new_name
            dst_lbl = lbl_dst / f"{new_stem}.txt"
            if dst_img.exists() and dst_lbl.exists():
                continue
            link_file(img, dst_img)
            src_lbl = lbl_dir / f"{img.stem}.txt"
            if src_lbl.exists():
                remap_label(src_lbl, dst_lbl, VN_CCTV_REMAP)
            split_n += 1
        print(f"[vn_cctv/{src_split} → v4/train] +{split_n} ảnh")
        n += split_n
    return n


def write_yaml() -> None:
    yaml_path = ROOT / "data_v4.yaml"
    content = f"""# Ultralytics YOLO config — V4 (V3 + vn_cctv Đà Nẵng CCTV)
# train: V3 (20.5k) + vn_cctv (29k) ≈ 49.5k
# val/test: giữ nguyên V3 để so sánh fair V3 vs V4
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
    print("\n=== merged_v4 summary ===")
    names = {0: "motorcycle", 1: "car", 2: "bus", 3: "truck"}
    for split in ("train", "val", "test"):
        img_dir = OUT / "images" / split
        lbl_dir = OUT / "labels" / split
        n_img = sum(1 for p in img_dir.iterdir() if is_image(p))
        src_cnt = Counter()
        for p in img_dir.iterdir():
            if not is_image(p):
                continue
            matched = False
            for pref in ("soict24_", "soictnight_", "vnv3_", "cantho_", "vncctv_"):
                if p.name.startswith(pref):
                    src_cnt[pref.rstrip("_")] += 1
                    matched = True
                    break
            if not matched:
                src_cnt["other"] += 1
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
    copy_from_merged_v3()
    copy_vn_cctv()
    invalidate_cache()
    write_yaml()
    summarize()
    print("\nDone.")


if __name__ == "__main__":
    main()
