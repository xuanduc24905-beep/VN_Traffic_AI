"""Merge V5 gold train + pseudo accepted -> V6 dataset.

Output structure:
  data/merged_v6/
    images/
      train/   <- V5 gold train + pseudo accepted (symlink để tiết kiệm disk)
      val/     <- V5 val (symlink, KHÔNG đổi)
      test/    <- V5 test (symlink, KHÔNG đổi)
    labels/
      train/   <- idem
      val/     <- idem
      test/    <- idem

Leak check:
  - Pseudo filenames (hcm_*) không được trùng V5 filenames
  - Mặc định không trùng vì V5 dùng SOICT/DETRAC, V6 dùng HCM CCTV

Chạy:
  python scripts/pseudo/08_build_v6_dataset.py              # dry-run
  python scripts/pseudo/08_build_v6_dataset.py --apply      # tạo symlink thật
"""

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

V5_DIR = ROOT / "data" / "merged_v5"
PSEUDO_DIR = ROOT / "data" / "pseudo_v6" / "accepted"
V6_DIR = ROOT / "data" / "merged_v6"


def check_leak(v5_set: set, pseudo_set: set):
    """Kiểm tra pseudo filename có trùng V5 không."""
    overlap = v5_set & pseudo_set
    if overlap:
        raise RuntimeError(
            f"LEAK! {len(overlap)} tên file trùng giữa V5 và pseudo. "
            f"Vài file: {list(overlap)[:5]}"
        )


def gather_stems(folder: Path) -> set:
    """Lấy set tên file (không đuôi) từ 1 folder."""
    if not folder.exists():
        return set()
    return {p.stem for p in folder.glob("*.jpg")}


def main(apply: bool, use_copy: bool):
    # 1. Verify sources
    v5_train_img = V5_DIR / "images" / "train"
    v5_train_lbl = V5_DIR / "labels" / "train"
    v5_val_img   = V5_DIR / "images" / "val"
    v5_test_img  = V5_DIR / "images" / "test"
    pseudo_img   = PSEUDO_DIR / "images"
    pseudo_lbl   = PSEUDO_DIR / "labels"

    for p in [v5_train_img, v5_train_lbl, v5_val_img, v5_test_img, pseudo_img, pseudo_lbl]:
        assert p.exists(), f"Không thấy: {p}"

    n_v5_train  = len(list(v5_train_img.glob("*.jpg")))
    n_v5_val    = len(list(v5_val_img.glob("*.jpg")))
    n_v5_test   = len(list(v5_test_img.glob("*.jpg")))
    n_pseudo    = len(list(pseudo_img.glob("*.jpg")))

    print(f"[source] V5 train : {n_v5_train} ảnh")
    print(f"[source] V5 val   : {n_v5_val} ảnh")
    print(f"[source] V5 test  : {n_v5_test} ảnh")
    print(f"[source] pseudo   : {n_pseudo} ảnh")

    # 2. Leak check
    v5_all = gather_stems(v5_train_img) | gather_stems(v5_val_img) | gather_stems(v5_test_img)
    pseudo_stems = gather_stems(pseudo_img)
    check_leak(v5_all, pseudo_stems)
    print(f"[leak-check] OK — không trùng filename giữa V5 và pseudo")

    # 3. Preview
    n_v6_train = n_v5_train + n_pseudo
    print(f"\n[plan] V6 train = V5 train ({n_v5_train}) + pseudo ({n_pseudo}) = {n_v6_train}")
    print(f"[plan] V6 val   = V5 val   ({n_v5_val})   [không đổi]")
    print(f"[plan] V6 test  = V5 test  ({n_v5_test})  [không đổi]")

    if not apply:
        print("\n(dry-run — chạy với --apply để tạo)")
        return

    # 4. Build V6 structure
    if V6_DIR.exists():
        print(f"\n[clean] xoá {V6_DIR} cũ")
        shutil.rmtree(V6_DIR)

    for split in ["train", "val", "test"]:
        (V6_DIR / "images" / split).mkdir(parents=True)
        (V6_DIR / "labels" / split).mkdir(parents=True)

    link_fn = shutil.copy2 if use_copy else os.symlink

    def link_batch(src_folder, dst_folder, pattern="*.jpg"):
        n = 0
        for src in src_folder.glob(pattern):
            dst = dst_folder / src.name
            if use_copy:
                shutil.copy2(src, dst)
            else:
                # symlink absolute (cần absolute để không depend cwd)
                os.symlink(src.resolve(), dst)
            n += 1
        return n

    print(f"\n[build] {'copy' if use_copy else 'symlink'} files...")

    # train = V5 train + pseudo
    n = link_batch(v5_train_img, V6_DIR / "images" / "train")
    n += link_batch(pseudo_img, V6_DIR / "images" / "train")
    print(f"  train images: {n}")

    n = link_batch(v5_train_lbl, V6_DIR / "labels" / "train", "*.txt")
    n += link_batch(pseudo_lbl, V6_DIR / "labels" / "train", "*.txt")
    print(f"  train labels: {n}")

    # val = V5 val
    n = link_batch(v5_val_img, V6_DIR / "images" / "val")
    n_lbl = link_batch(V5_DIR / "labels" / "val", V6_DIR / "labels" / "val", "*.txt")
    print(f"  val images/labels: {n}/{n_lbl}")

    # test = V5 test
    n = link_batch(v5_test_img, V6_DIR / "images" / "test")
    n_lbl = link_batch(V5_DIR / "labels" / "test", V6_DIR / "labels" / "test", "*.txt")
    print(f"  test images/labels: {n}/{n_lbl}")

    # 5. Write data.yaml
    data_yaml = ROOT / "configs" / "data_v6.yaml"
    data_yaml.write_text(f"""# V6 — fine-tune từ V5, thêm pseudo-label từ CCTV HCM
# Sinh tự động bởi scripts/pseudo/08_build_v6_dataset.py
# Train = V5 gold train ({n_v5_train}) + HCM pseudo ({n_pseudo}) = {n_v6_train}
# Val/Test: giữ nguyên V5 (honest comparison)

path: {V6_DIR}
train: images/train
val:   images/val
test:  images/test

nc: 4
names:
  0: motorcycle
  1: car
  2: bus
  3: truck
""")
    print(f"\n[config] ghi {data_yaml}")
    print(f"\n[DONE] V6 dataset tại: {V6_DIR}")
    print(f"       Config:          {data_yaml}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--copy", action="store_true",
                    help="copy file thay vì symlink (tốn disk, nhưng chắc chắn)")
    args = ap.parse_args()
    main(apply=args.apply, use_copy=args.copy)
