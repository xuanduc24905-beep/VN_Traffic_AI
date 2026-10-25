"""Build V6 dataset = V5 train (clean VN) + pseudo-labeled vn_cctv.

Val/test giữ nguyên V5 (unseen videos SOICT+Cantho+vnv3) → honest signal.

Output: data/merged_v6/
  images/{train,val,test}/
  labels/{train,val,test}/
  split_manifest.json
+ data/data_v6.yaml
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path("/home/xuand/vn_traffic_ai").resolve()
V5_ROOT = PROJECT_ROOT / "data" / "merged_v5"
PSEUDO_ROOT = PROJECT_ROOT / "data" / "pseudo_vn_cctv"
DST = PROJECT_ROOT / "data" / "merged_v6"
YAML_OUT = PROJECT_ROOT / "data" / "data_v6.yaml"


def symlink_force(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main() -> int:
    assert V5_ROOT.exists(), f"Thiếu {V5_ROOT}"
    assert PSEUDO_ROOT.exists(), f"Thiếu {PSEUDO_ROOT}. Chạy pseudo_label_vn_cctv.py trước."

    # Dọn
    if DST.exists():
        shutil.rmtree(DST)
    for split in ("train", "val", "test"):
        (DST / "images" / split).mkdir(parents=True)
        (DST / "labels" / split).mkdir(parents=True)

    counts = {}

    # 1. Copy toàn bộ V5 (symlinks sang symlinks → thẳng tới source)
    print("[1/3] Clone V5 train/val/test...")
    for split in ("train", "val", "test"):
        n = 0
        for p in (V5_ROOT / "images" / split).iterdir():
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img_src = p.resolve()  # follow V5 symlink về source thật
            lbl_src = (V5_ROOT / "labels" / split / (p.stem + ".txt")).resolve()
            if not lbl_src.exists():
                continue
            symlink_force(img_src, DST / "images" / split / p.name)
            symlink_force(lbl_src, DST / "labels" / split / (p.stem + ".txt"))
            n += 1
        counts[f"v5_{split}"] = n
        print(f"      V5 {split}: {n} frames")

    # 2. Add pseudo-labeled vn_cctv vào TRAIN (không add vào val/test)
    print("\n[2/3] Add pseudo-labeled vn_cctv vào train...")
    n = 0
    for p in (PSEUDO_ROOT / "images").iterdir():
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img_src = p.resolve()
        lbl_src = (PSEUDO_ROOT / "labels" / (p.stem + ".txt"))
        if not lbl_src.exists():
            continue
        symlink_force(img_src, DST / "images" / "train" / p.name)
        symlink_force(lbl_src.resolve(), DST / "labels" / "train" / (p.stem + ".txt"))
        n += 1
    counts["pseudo_vn_cctv"] = n
    print(f"      Pseudo vn_cctv added: {n} frames")

    total_train = counts["v5_train"] + counts["pseudo_vn_cctv"]
    pseudo_pct = 100 * counts["pseudo_vn_cctv"] / total_train if total_train else 0
    print(f"\n[Totals] V6 train = {total_train} (pseudo chiếm {pseudo_pct:.1f}%)"
          f"\n         V6 val = {counts['v5_val']},  V6 test = {counts['v5_test']}")

    # 3. YAML + manifest
    print("\n[3/3] Save YAML + manifest...")
    YAML_OUT.write_text(f"""# V6 — V5 clean + pseudo-labeled vn_cctv (self-training round 1)
# Sinh tự động bởi scripts/data_prep/build_v6_merged.py
# Val/test giữ nguyên V5 → honest signal (không chứa pseudo-labels)
# Schema: {{motorcycle, car, bus, truck}}

path: {DST}
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
    (DST / "split_manifest.json").write_text(
        json.dumps({"counts": counts, "total_train": total_train,
                    "pseudo_pct_in_train": round(pseudo_pct, 2)},
                   indent=2, ensure_ascii=False)
    )
    print(f"      YAML: {YAML_OUT}")
    print(f"      Manifest: {DST / 'split_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
