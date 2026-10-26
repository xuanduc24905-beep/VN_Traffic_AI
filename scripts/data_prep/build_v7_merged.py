"""Build V7 = V5 train (VN clean) + DETRAC bus/truck subset.

Val/test giữ nguyên V5 (unseen VN videos) → honest signal, không bị
ô nhiễm bởi DETRAC bus/truck VN-khác.

Output: data/merged_v7/{images,labels}/{train,val,test} + data/data_v7.yaml
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path("/home/xuand/vn_traffic_ai").resolve()
V5_ROOT = PROJECT_ROOT / "data" / "merged_v5"
DETRAC_ROOT = PROJECT_ROOT / "data" / "detrac_bustruck_subset"
DST = PROJECT_ROOT / "data" / "merged_v7"
YAML_OUT = PROJECT_ROOT / "data" / "data_v7.yaml"


def symlink_force(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main() -> int:
    assert V5_ROOT.exists(), f"Thiếu {V5_ROOT}"
    assert DETRAC_ROOT.exists(), f"Thiếu {DETRAC_ROOT}. Chạy import_detrac_bustruck.py trước."

    if DST.exists():
        shutil.rmtree(DST)
    for split in ("train", "val", "test"):
        (DST / "images" / split).mkdir(parents=True)
        (DST / "labels" / split).mkdir(parents=True)

    counts = {}

    # 1. Clone V5 (all splits)
    print("[1/3] Clone V5 train/val/test...")
    for split in ("train", "val", "test"):
        n = 0
        for p in (V5_ROOT / "images" / split).iterdir():
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img_src = p.resolve()
            lbl_src = (V5_ROOT / "labels" / split / (p.stem + ".txt")).resolve()
            if not lbl_src.exists():
                continue
            symlink_force(img_src, DST / "images" / split / p.name)
            symlink_force(lbl_src, DST / "labels" / split / (p.stem + ".txt"))
            n += 1
        counts[f"v5_{split}"] = n
        print(f"      V5 {split}: {n} frames")

    # 2. Add DETRAC bus/truck subset vào TRAIN
    print("\n[2/3] Add DETRAC bus/truck subset vào train...")
    n = 0
    for p in (DETRAC_ROOT / "images").iterdir():
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img_src = p.resolve()
        lbl_src = (DETRAC_ROOT / "labels" / (p.stem + ".txt"))
        if not lbl_src.exists():
            continue
        symlink_force(img_src, DST / "images" / "train" / p.name)
        symlink_force(lbl_src.resolve(), DST / "labels" / "train" / (p.stem + ".txt"))
        n += 1
    counts["detrac_bustruck"] = n
    print(f"      DETRAC bus/truck added: {n} frames")

    total_train = counts["v5_train"] + counts["detrac_bustruck"]
    detrac_pct = 100 * counts["detrac_bustruck"] / total_train
    print(f"\n[Totals] V7 train = {total_train} "
          f"(DETRAC chiếm {detrac_pct:.1f}%)")
    print(f"         V7 val = {counts['v5_val']}, V7 test = {counts['v5_test']}")

    # 3. YAML + manifest
    print("\n[3/3] Save YAML + manifest...")
    YAML_OUT.write_text(f"""# V7 — V5 clean + DETRAC bus/truck subset
# Sinh tự động bởi scripts/data_prep/build_v7_merged.py
# Val/test giữ nguyên V5 (unseen VN videos) → honest signal
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
                    "detrac_pct_in_train": round(detrac_pct, 2)},
                   indent=2, ensure_ascii=False)
    )
    print(f"      YAML: {YAML_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
