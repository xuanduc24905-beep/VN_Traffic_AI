"""Build V5+ dataset = V5 train (clean VN) + vietnam_vehicles v10 (Roboflow visionvn).

Val/test giữ nguyên V5 (unseen videos SOICT+Cantho+vnv3) → honest signal,
so sánh được fair với V5/V6/V7 trên cùng test set.

Nguồn mới: data/raw_vietnam_vehicles_v10/ (4,824 ảnh, 6 class, snapshot CCTV ~776 camera).

Class remap (6 → 4 V5 schema):
  0 container → 3 truck
  1 oto       → 1 car
  2 xe_ba_gac → SKIP (không có trong V5 schema; ảnh còn 0 bbox sẽ bị bỏ)
  3 xe_khach  → 2 bus
  4 xe_may    → 0 motorcycle
  5 xe_tai    → 3 truck

Output:
  data/merged_v5plus/images/{train,val,test}/   (symlinks)
  data/merged_v5plus/labels/{train,val,test}/   (file thật cho train mới, symlink cho V5)
  data/merged_v5plus/split_manifest.json
  data/data_v5plus.yaml
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path("/home/xuand/vn_traffic_ai").resolve()
V5_ROOT = PROJECT_ROOT / "data" / "merged_v5"
VEHICLE_ROOT = PROJECT_ROOT / "data" / "raw_vietnam_vehicles_v10"
DST = PROJECT_ROOT / "data" / "merged_v5plus"
YAML_OUT = PROJECT_ROOT / "data" / "data_v5plus.yaml"

VEHICLE_SPLITS = ("train", "valid", "test")

# Roboflow class id → V5 class id (None = skip bbox)
CLASS_MAP: dict[int, int | None] = {
    0: 3,     # container → truck
    1: 1,     # oto → car
    2: None,  # xe_ba_gac → SKIP
    3: 2,     # xe_khach → bus
    4: 0,     # xe_may → motorcycle
    5: 3,     # xe_tai → truck
}

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def symlink_force(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def remap_label_file(src_lbl: Path) -> list[str]:
    """Đọc label YOLO, remap class id, trả về list dòng mới (bỏ class skip)."""
    out: list[str] = []
    for line in src_lbl.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
        except ValueError:
            continue
        new_cls = CLASS_MAP.get(cls)
        if new_cls is None:
            continue
        parts[0] = str(new_cls)
        out.append(" ".join(parts))
    return out


def main() -> int:
    assert V5_ROOT.exists(), f"Thiếu {V5_ROOT}"
    assert VEHICLE_ROOT.exists(), f"Thiếu {VEHICLE_ROOT}. Giải nén vietnam_vehicles.v10i.yolov8.zip trước."

    if DST.exists():
        shutil.rmtree(DST)
    for split in ("train", "val", "test"):
        (DST / "images" / split).mkdir(parents=True)
        (DST / "labels" / split).mkdir(parents=True)

    counts: dict[str, int] = {}

    # ─────────── 1. Clone V5 train/val/test (symlink về source thật) ───────────
    print("[1/3] Clone V5 train/val/test (symlink)...")
    for split in ("train", "val", "test"):
        n = 0
        for p in (V5_ROOT / "images" / split).iterdir():
            if p.suffix.lower() not in IMG_EXTS:
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

    # ─────────── 2. Add vietnam_vehicles_v10 vào TRAIN (remap class) ───────────
    print("\n[2/3] Add vietnam_vehicles_v10 (remap 6→4, drop xe_ba_gac)...")
    kept = 0
    dropped_empty = 0
    bbox_total_before = 0
    bbox_total_after = 0
    bbox_dropped_babac = 0
    name_collisions = 0

    for split in VEHICLE_SPLITS:
        img_dir = VEHICLE_ROOT / split / "images"
        lbl_dir = VEHICLE_ROOT / split / "labels"
        if not img_dir.exists():
            continue
        for img_p in img_dir.iterdir():
            if img_p.suffix.lower() not in IMG_EXTS:
                continue
            lbl_p = lbl_dir / (img_p.stem + ".txt")
            if not lbl_p.exists():
                continue

            # Đếm bbox trước/sau remap
            raw_lines = [l for l in lbl_p.read_text().splitlines() if l.strip()]
            bbox_total_before += len(raw_lines)
            bbox_dropped_babac += sum(
                1 for l in raw_lines if l.split()[0:1] == ["2"]
            )

            new_lines = remap_label_file(lbl_p)
            bbox_total_after += len(new_lines)

            if not new_lines:
                dropped_empty += 1
                continue

            # Prefix tên để tránh trùng với V5 (hash Roboflow đã verify không trùng)
            new_stem = f"v10_{img_p.stem}"
            dst_img = DST / "images" / "train" / f"{new_stem}{img_p.suffix.lower()}"
            dst_lbl = DST / "labels" / "train" / f"{new_stem}.txt"

            if dst_img.exists():
                name_collisions += 1
                continue

            symlink_force(img_p.resolve(), dst_img)
            dst_lbl.write_text("\n".join(new_lines) + "\n")
            kept += 1

    counts["vehicle_v10_added"] = kept
    counts["vehicle_v10_dropped_empty"] = dropped_empty
    counts["vehicle_v10_name_collisions"] = name_collisions
    counts["bbox_before_remap"] = bbox_total_before
    counts["bbox_after_remap"] = bbox_total_after
    counts["bbox_dropped_xe_ba_gac"] = bbox_dropped_babac

    print(f"      Added: {kept} ảnh")
    print(f"      Dropped (sau remap 0 bbox): {dropped_empty} ảnh")
    print(f"      Name collisions (skip): {name_collisions}")
    print(f"      Bbox: {bbox_total_before} → {bbox_total_after} "
          f"(drop {bbox_dropped_babac} xe_ba_gac)")

    total_train = counts["v5_train"] + kept
    v10_pct = 100 * kept / total_train if total_train else 0
    print(f"\n[Totals] V5+ train = {total_train} (vehicle_v10 chiếm {v10_pct:.1f}%)"
          f"\n         V5+ val  = {counts['v5_val']}"
          f"\n         V5+ test = {counts['v5_test']}")

    # ─────────── 3. YAML + manifest ───────────
    print("\n[3/3] Save YAML + manifest...")
    YAML_OUT.write_text(f"""# V5+ — V5 clean + vietnam_vehicles v10 (Roboflow visionvn)
# Sinh tự động bởi scripts/data_prep/build_v5plus_merged.py
# Val/test giữ nguyên V5 → honest signal, so sánh fair với V5/V6/V7
# Schema: {{motorcycle, car, bus, truck}}
# Remap: container+xe_tai→truck, oto→car, xe_khach→bus, xe_may→motorcycle, xe_ba_gac dropped

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
                    "vehicle_v10_pct_in_train": round(v10_pct, 2),
                    "class_map": {str(k): v for k, v in CLASS_MAP.items()}},
                   indent=2, ensure_ascii=False)
    )
    print(f"      YAML: {YAML_OUT}")
    print(f"      Manifest: {DST / 'split_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
