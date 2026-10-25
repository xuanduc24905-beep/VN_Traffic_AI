"""Rebuild V5 — video-aware split, no leak.

Pool ALL frames from merged_v3/{train,val,test}, group by video_id,
assign each video to exactly ONE split (train/val/test) per SPLIT_PLAN,
subsample dense videos by stride, then symlink to data/merged_v5/.

Design choices (xem docs/v3_report/leak_audit_summary.md):
- Night cams (dense, ~5k frames each): stride=10 → giảm 90% redundancy
- Day cams + vnv3: stride=5
- Small cantho videos (<100 frames): keep all (stride=1)
- Val/test giữ nguyên tất cả frames của video unseen → max eval signal
  (OK vì không leak — model chưa thấy video này bao giờ)
- vn_cctv KHÔNG gộp: giữ nguyên làm OOD test set riêng

Output: data/merged_v5/{images,labels}/{train,val,test}/
        data/merged_v5/split_manifest.json  (reproducibility)
        data/data_v5.yaml
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path("/home/xuand/vn_traffic_ai").resolve()
SRC = PROJECT_ROOT / "data" / "merged_v3"
DST = PROJECT_ROOT / "data" / "merged_v5"
YAML_OUT = PROJECT_ROOT / "data" / "data_v5.yaml"

# (split, stride)
SPLIT_PLAN: dict[str, tuple[str, int]] = {
    # TRAIN — 11 video, ~2.4k frames sau subsample
    "soictnight_cam_03":           ("train", 10),   # night dense
    "soictnight_cam_05":           ("train", 10),
    "soictnight_cam_08":           ("train", 10),
    "soict24_cam_01":              ("train",  5),
    "soict24_cam_02":              ("train",  5),
    "soict24_cam_04":              ("train",  5),
    "soict24_cam_06":              ("train",  5),
    "cantho_CoopMart-8h-8h30":     ("train",  1),   # tiny
    "cantho_CoopMart-8h10-8h30":   ("train",  5),
    "vnv3":                        ("train",  5),

    # VAL — 3 video UNSEEN
    "soict24_cam_07":                 ("val", 1),
    "cantho_CoopMart-8h30-9h00":      ("val", 1),
    "cantho_camDLHB-13h20---13h57":   ("val", 1),

    # TEST — 2 video UNSEEN
    "soict24_cam_09": ("test", 1),
    "soict24_cam_10": ("test", 1),
}

# Reuse logic từ audit_leak.py
ROBOFLOW_SUFFIX = re.compile(r"_(?:jpg|png|JPG|PNG)\.rf\.[a-f0-9]+\.(?:jpg|png)$", re.I)
VIDEO_PATTERNS = [
    (re.compile(r"^(cantho_.+?)_mp4-\d+$"), lambda m: m.group(1).rstrip("_")),
    (re.compile(r"^(soict(?:\d+|night)_cam_\d+)_\d+$"), lambda m: m.group(1)),
    (re.compile(r"^vnv3_\d+$"), lambda m: "vnv3"),
]
FRAME_NUM = re.compile(r"_(\d+)$|_mp4-(\d+)$")


def extract_video_id(filename: str) -> str:
    stem = ROBOFLOW_SUFFIX.sub("", filename)
    stem = re.sub(r"\.(jpg|png|JPG|PNG)$", "", stem)
    for pat, fn in VIDEO_PATTERNS:
        m = pat.match(stem)
        if m:
            return fn(m)
    return f"UNKNOWN:{stem}"


def extract_frame_num(filename: str) -> int:
    stem = ROBOFLOW_SUFFIX.sub("", filename)
    stem = re.sub(r"\.(jpg|png|JPG|PNG)$", "", stem)
    m = FRAME_NUM.search(stem)
    if m:
        return int(m.group(1) or m.group(2))
    return 0


def pool_source_frames() -> dict[str, list[Path]]:
    """Pool ảnh từ cả 3 split của merged_v3 theo video_id."""
    buckets: dict[str, list[Path]] = defaultdict(list)
    for sub in ("train", "val", "test"):
        img_dir = SRC / "images" / sub
        if not img_dir.is_dir():
            continue
        for p in img_dir.iterdir():
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            vid = extract_video_id(p.name)
            buckets[vid].append(p)
    # Sort frames trong mỗi video theo frame_num (deterministic subsample)
    for vid in buckets:
        buckets[vid].sort(key=lambda p: extract_frame_num(p.name))
    return buckets


def ensure_dirs() -> None:
    for split in ("train", "val", "test"):
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)


def symlink_force(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def copy_pair(img_src: Path, split: str) -> bool:
    """Symlink image + corresponding label. Return True nếu label tồn tại."""
    # Original label có cùng tên file, khác suffix + ở cùng split
    orig_split = img_src.parent.name  # train/val/test của source
    label_src = SRC / "labels" / orig_split / (img_src.stem + ".txt")
    if not label_src.exists():
        return False
    img_dst = DST / "images" / split / img_src.name
    lbl_dst = DST / "labels" / split / (img_src.stem + ".txt")
    symlink_force(img_src, img_dst)
    symlink_force(label_src, lbl_dst)
    return True


def write_yaml() -> None:
    content = f"""# V5 — honest split, video-aware, no leak
# Sinh tự động bởi scripts/data_prep/rebuild_v5_by_video.py
# Xem docs/v3_report/leak_audit_summary.md cho rationale split.
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
"""
    YAML_OUT.write_text(content)
    print(f"[OK] YAML: {YAML_OUT}")


def main() -> int:
    print(f"[1/4] Pool frames từ {SRC}...")
    buckets = pool_source_frames()
    print(f"      Tìm thấy {len(buckets)} video, {sum(len(v) for v in buckets.values())} frames tổng")

    # Sanity check: SPLIT_PLAN phủ hết video có trong source?
    unknown = set(buckets.keys()) - set(SPLIT_PLAN.keys())
    if unknown:
        print(f"      [WARN] Video không có trong SPLIT_PLAN (sẽ bỏ): {sorted(unknown)}")
    missing = set(SPLIT_PLAN.keys()) - set(buckets.keys())
    if missing:
        print(f"      [WARN] Video trong SPLIT_PLAN nhưng không tìm thấy source: {sorted(missing)}")

    print(f"[2/4] Dọn {DST}...")
    ensure_dirs()
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            d = DST / sub / split
            for p in d.iterdir():
                if p.is_symlink() or p.is_file():
                    p.unlink()

    print(f"[3/4] Symlink theo SPLIT_PLAN...")
    manifest: dict[str, dict] = {}
    skipped_no_label = 0
    for vid, (split, stride) in SPLIT_PLAN.items():
        frames = buckets.get(vid, [])
        if not frames:
            continue
        selected = frames[::stride]
        kept = 0
        for p in selected:
            if copy_pair(p, split):
                kept += 1
            else:
                skipped_no_label += 1
        manifest[vid] = {
            "split": split,
            "stride": stride,
            "frames_total": len(frames),
            "frames_selected": len(selected),
            "frames_kept": kept,
        }
        print(f"      {vid:40s} → {split:5s} stride={stride:2d}  "
              f"{len(frames):5d} → {kept:5d}")

    if skipped_no_label:
        print(f"      [WARN] Bỏ {skipped_no_label} frames vì không có label")

    # Totals
    totals = defaultdict(int)
    for info in manifest.values():
        totals[info["split"]] += info["frames_kept"]
    print(f"\n[Totals] train={totals['train']}  val={totals['val']}  test={totals['test']}")

    print(f"\n[4/4] Save manifest + YAML...")
    (DST / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(f"      {DST / 'split_manifest.json'}")
    write_yaml()

    print("\n[DONE] V5 build xong. Verify bằng:")
    print(f"       python scripts/data_prep/audit_leak.py "
          f"--datasets data/merged_v5 --out docs/v3_report/leak_audit_v5.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
