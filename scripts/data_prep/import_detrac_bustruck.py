"""Import DETRAC bus/van subset vào format YOLO V5-compatible.

Chọn video thông minh: top-K video có nhiều bus + top-K video có nhiều van
(deduplicate nếu trùng). Subsample video-aware (stride) để tránh redundancy.

Class remap:
  DETRAC car (0)    → DROP (V5+ car đã mạnh 0.88)
  DETRAC bus (1)    → V5 bus (2)
  DETRAC van (2)    → V5 truck (3)   ← user accept risk domain mismatch
  DETRAC others (3) → DROP

Output: data/detrac_bustruck_subset/
  images/  (symlinks)
  labels/  (converted + filtered)
  manifest.json
"""
from __future__ import annotations

import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path("/home/xuand/vn_traffic_ai").resolve()
SRC_IMG = Path("/home/xuand/yolov8_finetune/data/detrac/images/train")
SRC_LBL = Path("/home/xuand/yolov8_finetune/data/detrac/labels/train")
DST = PROJECT_ROOT / "data" / "detrac_bustruck_subset"

VIDEO_RE = re.compile(r"^(MVI_\d+)_")
FRAME_RE = re.compile(r"_img(\d+)")

TOP_K_BUS = 6
TOP_K_VAN = 6
STRIDE = 20   # ~50 frames mỗi video → ~550-650 frames total

# DETRAC → V5
REMAP = {1: 2, 2: 3}   # bus→bus, van→truck. Drop 0 (car) + 3 (others).


def main() -> int:
    assert SRC_IMG.exists(), f"Thiếu {SRC_IMG}"
    assert SRC_LBL.exists(), f"Thiếu {SRC_LBL}"

    # Audit per-video class counts
    per_video_cls: dict[str, Counter] = defaultdict(Counter)
    per_video_frames: dict[str, list[Path]] = defaultdict(list)
    for p in SRC_LBL.iterdir():
        if p.suffix.lower() != ".txt":
            continue
        m = VIDEO_RE.match(p.name)
        if not m:
            continue
        vid = m.group(1)
        for line in p.read_text().splitlines():
            parts = line.split()
            if parts:
                per_video_cls[vid][int(parts[0])] += 1
        # Store image path (same stem)
        img = SRC_IMG / (p.stem + ".jpg")
        if img.exists():
            per_video_frames[vid].append(img)

    # Rank videos
    bus_rank = sorted(per_video_cls.items(), key=lambda kv: -kv[1][1])[:TOP_K_BUS]
    van_rank = sorted(per_video_cls.items(), key=lambda kv: -kv[1][2])[:TOP_K_VAN]
    picked_videos = sorted(set([v for v, _ in bus_rank] + [v for v, _ in van_rank]))

    print(f"[INFO] Picked {len(picked_videos)} videos:")
    for v in picked_videos:
        c = per_video_cls[v]
        print(f"  {v}: {c[1]} bus, {c[2]} van ({c[0]} car dropped)")

    # Dọn + tạo DST
    if DST.exists():
        shutil.rmtree(DST)
    (DST / "images").mkdir(parents=True)
    (DST / "labels").mkdir(parents=True)

    kept_imgs = 0
    kept_boxes = Counter()

    for vid in picked_videos:
        frames = sorted(per_video_frames[vid],
                        key=lambda p: int(FRAME_RE.search(p.name).group(1))
                        if FRAME_RE.search(p.name) else 0)
        selected = frames[::STRIDE]
        for img in selected:
            lbl_src = SRC_LBL / (img.stem + ".txt")
            if not lbl_src.exists():
                continue
            # Filter + remap
            new_lines = []
            for line in lbl_src.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                c = int(parts[0])
                if c not in REMAP:
                    continue
                new_c = REMAP[c]
                new_lines.append(f"{new_c} " + " ".join(parts[1:]))
                kept_boxes[new_c] += 1
            if not new_lines:
                continue   # skip ảnh không còn box bus/van sau filter
            new_name = f"detrac_{img.name}"
            (DST / "images" / new_name).symlink_to(img.resolve())
            (DST / "labels" / (Path(new_name).stem + ".txt")).write_text(
                "\n".join(new_lines)
            )
            kept_imgs += 1

    # Report
    NAMES = {2: "bus", 3: "truck"}
    print(f"\n[DONE] DETRAC bus/truck subset built:")
    print(f"  Videos picked: {len(picked_videos)}")
    print(f"  Frames kept: {kept_imgs}")
    print(f"  Boxes kept:")
    for c, n in sorted(kept_boxes.items()):
        print(f"    {NAMES.get(c, c):8s}: {n}")

    manifest = {
        "source": str(SRC_IMG),
        "picked_videos": picked_videos,
        "remap": {str(k): v for k, v in REMAP.items()},
        "stride": STRIDE,
        "frames_kept": kept_imgs,
        "boxes_per_class": {NAMES.get(c, str(c)): n for c, n in kept_boxes.items()},
        "per_video_original": {v: dict(per_video_cls[v]) for v in picked_videos},
    }
    (DST / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(f"\n[OK] Output: {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
