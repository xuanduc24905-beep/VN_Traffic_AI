#!/usr/bin/env python3
"""Count distinct cameras/scenes per data source in V5+ by parsing filenames."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/xuand/vn_traffic_ai")

SOURCE_PATTERNS = [
    ("soict24_day",     re.compile(r"^soict24_cam_(0[1-9]|1[0-2])_")),
    ("soict_night",     re.compile(r"(soict_night|soict-night|night)", re.IGNORECASE)),
    ("cantho",          re.compile(r"^cantho_")),
    ("vn_v3",           re.compile(r"^(vnv3|vn_v3|vietnamese_vehicle)", re.IGNORECASE)),
    ("vn_vehicles_v10", re.compile(r"^v10[_-]")),
]


def classify(name: str) -> str:
    for label, pat in SOURCE_PATTERNS:
        if pat.search(name):
            return label
    return "OTHER"


def cam_key(name: str, src: str) -> str | None:
    if src == "soict24_day":
        m = re.match(r"^(soict24_cam_\d+)", name)
        return m.group(1) if m else None
    if src == "soict_night":
        m = re.search(r"cam[_-]?(\d+)", name, re.IGNORECASE)
        return f"soict_night_cam_{m.group(1)}" if m else "soict_night_unknown"
    if src == "cantho":
        # Pattern: cantho_<location-time>_mp4-...
        # Keep up to "_mp4" as location-time
        m = re.match(r"^cantho_([^_]+)_", name)
        return f"cantho_{m.group(1)}" if m else None
    if src == "vn_vehicles_v10":
        # v10_<gen|general>_<cam>_... or v10_<gen|general>-<cam>-...
        m = re.match(r"^v10_(?:gen|general)[-_]([A-Za-z]+(?:\d+)?)", name)
        return f"v10_{m.group(1)}" if m else "v10_unknown"
    if src == "vn_v3":
        return "v3_scene_unknown"  # no clear scene ID in V3 filenames
    return None


def main():
    all_dirs = [
        ROOT / "data/merged_v5plus/images/train",
        ROOT / "data/merged_v5plus/images/val",
        ROOT / "data/merged_v5plus/images/test",
    ]
    src_cams = defaultdict(set)
    src_cams_basename = defaultdict(set)  # v10 only: ignore trailing digits
    src_count = defaultdict(int)
    for d in all_dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.jpg")):
            name = p.name
            src = classify(name)
            src_count[src] += 1
            cam = cam_key(name, src)
            if cam is None:
                continue
            src_cams[src].add(cam)
            if src == "vn_vehicles_v10":
                base = re.sub(r"\d+$", "", cam)
                src_cams_basename[src].add(base)

    print(f"{'Source':20s} {'Images':>8s} {'Unique cam/scene':>18s}")
    print("-" * 56)
    for src in ("soict24_day", "soict_night", "cantho", "vn_v3", "vn_vehicles_v10", "OTHER"):
        if src not in src_count:
            continue
        n_cam = len(src_cams[src])
        extra = ""
        if src == "vn_vehicles_v10":
            extra = f" (hoặc {len(src_cams_basename[src])} nếu gộp số đuôi)"
        if src == "vn_v3":
            extra = " (~310 tượng trưng mỗi ảnh một cảnh)"
        print(f"{src:20s} {src_count[src]:8d} {n_cam:18d}{extra}")


if __name__ == "__main__":
    main()
