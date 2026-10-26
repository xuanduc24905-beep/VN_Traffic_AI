"""Audit video-level data leakage in merged datasets.

Parse video_id from filename, count frames per (video, split), report:
- Total unique videos per dataset
- Videos that appear in more than one split (leaky)
- % of val/test frames whose source video is also in train

Usage:
    python scripts/data_prep/audit_leak.py \
        --datasets data/merged_v3 data/merged_v4 \
        --out docs/v3_report/leak_audit.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")

# Match Roboflow suffix: _jpg.rf.<hash>.jpg (also _png variant)
ROBOFLOW_SUFFIX = re.compile(r"_(?:jpg|png|JPG|PNG)\.rf\.[a-f0-9]+\.(?:jpg|png)$", re.I)

# Patterns to extract video_id from stem (after stripping Roboflow suffix)
# Order matters: specific → generic
VIDEO_PATTERNS = [
    # cantho_<VIDEONAME>_mp4-<FRAME>   → cantho_<VIDEONAME>
    (re.compile(r"^(cantho_.+?)_mp4-\d+$"), lambda m: m.group(1)),
    # soict24_cam_<XX>_<NNNNN>  or  soictnight_cam_<XX>_<NNNNN>
    (re.compile(r"^(soict(?:\d+|night)_cam_\d+)_\d+$"), lambda m: m.group(1)),
    # vncctv_<split>_<NNNNNN>  → vncctv_<split>  (train/test are 2 pseudo-"videos" from merge)
    (re.compile(r"^(vncctv_(?:train|test|val))_\d+$"), lambda m: m.group(1)),
    # vnv3_<NNNN>              → vnv3  (whole set = 1 video)
    (re.compile(r"^vnv3_\d+$"), lambda m: "vnv3"),
    # DETRAC subset: detrac_MVI_<XXXXX>_img<NNNNN>  → detrac_MVI_<XXXXX>
    (re.compile(r"^detrac_(MVI_\d+)_img\d+$"), lambda m: f"detrac_{m.group(1)}"),
    # vn_cctv pseudo-labeled: vncctv_pseudo_<NNNNNN>  → vncctv_pseudo (1 video source)
    (re.compile(r"^vncctv_pseudo_\d+$"), lambda m: "vncctv_pseudo"),
]


def extract_video_id(filename: str) -> str:
    stem = ROBOFLOW_SUFFIX.sub("", filename)
    # Also drop any trailing .jpg if Roboflow suffix didn't match
    stem = re.sub(r"\.(jpg|png|JPG|PNG)$", "", stem)
    for pattern, extractor in VIDEO_PATTERNS:
        m = pattern.match(stem)
        if m:
            return extractor(m)
    return f"UNKNOWN:{stem}"


def scan_split(images_dir: Path) -> Counter:
    """Return Counter[video_id] = frame_count for all image files in dir."""
    counts: Counter = Counter()
    if not images_dir.is_dir():
        return counts
    for p in images_dir.iterdir():
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        counts[extract_video_id(p.name)] += 1
    return counts


def audit_dataset(root: Path) -> dict:
    """Audit one merged_vN dataset."""
    per_split: dict[str, Counter] = {}
    for split in SPLITS:
        per_split[split] = scan_split(root / "images" / split)

    all_videos = set().union(*(c.keys() for c in per_split.values()))

    # Per-video breakdown
    per_video: dict[str, dict] = {}
    for v in sorted(all_videos):
        per_video[v] = {s: per_split[s].get(v, 0) for s in SPLITS}
        per_video[v]["splits_present"] = [
            s for s in SPLITS if per_split[s].get(v, 0) > 0
        ]
        per_video[v]["is_leaky"] = len(per_video[v]["splits_present"]) > 1

    # Aggregate leak metrics
    train_videos = {v for v, c in per_split["train"].items() if c > 0}

    def leak_pct(split_name: str) -> dict:
        total = sum(per_split[split_name].values())
        leaked = sum(c for v, c in per_split[split_name].items() if v in train_videos)
        return {
            "total_frames": total,
            "frames_from_train_videos": leaked,
            "pct_leaked": round(100 * leaked / total, 2) if total else 0.0,
        }

    leaky_videos = [v for v, info in per_video.items() if info["is_leaky"]]

    return {
        "dataset": str(root),
        "n_videos_total": len(all_videos),
        "n_videos_train": len(train_videos),
        "n_videos_leaky_crosssplit": len(leaky_videos),
        "frames_per_split": {s: sum(c.values()) for s, c in per_split.items()},
        "leak_metrics": {
            "val": leak_pct("val"),
            "test": leak_pct("test"),
        },
        "leaky_videos": leaky_videos,
        "per_video": per_video,
    }


def summarize_console(audits: list[dict]) -> None:
    for a in audits:
        print(f"\n{'=' * 70}")
        print(f"DATASET: {a['dataset']}")
        print(f"{'=' * 70}")
        print(f"  Total unique videos: {a['n_videos_total']}")
        print(f"  Videos in train:     {a['n_videos_train']}")
        print(f"  Videos leaky (multi-split): {a['n_videos_leaky_crosssplit']}")
        print(f"  Frames per split: {a['frames_per_split']}")
        for split in ("val", "test"):
            m = a["leak_metrics"][split]
            print(
                f"  {split.upper()}: {m['frames_from_train_videos']}/{m['total_frames']} "
                f"frames ({m['pct_leaked']}%) come from videos also in train"
            )
        print(f"\n  Top videos (frame counts train/val/test):")
        rows = sorted(
            a["per_video"].items(),
            key=lambda kv: -sum(kv[1][s] for s in SPLITS),
        )
        for v, info in rows[:15]:
            flag = " [LEAK]" if info["is_leaky"] else ""
            print(
                f"    {v:40s}  train={info['train']:5d}  "
                f"val={info['val']:4d}  test={info['test']:4d}{flag}"
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True, help="paths to merged_vN roots")
    ap.add_argument("--out", type=Path, required=True, help="output JSON path")
    args = ap.parse_args()

    audits = [audit_dataset(Path(d)) for d in args.datasets]
    summarize_console(audits)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audits, indent=2, ensure_ascii=False))
    print(f"\n[OK] Full audit saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
