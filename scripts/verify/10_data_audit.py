#!/usr/bin/env python3
"""BƯỚC 1 — Data audit.

- Count images per split per source (V5+ train/val/test, VN Aerial val/test).
- Per-class bbox counts.
- Per-size (COCO) bbox counts: areas in <32², 32²–96², >96² pixels (using original image resolution).
- Label integrity: malformed lines, class ids out of {0,1,2,3}, coords out of [0,1], duplicate bboxes, empty images.
- Leakage check: videos/cameras appearing in >1 split (match by base-name prefix).

Outputs:
  outputs/verify/data_audit.json  (full details)
  outputs/verify/data_audit_summary.txt (human-readable)
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path("/home/xuand/vn_traffic_ai")

CLASSES = {0: "motorcycle", 1: "car", 2: "bus", 3: "truck"}

# Source classifiers based on filename patterns. Order matters (first match wins).
SOURCE_PATTERNS = [
    ("soict24_day",   re.compile(r"^soict24_cam_(0[1-9]|1[0-2])_")),  # SOICT24 day cameras
    ("soict_night",   re.compile(r"(soict_night|soict-night|night)", re.IGNORECASE)),
    ("cantho",        re.compile(r"^cantho_")),
    ("vn_vehicles_v10", re.compile(r"^(vnv10|vietnam_vehicles|vn_vehicles|visionvn|vv)[_\.]", re.IGNORECASE)),
    ("vn_v3",         re.compile(r"^(vnv3|vn_v3|vietnamese_vehicle)", re.IGNORECASE)),
    ("vn_cctv",       re.compile(r"^vn_?cctv")),
]


def classify_source(name: str) -> str:
    for label, pat in SOURCE_PATTERNS:
        if pat.search(name):
            return label
    return "OTHER"


def base_key(name: str) -> str:
    """Strip Roboflow hash and extension to get video-ish id."""
    s = name
    # drop ".jpg" or similar
    s = re.sub(r"\.[A-Za-z]+$", "", s)
    # drop ".rf.<hash>"
    s = re.sub(r"\.rf\.[A-Fa-f0-9]+$", "", s)
    # drop trailing "_jpg" if present
    s = re.sub(r"_jpg$", "", s)
    # drop trailing "-<digits>" frame id to collapse frames of the same video
    s = re.sub(r"[-_]\d+$", "", s)
    return s


def iter_images(split_dir: Path):
    for p in sorted(split_dir.iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            yield p


def audit_split(images_dir: Path, labels_dir: Path, split_name: str, dataset: str) -> dict:
    per_source_imgs = Counter()
    per_source_cls_bbox = defaultdict(lambda: Counter())       # src -> class_id -> count
    per_source_size_bbox = defaultdict(lambda: Counter())      # src -> size bin -> count
    overall_cls = Counter()
    overall_size = Counter()
    base_to_split = defaultdict(set)  # computed outside across splits
    issues = {
        "malformed_lines": [],
        "bad_class_id": [],
        "coords_out_of_range": [],
        "duplicate_bboxes": [],
        "images_without_labels": [],
        "labels_without_images": [],
    }
    bases = []
    n_img_total = 0
    n_bbox_total = 0
    empty_labels = 0  # labels file exists but zero lines
    # Enumerate images first
    img_bases = set()
    for img_path in iter_images(images_dir):
        n_img_total += 1
        src = classify_source(img_path.name)
        per_source_imgs[src] += 1
        bk = base_key(img_path.name)
        bases.append((bk, src))
        img_bases.add(img_path.stem)
        # resolve label
        lab = labels_dir / (img_path.stem + ".txt")
        if not lab.exists():
            issues["images_without_labels"].append(str(img_path.relative_to(ROOT)))
            continue
        try:
            w, h = Image.open(img_path).size
        except Exception as e:
            issues["malformed_lines"].append(f"{img_path.name}: cannot open image: {e}")
            continue
        boxes_seen = set()
        n_lines = 0
        with lab.open() as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                n_lines += 1
                parts = line.split()
                if len(parts) != 5:
                    issues["malformed_lines"].append(f"{lab.name}:{ln}  '{line}'")
                    continue
                try:
                    cid = int(parts[0])
                    cx, cy, bw, bh = (float(x) for x in parts[1:])
                except ValueError:
                    issues["malformed_lines"].append(f"{lab.name}:{ln}  parse error: '{line}'")
                    continue
                if cid not in CLASSES:
                    issues["bad_class_id"].append(f"{lab.name}:{ln} cid={cid}")
                    continue
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= bw <= 1 and 0 <= bh <= 1):
                    issues["coords_out_of_range"].append(f"{lab.name}:{ln}  {parts}")
                    continue
                key = (cid, round(cx, 4), round(cy, 4), round(bw, 4), round(bh, 4))
                if key in boxes_seen:
                    issues["duplicate_bboxes"].append(f"{lab.name}:{ln}  {parts}")
                    continue
                boxes_seen.add(key)
                # area (COCO) based on image resolution
                area_px = (bw * w) * (bh * h)
                if area_px < 32 * 32:
                    size_bin = "small"
                elif area_px < 96 * 96:
                    size_bin = "medium"
                else:
                    size_bin = "large"
                per_source_cls_bbox[src][cid] += 1
                per_source_size_bbox[src][size_bin] += 1
                overall_cls[cid] += 1
                overall_size[size_bin] += 1
                n_bbox_total += 1
        if n_lines == 0:
            empty_labels += 1
    # Labels without images
    for lab in sorted(labels_dir.glob("*.txt")):
        if lab.stem not in img_bases:
            issues["labels_without_images"].append(str(lab.relative_to(ROOT)))

    # trim issue lists to 50 for JSON
    issues_short = {k: (v[:50] + (["…"] if len(v) > 50 else [])) for k, v in issues.items()}
    issue_counts = {k: len(v) for k, v in issues.items()}

    return {
        "dataset": dataset,
        "split": split_name,
        "n_images": n_img_total,
        "n_bboxes": n_bbox_total,
        "empty_labels": empty_labels,
        "per_source_images": dict(per_source_imgs),
        "per_source_cls_bbox": {s: dict(c) for s, c in per_source_cls_bbox.items()},
        "per_source_size_bbox": {s: dict(c) for s, c in per_source_size_bbox.items()},
        "overall_cls_bbox": dict(overall_cls),
        "overall_size_bbox": dict(overall_size),
        "issue_counts": issue_counts,
        "issues_sample": issues_short,
        "_bases": bases,  # removed from final JSON
    }


def check_leakage(audits_by_split: dict[str, dict]) -> dict:
    """base_key may appear in multiple splits -> leakage."""
    base_to_splits = defaultdict(lambda: defaultdict(int))
    for split, a in audits_by_split.items():
        for bk, src in a.get("_bases", []):
            base_to_splits[bk][split] += 1
    leaks = []
    for bk, splits in base_to_splits.items():
        if len(splits) > 1:
            leaks.append({"base": bk, "splits": dict(splits)})
    return {"n_leaks": len(leaks), "leaks": leaks[:200]}


def main():
    out = {"datasets": {}}

    # V5+ train/val/test
    v5 = {
        "train": (ROOT / "data/merged_v5plus/images/train", ROOT / "data/merged_v5plus/labels/train"),
        "val":   (ROOT / "data/merged_v5plus/images/val",   ROOT / "data/merged_v5plus/labels/val"),
        "test":  (ROOT / "data/merged_v5plus/images/test",  ROOT / "data/merged_v5plus/labels/test"),
    }
    v5_audits = {}
    for split, (img, lab) in v5.items():
        print(f"V5+ {split}…", flush=True)
        a = audit_split(img, lab, split, "V5+")
        v5_audits[split] = a
    out["datasets"]["V5+"] = {s: {k: v for k, v in a.items() if k != "_bases"} for s, a in v5_audits.items()}
    out["datasets"]["V5+"]["leakage_across_splits"] = check_leakage(v5_audits)

    # VN Aerial val/test
    vna = {
        "val":   (ROOT / "data/vn_vehicles_aerial/images/val",   ROOT / "data/vn_vehicles_aerial/labels/val"),
        "test":  (ROOT / "data/vn_vehicles_aerial/images/test",  ROOT / "data/vn_vehicles_aerial/labels/test"),
    }
    vna_audits = {}
    for split, (img, lab) in vna.items():
        print(f"VN Aerial {split}…", flush=True)
        a = audit_split(img, lab, split, "VN_Aerial")
        vna_audits[split] = a
    out["datasets"]["VN_Aerial"] = {s: {k: v for k, v in a.items() if k != "_bases"} for s, a in vna_audits.items()}
    out["datasets"]["VN_Aerial"]["leakage_across_splits"] = check_leakage(vna_audits)

    # Write
    outfile = ROOT / "outputs/verify/data_audit.json"
    outfile.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {outfile}")

    # Human-readable summary
    lines = []
    def emit(s=""):
        lines.append(s)
        print(s)
    for ds in ["V5+", "VN_Aerial"]:
        emit(f"\n=== {ds} ===")
        for sp in ("train", "val", "test"):
            a = out["datasets"][ds].get(sp)
            if not a:
                continue
            emit(f"  {sp}: {a['n_images']} images, {a['n_bboxes']} bboxes, empty={a['empty_labels']}")
            emit(f"    cls bbox: {a['overall_cls_bbox']}  (0=moto, 1=car, 2=bus, 3=truck)")
            emit(f"    size bbox (COCO): {a['overall_size_bbox']}")
            emit(f"    per-source images: {a['per_source_images']}")
            emit(f"    per-source cls bbox: {a['per_source_cls_bbox']}")
            emit(f"    per-source size bbox: {a['per_source_size_bbox']}")
            emit(f"    issue_counts: {a['issue_counts']}")
        leaks = out["datasets"][ds]["leakage_across_splits"]
        emit(f"  LEAKAGE across splits: {leaks['n_leaks']} base keys appear in >1 split")
        for lk in leaks["leaks"][:10]:
            emit(f"    - {lk['base']} -> {lk['splits']}")

    (ROOT / "outputs/verify/data_audit_summary.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
