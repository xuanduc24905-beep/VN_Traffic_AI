"""Kiểm tra toàn vẹn nhãn YOLO cho dataset.

Khác với `audit_leak.py` (chỉ kiểm tra rò rỉ cấp video), script này kiểm tra
các ràng buộc chất lượng nhãn:

1. Mỗi ảnh có file nhãn .txt tương ứng (song song images/<split>/ ↔ labels/<split>/)
2. class_id nằm trong khoảng hợp lệ [0, nc-1]
3. Tọa độ (cx, cy, w, h) nằm trong [0.0, 1.0]
4. Chiều rộng/chiều cao bbox > 0 (không bbox suy biến)
5. Bbox không vượt biên ảnh (cx±w/2 ∈ [0,1], cy±h/2 ∈ [0,1])
6. Format đúng: mỗi dòng có đúng 5 giá trị cách nhau bằng khoảng trắng
7. Số ảnh "background" (nhãn rỗng) — nhiều quá có thể hint data imbalance

Mục đích: fail-hard trước huấn luyện, tránh crash giữa chừng hoặc kết quả
bất thường do dữ liệu lỗi.

Usage:
    python scripts/data_prep/validate_labels.py \
        --dataset data/merged_v5 \
        --nc 4 \
        --out docs/v3_report/label_validation_v5.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

SPLITS = ("train", "val", "test")
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def check_label_file(lbl_path: Path, nc: int) -> list[str]:
    """Trả về list các issue tìm thấy trong file label."""
    issues = []
    try:
        text = lbl_path.read_text()
    except Exception as e:
        return [f"cannot_read: {e}"]

    if not text.strip():
        # Empty label is valid (negative sample) — track separately
        return ["__empty__"]

    for line_idx, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()

        # 6. Format check — exactly 5 values
        if len(parts) != 5:
            issues.append(f"line{line_idx}: wrong_format ({len(parts)} tokens, expected 5)")
            continue

        try:
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
        except ValueError as e:
            issues.append(f"line{line_idx}: parse_error ({e})")
            continue

        # 2. Class ID check
        if not (0 <= cls_id < nc):
            issues.append(f"line{line_idx}: bad_class_id={cls_id} (expected 0..{nc-1})")

        # 3. Coord in [0,1]
        for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
            if not (0.0 <= val <= 1.0):
                issues.append(f"line{line_idx}: {name}={val:.4f} out_of_range")

        # 4. Non-degenerate bbox
        if w <= 0 or h <= 0:
            issues.append(f"line{line_idx}: degenerate_bbox w={w:.4f} h={h:.4f}")

        # 5. Bbox not crossing image border
        left, right = cx - w / 2, cx + w / 2
        top, bottom = cy - h / 2, cy + h / 2
        if left < -0.001 or right > 1.001 or top < -0.001 or bottom > 1.001:
            issues.append(
                f"line{line_idx}: bbox_exceeds_border "
                f"[{left:.3f},{right:.3f}]x[{top:.3f},{bottom:.3f}]"
            )

    return issues if issues else []


def validate_split(dataset_root: Path, split: str, nc: int) -> dict:
    """Validate một split, trả về dict summary + list issues."""
    img_dir = dataset_root / "images" / split
    lbl_dir = dataset_root / "labels" / split

    if not img_dir.is_dir():
        return {"status": "skip", "reason": f"missing {img_dir}"}

    # Collect all images and labels by stem
    img_stems = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    lbl_stems = {p.stem for p in lbl_dir.iterdir() if p.suffix.lower() == ".txt"} \
                if lbl_dir.is_dir() else set()

    # 1. Pairing check
    images_without_label = sorted(img_stems - lbl_stems)
    labels_without_image = sorted(lbl_stems - img_stems)

    # 2-6. Validate each label file
    n_empty = 0
    file_issues: dict[str, list[str]] = {}
    cls_counter: Counter = Counter()

    for stem in sorted(img_stems & lbl_stems):
        lbl_path = lbl_dir / f"{stem}.txt"
        issues = check_label_file(lbl_path, nc)
        if issues == ["__empty__"]:
            n_empty += 1
        elif issues:
            file_issues[stem] = issues
        else:
            # Count classes for distribution report
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if parts and parts[0].isdigit():
                    cls_counter[int(parts[0])] += 1

    return {
        "status": "ok" if not (file_issues or images_without_label or labels_without_image) else "has_issues",
        "n_images": len(img_stems),
        "n_labels": len(lbl_stems),
        "n_paired": len(img_stems & lbl_stems),
        "n_empty_labels": n_empty,
        "images_without_label": images_without_label[:20],  # cap for brevity
        "images_without_label_count": len(images_without_label),
        "labels_without_image": labels_without_image[:20],
        "labels_without_image_count": len(labels_without_image),
        "n_files_with_issues": len(file_issues),
        "file_issues": dict(list(file_issues.items())[:20]),  # first 20 only
        "class_distribution": {str(k): v for k, v in sorted(cls_counter.items())},
    }


def print_summary(results: dict, nc: int) -> bool:
    """Print human-readable report. Return True if all splits are OK."""
    all_ok = True
    for split, r in results.items():
        if r.get("status") == "skip":
            print(f"\n[{split.upper()}] SKIP: {r.get('reason')}")
            continue

        print(f"\n{'=' * 70}")
        print(f"[{split.upper()}] {r['n_paired']} pairs, "
              f"{r['n_empty_labels']} empty labels (negative samples)")
        print(f"{'=' * 70}")

        if r["images_without_label_count"]:
            all_ok = False
            print(f"  ❌ {r['images_without_label_count']} images missing .txt label")
            for stem in r["images_without_label"][:5]:
                print(f"       {stem}")
            if r["images_without_label_count"] > 5:
                print(f"       ... (và {r['images_without_label_count'] - 5} ảnh khác)")

        if r["labels_without_image_count"]:
            all_ok = False
            print(f"  ❌ {r['labels_without_image_count']} labels without image")

        if r["n_files_with_issues"]:
            all_ok = False
            print(f"  ❌ {r['n_files_with_issues']} files have content issues:")
            for stem, issues in list(r["file_issues"].items())[:3]:
                print(f"     • {stem}: {issues[:3]}")

        if r["status"] == "ok":
            print(f"  ✅ All checks PASSED")

        # Class distribution
        total_boxes = sum(r["class_distribution"].values())
        if total_boxes:
            print(f"\n  Class distribution ({total_boxes} boxes):")
            for cls_id in range(nc):
                n = r["class_distribution"].get(str(cls_id), 0)
                pct = 100 * n / total_boxes if total_boxes else 0
                print(f"    class {cls_id}: {n:6d} ({pct:5.1f}%)")

    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True,
                    help="Dataset root (chứa images/ và labels/)")
    ap.add_argument("--nc", type=int, required=True,
                    help="Số class hợp lệ (vd 4 cho {motorcycle, car, bus, truck})")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSON path")
    ap.add_argument("--fail-on-issues", action="store_true",
                    help="Exit code 1 nếu có bất kỳ issue nào")
    args = ap.parse_args()

    assert args.dataset.exists(), f"Thiếu {args.dataset}"

    print(f"Validate dataset: {args.dataset}")
    print(f"Expected nc: {args.nc}")

    results = {}
    for split in SPLITS:
        results[split] = validate_split(args.dataset, split, args.nc)

    all_ok = print_summary(results, args.nc)

    # Save JSON
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({
            "dataset": str(args.dataset),
            "nc": args.nc,
            "all_passed": all_ok,
            "splits": results,
        }, indent=2, ensure_ascii=False)
    )
    print(f"\n[OK] Chi tiết: {args.out}")

    if not all_ok and args.fail_on_issues:
        print("\n[FAIL] Dataset có issues. Exit 1.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
