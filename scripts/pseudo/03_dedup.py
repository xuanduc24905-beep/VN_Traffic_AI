"""Loại ảnh trùng trong data/raw/frames/hcm/ bằng perceptual hash.

Windy có thể cache → cùng 1 camera 2 vòng liền kề trả ảnh giống nhau.
Perceptual hash (pHash) phát hiện cả ảnh giống về mặt nội dung (dù JPEG khác byte).

Chạy:
  python scripts/pseudo/03_dedup.py                 # dry-run, chỉ report
  python scripts/pseudo/03_dedup.py --apply         # thật sự xoá
  python scripts/pseudo/03_dedup.py --threshold 5   # chặt hơn (mặc định 3)
"""

import argparse
from pathlib import Path
from collections import defaultdict

try:
    import imagehash
    from PIL import Image
except ImportError:
    print("Cần cài: pip install imagehash pillow")
    raise

FRAMES_DIR = Path("data/raw/frames/hcm")


def compute_hashes(ward_dir: Path):
    """Trả list [(path, phash)] sort theo filename."""
    items = []
    for p in sorted(ward_dir.glob("*.jpg")):
        try:
            h = imagehash.phash(Image.open(p))
            items.append((p, h))
        except Exception as e:
            print(f"  skip {p.name}: {e}")
    return items


def find_duplicates(items, threshold: int):
    """Sliding-window: với mỗi ảnh, so với <=5 ảnh trước đó.
    Khoảng cách Hamming <= threshold -> trùng (giữ cái sớm hơn)."""
    to_remove = []
    for i in range(len(items)):
        path_i, h_i = items[i]
        for j in range(max(0, i - 5), i):
            path_j, h_j = items[j]
            if path_j in to_remove:
                continue
            if h_i - h_j <= threshold:
                to_remove.append(path_i)
                break
    return to_remove


def main(apply: bool, threshold: int):
    stats = defaultdict(lambda: {"total": 0, "dup": 0})
    all_removed = []

    for ward_dir in sorted(FRAMES_DIR.iterdir()):
        if not ward_dir.is_dir():
            continue
        items = compute_hashes(ward_dir)
        dups = find_duplicates(items, threshold)
        stats[ward_dir.name]["total"] = len(items)
        stats[ward_dir.name]["dup"] = len(dups)
        all_removed.extend(dups)
        print(f"[{ward_dir.name}] total={len(items)} dup={len(dups)}")

    total = sum(s["total"] for s in stats.values())
    total_dup = sum(s["dup"] for s in stats.values())
    print(f"\n[SUMMARY] total={total} dup={total_dup} "
          f"keep={total - total_dup} ({100 * (total - total_dup) / max(1, total):.1f}%)")

    if apply:
        for p in all_removed:
            p.unlink()
        print(f"[APPLIED] removed {len(all_removed)} files")
    else:
        print("\n(dry-run — chạy với --apply để xoá thật)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--threshold", type=int, default=3,
                    help="Hamming distance <= threshold => trùng (mặc định 3)")
    args = ap.parse_args()
    main(args.apply, args.threshold)
