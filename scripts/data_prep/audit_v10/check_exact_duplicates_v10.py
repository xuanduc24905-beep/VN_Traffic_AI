"""(a) Kiểm tra trùng lặp TUYỆT ĐỐI (byte-level) trong v10.

- MD5 toàn bộ ảnh train/valid/test.
- Cross-split: cùng MD5 ở ≥2 split  → LEAK trực tiếp.
- Within-split: cùng MD5 trong 1 split → dư thừa (không leak, nhưng làm lệch phân phối).
- Bổ sung: cùng `source_name` gốc (trước hậu tố Roboflow `.rf.<hash>`) ở ≥2 split,
  và các file có hậu tố `-1-` (bản copy khi export) — byte có thể khác do
  Roboflow re-encode nên MD5 không bắt được.

Verdict: PASS nếu 0 nhóm MD5 cross-split và 0 source_name cross-split.

Usage:
    python scripts/data_prep/audit_v10/check_exact_duplicates_v10.py
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_common import REPORTS_DIR, REPO_ROOT, V10_ROOT, load_records, write_json  # noqa: E402

OUT = REPORTS_DIR / "v10_exact_duplicates.json"


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(root: Path = V10_ROOT, out: Path = OUT) -> dict:
    recs = load_records(root)
    by_md5: dict[str, list] = defaultdict(list)
    for r in recs:
        by_md5[md5_file(REPO_ROOT / r.path)].append(r)

    cross, within = [], []
    for h, group in by_md5.items():
        if len(group) < 2:
            continue
        splits = sorted({g.split for g in group})
        entry = {"md5": h, "splits": splits, "files": [f"{g.split}/{g.stem}" for g in group]}
        (cross if len(splits) > 1 else within).append(entry)

    # Trùng source_name gốc (Roboflow có thể re-encode → MD5 khác)
    by_src: dict[str, list] = defaultdict(list)
    for r in recs:
        # bỏ hậu tố "-1-" để gom bản copy về cùng tên gốc
        by_src[r.source_name.replace("-1-", "") if r.is_copy_suffix else r.source_name].append(r)
    src_cross, src_within = [], []
    for name, group in by_src.items():
        if len(group) < 2:
            continue
        splits = sorted({g.split for g in group})
        entry = {"source_name": name, "splits": splits,
                 "files": [f"{g.split}/{g.stem}" for g in group]}
        (src_cross if len(splits) > 1 else src_within).append(entry)

    n_cross_imgs = sum(len(e["files"]) for e in cross)
    passed = not cross and not src_cross
    result = {
        "check": "exact_duplicates",
        "threshold": "0 nhóm MD5 trùng cross-split, 0 source_name trùng cross-split",
        "verdict": "PASS" if passed else "FAIL",
        "n_images": len(recs),
        "n_unique_md5": len(by_md5),
        "n_groups_cross_split": len(cross),
        "n_images_in_cross_split_groups": n_cross_imgs,
        "n_groups_within_split": len(within),
        "n_redundant_within_split": sum(len(e["files"]) - 1 for e in within),
        "n_source_name_cross_split": len(src_cross),
        "n_source_name_within_split": len(src_within),
        "n_copy_suffix_files": sum(r.is_copy_suffix for r in recs),
        "cross_split_groups": cross,
        "within_split_groups": within,
        "source_name_cross_split": src_cross,
        "source_name_within_split": src_within,
    }
    write_json(result, out)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=V10_ROOT)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    r = run(a.root, a.out)
    print(f"[{r['verdict']}] MD5 cross-split: {r['n_groups_cross_split']} nhóm | "
          f"within-split: {r['n_groups_within_split']} nhóm | "
          f"source_name cross-split: {r['n_source_name_cross_split']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
