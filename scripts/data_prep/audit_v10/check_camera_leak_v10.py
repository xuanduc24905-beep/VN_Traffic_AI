"""(b) Kiểm tra rò rỉ cấp CAMERA trong v10.

v10 là ảnh snapshot CCTV: cùng một camera → cùng góc nhìn, nền, ánh sáng, vạch kẻ.
Nếu 1 camera có ảnh ở cả train lẫn test thì model "học thuộc" nền camera đó,
mAP test bị thổi phồng so với camera mới hoàn toàn (giống video-level leak V1-V4).

Hai cấp:
- camera_id: vd `nga3phuocthe554` (địa điểm + kênh)
- site_id:   vd `nga3phuocthe` (bỏ số kênh) — nhiều kênh cùng 1 ngã ba vẫn
             nhìn chung cảnh/xe → leak mềm.

Verdict: PASS nếu 0 camera_id xuất hiện ở >1 split. Site-level báo WARN.

Usage:
    python scripts/data_prep/audit_v10/check_camera_leak_v10.py
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_common import REPORTS_DIR, SPLITS, V10_ROOT, load_records, write_json  # noqa: E402

OUT = REPORTS_DIR / "v10_camera_leak.json"


def _leak_stats(recs, key: str) -> dict:
    dist: dict[str, Counter] = defaultdict(Counter)
    for r in recs:
        dist[getattr(r, key) or "UNPARSED"][r.split] += 1
    shared = {k: dict(c) for k, c in dist.items() if len(c) > 1}
    n_test_leaky = sum(c.get("test", 0) for k, c in dist.items() if k in shared)
    n_val_leaky = sum(c.get("valid", 0) for k, c in dist.items() if k in shared)
    n_test = sum(r.split == "test" for r in recs)
    n_val = sum(r.split == "valid" for r in recs)
    per_split_unique = {s: sorted(k for k, c in dist.items() if c.get(s)) for s in SPLITS}
    return {
        "n_unique": len(dist),
        "n_unique_per_split": {s: len(v) for s, v in per_split_unique.items()},
        "n_shared_across_splits": len(shared),
        "n_only_in": {s: sum(1 for c in dist.values() if set(c) == {s}) for s in SPLITS},
        "pct_test_images_from_shared": round(100 * n_test_leaky / n_test, 2) if n_test else 0,
        "pct_valid_images_from_shared": round(100 * n_val_leaky / n_val, 2) if n_val else 0,
        "shared": dict(sorted(shared.items(), key=lambda kv: -sum(kv[1].values()))),
        "distribution": {k: dict(c) for k, c in sorted(dist.items(), key=lambda kv: -sum(kv[1].values()))},
    }


def run(root: Path = V10_ROOT, out: Path = OUT) -> dict:
    recs = load_records(root)
    cam = _leak_stats(recs, "camera_id")
    site = _leak_stats(recs, "site_id")
    passed = cam["n_shared_across_splits"] == 0
    result = {
        "check": "camera_leak",
        "threshold": "0 camera_id xuất hiện ở >1 split (site-level chỉ WARN)",
        "verdict": "PASS" if passed else "FAIL",
        "site_verdict": "PASS" if site["n_shared_across_splits"] == 0 else "WARN",
        "camera": cam,
        "site": site,
    }
    write_json(result, out)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=V10_ROOT)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    r = run(a.root, a.out)
    c = r["camera"]
    print(f"[{r['verdict']}] {c['n_unique']} camera, {c['n_shared_across_splits']} camera ở >1 split; "
          f"{c['pct_test_images_from_shared']}% ảnh test đến từ camera đã thấy khi train/val")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
