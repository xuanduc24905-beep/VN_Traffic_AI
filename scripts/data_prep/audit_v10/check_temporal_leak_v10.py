"""(d) Kiểm tra rò rỉ THỜI GIAN: ảnh gần nhau về thời gian trên cùng camera.

Ảnh snapshot cách nhau vài giây trên cùng camera gần như cùng một cảnh
giao thông (cùng xe xung quanh, cùng ánh sáng) → nếu nằm ở 2 split khác nhau
thì tương đương leak frame liền kề trong video.

Ngưỡng: cùng camera_id + cùng ngày + |Δt| < 60 giây, khác split.

Verdict: PASS nếu 0 cặp vi phạm.

Usage:
    python scripts/data_prep/audit_v10/check_temporal_leak_v10.py --window 60
"""
from __future__ import annotations

import argparse
import sys
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_common import REPORTS_DIR, V10_ROOT, load_records, write_json  # noqa: E402

OUT = REPORTS_DIR / "v10_temporal_leak.json"


def run(root: Path = V10_ROOT, out: Path = OUT, window_s: int = 60) -> dict:
    recs = [r for r in load_records(root) if r.dt and r.camera_id]
    by_cam_day: dict[tuple, list] = defaultdict(list)
    for r in recs:
        by_cam_day[(r.camera_id, r.dt.date())].append(r)

    pairs = []
    flagged = set()
    within_pairs = 0
    for (cam, day), rs in by_cam_day.items():
        rs.sort(key=lambda r: r.dt)
        ts = [r.dt.timestamp() for r in rs]
        for i, r in enumerate(rs):
            # chỉ xét j > i để không đếm trùng
            j_end = bisect_left(ts, ts[i] + window_s, lo=i + 1)
            for j in range(i + 1, j_end):
                o = rs[j]
                if o.split == r.split:
                    within_pairs += 1
                    continue
                dt = abs(ts[j] - ts[i])
                pairs.append({"camera_id": cam, "date": day.isoformat(), "delta_s": dt,
                              "a": f"{r.split}/{r.source_name}", "b": f"{o.split}/{o.source_name}",
                              "split_pair": "-".join(sorted((r.split, o.split)))})
                flagged.add((r.split, r.path))
                flagged.add((o.split, o.path))

    n_split = Counter(r.split for r in recs)
    flagged_split = Counter(s for s, _ in flagged)
    # phân bố Δt để thấy mức độ "dính" của cảnh
    delta_hist = Counter(
        "0-1s" if p["delta_s"] <= 1 else "2-5s" if p["delta_s"] <= 5 else
        "6-15s" if p["delta_s"] <= 15 else "16-30s" if p["delta_s"] <= 30 else "31-59s"
        for p in pairs)
    passed = not pairs
    result = {
        "check": "temporal_leak",
        "threshold": f"0 cặp ảnh khác split, cùng camera + cùng ngày, cách nhau < {window_s}s",
        "verdict": "PASS" if passed else "FAIL",
        "window_s": window_s,
        "n_cross_split_pairs": len(pairs),
        "n_within_split_pairs_INFO": within_pairs,
        "pairs_by_split_pair": dict(Counter(p["split_pair"] for p in pairs)),
        "delta_histogram": dict(delta_hist),
        "n_images_flagged": {s: flagged_split.get(s, 0) for s in n_split},
        "pct_images_flagged": {s: round(100 * flagged_split.get(s, 0) / n, 2) for s, n in n_split.items()},
        "pairs": sorted(pairs, key=lambda p: p["delta_s"]),
    }
    write_json(result, out)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=V10_ROOT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--window", type=int, default=60, help="Ngưỡng giây (mặc định 60)")
    a = ap.parse_args()
    r = run(a.root, a.out, a.window)
    print(f"[{r['verdict']}] {r['n_cross_split_pairs']} cặp cross-split <{a.window}s; "
          f"% ảnh bị dính: {r['pct_images_flagged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
