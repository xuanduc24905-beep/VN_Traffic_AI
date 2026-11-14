"""(c) Kiểm tra rò rỉ cấp XE (cùng chiếc xe ở nhiều split).

Hai nguồn định danh xe trong tên file:
1. vehicle_id (`ID123`/`id123`): đây là track-ID của tracker trên TỪNG camera,
   reset theo phiên → chỉ có nghĩa khi đi kèm camera_id. Ta dùng:
     - khoá chặt  (camera_id, vehicle_id, ngày)  → gần như chắc chắn cùng 1 track
     - khoá lỏng  (camera_id, vehicle_id)         → có thể là xe khác (tracker reset)
   vehicle_id "trần" (không kèm camera) chỉ báo để tham khảo, KHÔNG dùng làm verdict.
2. license_plate (kết quả OCR): cùng biển ở 2 split → cùng chiếc xe (dù khác camera).
   Bỏ qua biển quá ngắn/rác (<6 ký tự) để tránh trùng giả.

Verdict: PASS nếu 0 khoá chặt cross-split và 0 biển số cross-split.

Usage:
    python scripts/data_prep/audit_v10/check_vehicle_id_leak_v10.py
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_common import REPORTS_DIR, V10_ROOT, load_records, write_json  # noqa: E402

OUT = REPORTS_DIR / "v10_vehicle_leak.json"
MIN_PLATE_LEN = 6


def _cross(groups: dict) -> list[dict]:
    out = []
    for k, rs in groups.items():
        splits = sorted({r.split for r in rs})
        if len(splits) > 1:
            out.append({"key": k if isinstance(k, str) else "|".join(map(str, k)),
                        "splits": splits,
                        "files": [f"{r.split}/{r.source_name}" for r in rs]})
    return sorted(out, key=lambda e: -len(e["files"]))


def _pct_eval_affected(cross: list[dict], recs) -> dict:
    files = {f for e in cross for f in e["files"]}
    out = {}
    for s in ("valid", "test"):
        n = sum(r.split == s for r in recs)
        k = sum(1 for f in files if f.startswith(s + "/"))
        out[s] = round(100 * k / n, 2) if n else 0
    return out


def run(root: Path = V10_ROOT, out: Path = OUT) -> dict:
    recs = load_records(root)

    strict, loose, bare, plate = (defaultdict(list) for _ in range(4))
    for r in recs:
        if r.vehicle_id:
            d = r.dt.date().isoformat() if r.dt else "?"
            strict[(r.camera_id, r.vehicle_id, d)].append(r)
            loose[(r.camera_id, r.vehicle_id)].append(r)
            bare[r.vehicle_id].append(r)
        if r.plate and len(r.plate) >= MIN_PLATE_LEN and "UNKNOWN" not in r.plate:
            plate[r.plate].append(r)

    c_strict, c_loose, c_bare, c_plate = map(_cross, (strict, loose, bare, plate))
    passed = not c_strict and not c_plate
    result = {
        "check": "vehicle_leak",
        "threshold": "0 (camera,vehicle_id,ngày) cross-split và 0 biển số cross-split",
        "verdict": "PASS" if passed else "FAIL",
        "n_with_vehicle_id": sum(1 for r in recs if r.vehicle_id),
        "n_with_plate": sum(len(v) for v in plate.values()),
        "n_unique_plate": len(plate),
        "strict_camera_vid_date": {
            "n_keys": len(strict), "n_cross_split": len(c_strict),
            "pct_eval_images_affected": _pct_eval_affected(c_strict, recs), "items": c_strict},
        "plate": {
            "n_keys": len(plate), "n_cross_split": len(c_plate),
            "pct_eval_images_affected": _pct_eval_affected(c_plate, recs), "items": c_plate},
        "loose_camera_vid_INFO": {
            "n_keys": len(loose), "n_cross_split": len(c_loose), "items": c_loose[:50]},
        "bare_vehicle_id_INFO_ONLY": {
            "note": "track-ID reset theo camera → trùng ID trần KHÔNG có nghĩa là cùng xe",
            "n_keys": len(bare), "n_cross_split": len(c_bare)},
    }
    write_json(result, out)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=V10_ROOT)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    r = run(a.root, a.out)
    print(f"[{r['verdict']}] (cam,vid,ngày) cross-split: {r['strict_camera_vid_date']['n_cross_split']} | "
          f"biển số cross-split: {r['plate']['n_cross_split']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
