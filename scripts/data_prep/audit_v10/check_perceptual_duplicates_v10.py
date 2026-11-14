"""(e) Kiểm tra ảnh GẦN TRÙNG (perceptual hash) giữa các split.

MD5 chỉ bắt ảnh y hệt byte. Ảnh re-encode, resize, hay 2 snapshot cùng cảnh
cách nhau vài giây vẫn có pHash rất gần nhau.

- pHash 64-bit (imagehash.phash, hash_size=8) cho mọi ảnh.
- Hamming distance ≤ 5 giữa 2 split khác nhau → near-duplicate.
- So sánh pairwise vector hoá bằng numpy: train×valid, train×test, valid×test.

Cảnh báo: ảnh CCTV cùng camera có nền tĩnh rất giống nhau → pHash có thể
coi 2 ảnh khác xe là "gần trùng". Đây vẫn là tín hiệu leak nền camera
(bổ trợ cho check_camera_leak).

Verdict: PASS nếu 0 cặp cross-split có Hamming ≤ threshold.

Usage:
    python scripts/data_prep/audit_v10/check_perceptual_duplicates_v10.py --threshold 5
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_common import REPORTS_DIR, REPO_ROOT, V10_ROOT, load_records, write_json  # noqa: E402

OUT = REPORTS_DIR / "v10_phash_near_duplicates.json"
SPLIT_PAIRS = (("train", "valid"), ("train", "test"), ("valid", "test"))
_POPCNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _phash(path: str) -> bytes:
    import imagehash
    from PIL import Image
    with Image.open(path) as im:
        h = imagehash.phash(im.convert("L"), hash_size=8)
    return np.packbits(h.hash.flatten()).tobytes()


def _hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,8) uint8, b: (M,8) uint8 → (N,M) Hamming distance."""
    x = np.bitwise_xor(a[:, None, :], b[None, :, :])
    return _POPCNT[x].sum(axis=2, dtype=np.uint16)


def run(root: Path = V10_ROOT, out: Path = OUT, threshold: int = 5, workers: int = 8) -> dict:
    recs = load_records(root)
    paths = [str(REPO_ROOT / r.path) for r in recs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        hashes = list(ex.map(_phash, paths, chunksize=64))
    H = np.frombuffer(b"".join(hashes), dtype=np.uint8).reshape(-1, 8)

    idx = {s: np.array([i for i, r in enumerate(recs) if r.split == s]) for s in ("train", "valid", "test")}
    pairs, per_pair, dist_hist = [], {}, Counter()
    flagged = {s: set() for s in idx}
    for sa, sb in SPLIT_PAIRS:
        ia, ib = idx[sa], idx[sb]
        if not len(ia) or not len(ib):
            continue
        n = 0
        for start in range(0, len(ia), 512):  # chia khối để tiết kiệm RAM
            blk = ia[start:start + 512]
            D = _hamming_matrix(H[blk], H[ib])
            ii, jj = np.nonzero(D <= threshold)
            for i, j in zip(ii, jj):
                a, b = recs[blk[i]], recs[ib[j]]
                d = int(D[i, j])
                dist_hist[d] += 1
                flagged[sa].add(a.path)
                flagged[sb].add(b.path)
                pairs.append({"split_pair": f"{sa}-{sb}", "hamming": d,
                              "same_camera": a.camera_id == b.camera_id,
                              "a": f"{sa}/{a.source_name}", "b": f"{sb}/{b.source_name}"})
                n += 1
        per_pair[f"{sa}-{sb}"] = n

    # Gần trùng trong cùng split (chỉ đếm, tham khảo)
    within = {}
    for s, ii in idx.items():
        cnt = 0
        for start in range(0, len(ii), 512):
            blk = ii[start:start + 512]
            D = _hamming_matrix(H[blk], H[ii])
            r_, c_ = np.nonzero(D <= threshold)
            cnt += int(np.sum(ii[c_] > blk[r_]))
        within[s] = cnt

    n_split = Counter(r.split for r in recs)
    pairs.sort(key=lambda p: p["hamming"])
    result = {
        "check": "phash_near_duplicates",
        "threshold": f"0 cặp cross-split có Hamming(pHash 64-bit) ≤ {threshold}",
        "verdict": "PASS" if not pairs else "FAIL",
        "hamming_threshold": threshold,
        "n_cross_split_pairs": len(pairs),
        "pairs_by_split_pair": per_pair,
        "hamming_histogram": {str(k): v for k, v in sorted(dist_hist.items())},
        "pct_pairs_same_camera": round(100 * sum(p["same_camera"] for p in pairs) / len(pairs), 2) if pairs else 0,
        "n_images_flagged": {s: len(v) for s, v in flagged.items()},
        "pct_images_flagged": {s: round(100 * len(v) / n_split[s], 2) for s, v in flagged.items() if n_split[s]},
        "n_within_split_pairs_INFO": within,
        "pairs": pairs[:2000],
        "pairs_truncated": len(pairs) > 2000,
    }
    write_json(result, out)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=V10_ROOT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    r = run(a.root, a.out, a.threshold, a.workers)
    print(f"[{r['verdict']}] {r['n_cross_split_pairs']} cặp near-dup cross-split "
          f"{r['pairs_by_split_pair']}; % ảnh dính: {r['pct_images_flagged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
