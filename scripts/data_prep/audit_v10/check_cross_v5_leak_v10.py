"""(g) Kiểm tra rò rỉ CHÉO giữa v10 và val/test của V5.

Bối cảnh: build_v5plus_merged.py gộp TOÀN BỘ v10 (train+valid+test) vào train V5+,
val/test V5+ giữ nguyên V5 (SOICT24 cam_07/09/10 + Cần Thơ CoopMart/camDLHB).
→ Leak nội bộ v10 không ảnh hưởng eval V5+, nhưng nếu ảnh/camera của v10 trùng
với val/test V5 thì số liệu V5+ bị thổi phồng. Script này kiểm tra điều đó.

Tên camera hai bộ khác hẳn nhau (v10: tên địa điểm; V5: soict24_cam_XX, cantho_...)
nên phải so bằng NỘI DUNG ảnh, 3 cấp:

1. MD5: ảnh y hệt byte.
2. pHash 64-bit từng ảnh: Hamming ≤ 5 → near-duplicate (cùng khung/cảnh).
3. Ảnh nền camera: trung bình xám (128×72) của tối đa N ảnh mỗi camera ≈ nền tĩnh
   (xe chạy qua bị "nhoè" đi). So NCC (normalized cross-correlation) giữa mọi cặp
   (camera v10, video V5 eval). NCC ≥ 0.90 → nghi cùng góc camera.
   Để hiểu ngưỡng, báo kèm phân bố NCC giữa các site KHÁC nhau trong nội bộ v10
   (baseline "khác camera") và giữa các kênh cùng site (baseline "cùng địa điểm").

Thêm: sanity check merged_v5plus val/test không chứa file `v10_*` và trùng với V5.

Xuất preview ảnh nền các cặp NCC cao nhất + cặp pHash gần nhất để kiểm tra bằng mắt.

Verdict: PASS nếu 0 MD5 trùng, 0 cặp pHash ≤5, 0 cặp camera NCC ≥ 0.90, sanity OK.

Usage:
    python scripts/data_prep/audit_v10/check_cross_v5_leak_v10.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_perceptual_duplicates_v10 import _hamming_matrix  # noqa: E402
from v10_common import IMG_EXTS, REPORTS_DIR, REPO_ROOT, load_records, write_json  # noqa: E402

V5_ROOT = REPO_ROOT / "data" / "merged_v5"
V5PLUS_ROOT = REPO_ROOT / "data" / "merged_v5plus"
EVAL_SPLITS = ("val", "test")
OUT = REPORTS_DIR / "v10_cross_v5_leak.json"
PREVIEW = REPORTS_DIR / "v10_cross_v5_preview.jpg"
BG_SIZE = (128, 72)  # (w, h) cho ảnh nền
BG_MAX_IMAGES = 150


def _features(path: str) -> tuple[str, bytes, np.ndarray]:
    """Trả về (md5, phash 8 byte, ảnh xám thu nhỏ float32)."""
    import imagehash
    from PIL import Image
    data = Path(path).read_bytes()
    md5 = hashlib.md5(data).hexdigest()
    with Image.open(path) as im:
        g = im.convert("L")
        ph = np.packbits(imagehash.phash(g, hash_size=8).hash.flatten()).tobytes()
        small = np.asarray(g.resize(BG_SIZE), dtype=np.float32)
    return md5, ph, small


def _v5_eval_images() -> list[dict]:
    manifest = json.loads((V5_ROOT / "split_manifest.json").read_text())
    vids = sorted((k for k, v in manifest.items() if isinstance(v, dict) and v.get("split") in EVAL_SPLITS),
                  key=len, reverse=True)  # khớp prefix dài trước
    out = []
    for s in EVAL_SPLITS:
        for p in sorted((V5_ROOT / "images" / s).iterdir()):
            if p.suffix.lower() not in IMG_EXTS:
                continue  # bỏ cache .npy
            vid = next((v for v in vids if p.name.startswith(v)), "unknown")
            out.append({"split": s, "video": vid, "path": str(p.resolve()), "name": p.stem})
    return out


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    return float((a * b).mean())


def _sanity_v5plus() -> dict:
    res = {}
    for s in EVAL_SPLITS:
        a = {p.name for p in (V5_ROOT / "images" / s).iterdir() if p.suffix.lower() in IMG_EXTS}
        b = {p.name for p in (V5PLUS_ROOT / "images" / s).iterdir() if p.suffix.lower() in IMG_EXTS} \
            if (V5PLUS_ROOT / "images" / s).is_dir() else set()
        res[s] = {"n_v5": len(a), "n_v5plus": len(b), "identical_sets": a == b,
                  "n_v10_files_in_v5plus": sum(n.startswith("v10_") for n in b)}
    res["ok"] = all(v["identical_sets"] and v["n_v10_files_in_v5plus"] == 0
                    for k, v in res.items() if k in EVAL_SPLITS)
    return res


def _preview(pairs_bg, ph_pairs, bg10, bg5, v10_small, v5_small, out: Path) -> None:
    from PIL import Image, ImageDraw
    rows = [(f"NCC={p['ncc']:.3f}  {p['v10_camera']} vs {p['v5_video']}",
             bg10[p["v10_camera"]], bg5[p["v5_video"]]) for p in pairs_bg[:6]]
    rows += [(f"pHash d={p['hamming']}  {p['v10'][:40]} vs {p['v5'][:30]}",
              v10_small[p["_i"]], v5_small[p["_j"]]) for p in ph_pairs[:4]]
    if not rows:
        return
    W, H, pad, txt = BG_SIZE[0] * 3, BG_SIZE[1] * 3, 6, 16
    sheet = Image.new("RGB", (2 * W + 3 * pad, len(rows) * (H + txt + pad) + pad), "white")
    d = ImageDraw.Draw(sheet)
    for r, (label, a, b) in enumerate(rows):
        y = pad + r * (H + txt + pad)
        d.text((pad, y), label, fill="black")
        for c, arr in enumerate((a, b)):
            im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).resize((W, H))
            sheet.paste(im.convert("RGB"), (pad + c * (W + pad), y + txt))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    print(f"[OK] Ghi preview {out.relative_to(REPO_ROOT)}")


def run(out: Path = OUT, phash_thr: int = 5, ncc_thr: float = 0.90, workers: int = 8,
        preview: Path | None = PREVIEW) -> dict:
    v10 = load_records()
    v5 = _v5_eval_images()
    paths = [str(REPO_ROOT / r.path) for r in v10] + [x["path"] for x in v5]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        feats = list(ex.map(_features, paths, chunksize=64))
    f10, f5 = feats[:len(v10)], feats[len(v10):]

    # 1. MD5
    md5_5 = defaultdict(list)
    for x, (m, _, _) in zip(v5, f5):
        md5_5[m].append(x["name"])
    md5_hits = [{"v10": r.source_name, "v5": md5_5[m]} for r, (m, _, _) in zip(v10, f10) if m in md5_5]

    # 2. pHash từng ảnh
    H10 = np.frombuffer(b"".join(f[1] for f in f10), dtype=np.uint8).reshape(-1, 8)
    H5 = np.frombuffer(b"".join(f[1] for f in f5), dtype=np.uint8).reshape(-1, 8)
    ph_pairs, min_d = [], 64
    for start in range(0, len(H10), 512):
        D = _hamming_matrix(H10[start:start + 512], H5)
        min_d = min(min_d, int(D.min()))
        for i, j in zip(*np.nonzero(D <= phash_thr)):
            ph_pairs.append({"hamming": int(D[i, j]), "_i": start + int(i), "_j": int(j),
                             "v10": f"{v10[start + i].split}/{v10[start + i].source_name}",
                             "v5": f"{v5[j]['split']}/{v5[j]['name']}", "v5_video": v5[j]["video"]})
    ph_pairs.sort(key=lambda p: p["hamming"])
    # cặp gần nhất (kể cả > ngưỡng) để xem trực quan
    nearest = []
    if not ph_pairs:
        best = []
        for start in range(0, len(H10), 512):
            D = _hamming_matrix(H10[start:start + 512], H5)
            i, j = np.unravel_index(np.argsort(D, axis=None)[:4], D.shape)
            best += [(int(D[a, b]), start + int(a), int(b)) for a, b in zip(i, j)]
        for d, i, j in sorted(best)[:4]:
            nearest.append({"hamming": d, "_i": i, "_j": j, "v10": v10[i].source_name, "v5": v5[j]["name"]})

    # 3. Ảnh nền theo camera
    rng = random.Random(0)
    g10, g5 = defaultdict(list), defaultdict(list)
    for i, r in enumerate(v10):
        g10[r.camera_id].append(i)
    for j, x in enumerate(v5):
        g5[x["video"]].append(j)

    def _mean(idx, feats_):
        idx = idx if len(idx) <= BG_MAX_IMAGES else rng.sample(idx, BG_MAX_IMAGES)
        return np.mean([feats_[k][2] for k in idx], axis=0)

    bg10 = {c: _mean(ix, f10) for c, ix in g10.items()}
    bg5 = {v: _mean(ix, f5) for v, ix in g5.items()}
    site_of = {r.camera_id: r.site_id for r in v10}

    cross = sorted(({"v10_camera": c, "v5_video": v, "ncc": round(_ncc(a, b), 4)}
                    for c, a in bg10.items() for v, b in bg5.items()), key=lambda p: -p["ncc"])
    cams = sorted(bg10)
    diff_site, same_site = [], []
    for i, a in enumerate(cams):
        for b in cams[i + 1:]:
            (same_site if site_of[a] == site_of[b] else diff_site).append(_ncc(bg10[a], bg10[b]))
    # baseline giữa các video V5 eval khác nhau
    vids = sorted(bg5)
    v5_pairs = {f"{a} vs {b}": round(_ncc(bg5[a], bg5[b]), 4) for i, a in enumerate(vids) for b in vids[i + 1:]}

    # Đối chứng dương: cùng 1 camera, chia ngẫu nhiên 2 nửa → NCC nền phải cao.
    # Nếu p5 của đối chứng < ngưỡng thì phương pháp không đủ nhạy → báo trong JSON.
    def _split_half(groups, feats_):
        vals = {}
        for k, ix in groups.items():
            if len(ix) < 20:
                continue
            ix = ix[:]
            rng.shuffle(ix)
            h = len(ix) // 2
            vals[k] = round(_ncc(_mean(ix[:h], feats_), _mean(ix[h:], feats_)), 4)
        return vals
    pos10, pos5 = _split_half(g10, f10), _split_half(g5, f5)
    pos_all = list(pos10.values()) + list(pos5.values())

    def _q(x):
        return {k: round(float(np.percentile(x, q)), 4) for k, q in
                (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))} if x else {}

    bg_hits = [p for p in cross if p["ncc"] >= ncc_thr]
    sanity = _sanity_v5plus()

    if preview:
        _preview(cross, ph_pairs or nearest, bg10, bg5, [f[2] for f in f10], [f[2] for f in f5], preview)

    # Điều kiện bảo thủ: NCC chéo lớn nhất phải thấp hơn NCC nhỏ nhất của đối chứng dương
    # (nếu không, có thể có camera trùng mà ngưỡng cố định bỏ sót).
    max_cross = cross[0]["ncc"] if cross else -1.0
    margin_ok = (not pos_all) or max_cross < min(pos_all)
    passed = not md5_hits and not ph_pairs and not bg_hits and sanity["ok"] and margin_ok
    for p in ph_pairs + nearest:
        p.pop("_i", None), p.pop("_j", None)
    result = {
        "check": "cross_v5_leak",
        "threshold": (f"0 MD5 trùng, 0 cặp pHash ≤{phash_thr}, 0 cặp camera NCC ≥{ncc_thr} "
                      f"và NCC chéo max < NCC min của đối chứng cùng camera, V5+ val/test = V5"),
        "verdict": "PASS" if passed else "FAIL",
        "n_v10_images": len(v10),
        "n_v5_eval_images": len(v5),
        "v5_eval_videos": {v: len(ix) for v, ix in sorted(g5.items())},
        "md5": {"n_hits": len(md5_hits), "hits": md5_hits[:100]},
        "phash": {"threshold": phash_thr, "n_pairs": len(ph_pairs), "min_hamming": min_d,
                  "pairs": ph_pairs[:500], "nearest_pairs_INFO": nearest},
        "background": {
            "threshold": ncc_thr, "n_hits": len(bg_hits), "hits": bg_hits,
            "max_cross_ncc": max_cross,
            "margin_below_positive_control_min": margin_ok,
            "top10_cross": cross[:10],
            "max_ncc_per_v5_video": {v: max((p for p in cross if p["v5_video"] == v), key=lambda p: p["ncc"])
                                     for v in vids},
            "baseline_v10_diff_site": _q(diff_site),
            "baseline_v10_same_site_diff_channel": _q(same_site),
            "baseline_v5_eval_between_videos": v5_pairs,
            "positive_control_same_camera_split_half": {
                "summary": {**_q(pos_all), "p5": round(float(np.percentile(pos_all, 5)), 4) if pos_all else None,
                            "min": round(min(pos_all), 4) if pos_all else None},
                "n_below_threshold": sum(v < ncc_thr for v in pos_all),
                "v10": pos10, "v5": pos5,
            },
        },
        "sanity_v5plus_eval": sanity,
        "preview": str(preview.relative_to(REPO_ROOT)) if preview else None,
    }
    write_json(result, out)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--phash-threshold", type=int, default=5)
    ap.add_argument("--ncc-threshold", type=float, default=0.90)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    r = run(a.out, a.phash_threshold, a.ncc_threshold, a.workers)
    b = r["background"]
    print(f"[{r['verdict']}] MD5 {r['md5']['n_hits']} | pHash≤{a.phash_threshold}: {r['phash']['n_pairs']} "
          f"(min d={r['phash']['min_hamming']}) | camera NCC≥{a.ncc_threshold}: {b['n_hits']} | "
          f"sanity V5+: {r['sanity_v5plus_eval']['ok']}")
    print("  top NCC:", b["top10_cross"][:5])
    print("  baseline v10 khác site:", b["baseline_v10_diff_site"], "| cùng site:", b["baseline_v10_same_site_diff_channel"])
    print("  V5 eval giữa các video:", b["baseline_v5_eval_between_videos"])
    pc = b["positive_control_same_camera_split_half"]
    print("  Đối chứng dương (cùng camera, 2 nửa):", pc["summary"], "| dưới ngưỡng:", pc["n_below_threshold"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
