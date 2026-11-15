"""Download Vietnam Vehicles aerial dataset từ Ultralytics Platform.

Nguồn: data/vietnam-vehicles.ndjson (metadata + per-image URL + annotations)
Đích:  data/vn_vehicles_aerial/images/{val,test}/
       data/vn_vehicles_aerial/labels/{val,test}/
       data/data_vn_aerial.yaml

Chỉ download split VAL + TEST (635 ảnh) để làm eval độc lập, không dùng train.

Class remap:
  src 0 bus    → dst 2 bus
  src 1 car    → dst 1 car
  src 2 motor  → dst 0 motorcycle  (V5+ schema)
  src 3 truck  → dst 3 truck
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path("/home/xuand/vn_traffic_ai")
NDJSON = ROOT / "data" / "vietnam-vehicles.ndjson"
DST = ROOT / "data" / "vn_vehicles_aerial"
YAML_OUT = ROOT / "data" / "data_vn_aerial.yaml"
V5PLUS_HASHES_FILE = ROOT / "data" / "v5plus_md5.json"  # tạo nếu chưa có

# src_class_id → V5+ dst_class_id
CLASS_REMAP = {0: 2, 1: 1, 2: 0, 3: 3}
DST_CLASS_NAMES = {0: "motorcycle", 1: "car", 2: "bus", 3: "truck"}

SPLITS_TO_DOWNLOAD = {"val", "test"}
TIMEOUT = 20  # giây
MAX_WORKERS = 16


def build_v5plus_hashes() -> set[str]:
    """Compute MD5 của mọi ảnh trong V5+ để check leak sau. Cache vào JSON."""
    if V5PLUS_HASHES_FILE.exists():
        print(f"[Hash] load cache {V5PLUS_HASHES_FILE.name}")
        return set(json.loads(V5PLUS_HASHES_FILE.read_text()))

    hashes: set[str] = set()
    for split in ("train", "val", "test"):
        d = ROOT / "data" / "merged_v5plus" / "images" / split
        if not d.exists():
            continue
        imgs = [p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        print(f"[Hash] V5+ {split}: {len(imgs)} ảnh...")
        for p in imgs:
            try:
                hashes.add(hashlib.md5(p.read_bytes()).hexdigest())
            except Exception:
                pass
    print(f"[Hash] Tổng {len(hashes)} hash unique V5+")
    V5PLUS_HASHES_FILE.write_text(json.dumps(sorted(hashes)))
    return hashes


def download_one(record: dict, img_dir: Path) -> tuple[str, bool, str]:
    """Tải 1 ảnh. Trả về (filename, success, md5)."""
    fname = record["file"]
    url = record["url"]
    out_path = img_dir / fname
    if out_path.exists() and out_path.stat().st_size > 0:
        md5 = hashlib.md5(out_path.read_bytes()).hexdigest()
        return (fname, True, md5)
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        if resp.status_code != 200:
            return (fname, False, "")
        data = resp.content
        out_path.write_bytes(data)
        md5 = hashlib.md5(data).hexdigest()
        return (fname, True, md5)
    except Exception as e:
        return (fname, False, f"error: {e}")


def write_label(record: dict, lbl_dir: Path) -> int:
    """Viết nhãn YOLO txt. Trả về số box valid."""
    fname = record["file"]
    boxes = record.get("annotations", {}).get("boxes", [])
    lines = []
    for b in boxes:
        if not isinstance(b, (list, tuple)) or len(b) < 5:
            continue
        src_cls, cx, cy, w, h = int(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4])
        dst_cls = CLASS_REMAP.get(src_cls)
        if dst_cls is None:
            continue
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            continue
        lines.append(f"{dst_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    lbl_path = lbl_dir / (Path(fname).stem + ".txt")
    lbl_path.write_text("\n".join(lines))
    return len(lines)


def main():
    assert NDJSON.exists(), f"Thiếu {NDJSON}"

    # Setup dirs
    for split in SPLITS_TO_DOWNLOAD:
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Parse records
    records = []
    with NDJSON.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("type") == "image" and obj.get("split") in SPLITS_TO_DOWNLOAD:
                records.append(obj)
    print(f"[Scan] {len(records)} ảnh val+test cần tải")

    # Build V5+ hash set (để check leak)
    v5plus_hashes = build_v5plus_hashes()

    # Download songsong
    print(f"\n[Download] bắt đầu với {MAX_WORKERS} threads...")
    t0 = time.time()
    per_split_count = {s: 0 for s in SPLITS_TO_DOWNLOAD}
    per_split_leak = {s: 0 for s in SPLITS_TO_DOWNLOAD}
    failed = []
    md5_map = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_one, r, DST / "images" / r["split"]): r for r in records}
        done = 0
        for fut in as_completed(futures):
            r = futures[fut]
            fname, ok, md5 = fut.result()
            done += 1
            if not ok:
                failed.append((fname, md5))
                continue
            md5_map[fname] = md5
            # Check leak
            if md5 in v5plus_hashes:
                per_split_leak[r["split"]] += 1
                print(f"  [LEAK] {fname} trùng V5+ → sẽ bỏ")
                # Xoá ảnh đã tải
                (DST / "images" / r["split"] / fname).unlink(missing_ok=True)
                continue
            # Viết label
            n_box = write_label(r, DST / "labels" / r["split"])
            per_split_count[r["split"]] += 1
            if done % 100 == 0:
                print(f"  [{done}/{len(records)}] {time.time()-t0:.0f}s")

    elapsed = time.time() - t0
    print(f"\n[Download] Xong trong {elapsed:.0f}s")
    print(f"[Fail] {len(failed)} ảnh không tải được")
    for s in SPLITS_TO_DOWNLOAD:
        total_s = sum(1 for r in records if r['split'] == s)
        print(f"  {s}: {per_split_count[s]}/{total_s} ảnh OK (leak: {per_split_leak[s]})")

    # Write yaml
    yaml_content = f"""# Vietnam Vehicles aerial - eval độc lập (Ultralytics Platform)
# Camera trên cao (aerial), khác với V5+ (CCTV ngang)
# Class 0 motorcycle, 1 car, 2 bus, 3 truck (đã remap)

path: {DST}
val:   images/val
test:  images/test

nc: 4
names:
  0: motorcycle
  1: car
  2: bus
  3: truck
"""
    YAML_OUT.write_text(yaml_content)
    print(f"\n[YAML] Lưu {YAML_OUT}")

    # Summary
    print("\n=== TÓM TẮT ===")
    for s in SPLITS_TO_DOWNLOAD:
        n_img = len(list((DST / "images" / s).iterdir()))
        n_lbl = len(list((DST / "labels" / s).iterdir()))
        print(f"  {s}: {n_img} ảnh, {n_lbl} label")


if __name__ == "__main__":
    sys.exit(main())
