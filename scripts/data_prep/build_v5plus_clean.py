"""Build V5+clean — bản V5+ làm sạch có kiểm chứng, kèm biến thể ablation.

KHÔNG sửa/xoá data/merged_v5plus cũ (giữ làm baseline tái lập được).
KHÔNG train. Chỉ tạo dataset (symlink ảnh + file nhãn đã remap) + manifest.

Nguyên tắc thiết kế (xem reports/v5plus_clean_datasheet.md):
  • Test = V5 test NGUYÊN VẸN (SOICT24 cam_09/10; fingerprint SHA-256 để kiểm).
  • Val = V5 val BỎ phần Cần Thơ (`--val-policy clean`, mặc định): 2 video Cần Thơ trong val
    (CoopMart-8h30-9h00, camDLHB) là CÙNG CAMERA với 2 video Cần Thơ trong train (pHash d=2,
    NCC ảnh nền 0.98/0.76 vs đối chứng cùng video ≥0.73; kiểm bằng mắt) → val bị leak cấp camera,
    làm sai việc chọn checkpoint/early-stop. Val sạch = soict24_cam_07 (1000 frame, camera riêng).
    466 frame bị bỏ KHÔNG chuyển sang train (giữ V5 train y nguyên để so sánh được).
    `--val-policy original` tái tạo val cũ.
  • V5 train giữ nguyên (đã audit video-level, 0% leak).
  • v10 (Roboflow vietnam_vehicles) CHỈ vào train, cả 3 split gốc (split gốc của v10 bị leak nội bộ,
    nhưng không chạm tới val/test V5 — đã kiểm bằng check_cross_v5_leak_v10.py).
  • Các quy tắc lọc v10 được cố định TRƯỚC khi train (không chỉnh theo kết quả test):
      R1  Bỏ CẢ ẢNH nếu có box xe_ba_gac. Bản cũ chỉ xoá box → xe ba gác thành "nền" chưa gán
          nhãn (false negative có hệ thống).
      R2  Gộp ảnh trùng thật: cùng camera + pHash ≤ 4 + (cùng biển số HOẶC cách ≤ 2s); giữ ảnh
          có nhiều box nhất. Không dùng pHash thuần vì camera tĩnh → nền giống nhau dù khác xe
          (đã kiểm bằng mắt: 6/8 cặp pHash≤2 là xe khác).
      R3  Bỏ ảnh có xe chưa gán nhãn theo detector COCO YOLOv8m (conf≥0.5, IoU<0.3, IoA<0.6,
          diện tích ≥0.2%) — giảm false negative trong nhãn.
      R4  Bỏ ảnh có nhãn lỗi format (validate_labels.check_label_file).
  • Overlay vẽ sẵn (tag/khung của hệ thống tracking gốc) ⇔ tên file có vehicle ID — quy tắc đã
    kiểm định 40/40 trên mẫu held-out mù (tune 59/60). Dùng để tạo biến thể ablation.

Biến thể (cùng V5 train/val/test, chỉ khác phần v10):
  v5base      CHỈ V5 train (không v10) + val sạch               → baseline train lại theo cùng giao thức
  full        V5 train + toàn bộ v10 sau R1–R4                   → bản chính để train
  nooverlay   V5 train + v10 sau R1–R4 KHÔNG có overlay (no-ID)  → ablation
  ovmatched   V5 train + v10 sau R1–R4 CÓ overlay, lấy mẫu cùng số lượng & cùng phân bố camera
              với `nooverlay` (seed cố định)                    → đối chứng cho `nooverlay`
  So nooverlay vs ovmatched ⇒ ảnh hưởng của overlay ở cùng kích thước dữ liệu.

Class remap v10 (6) → V5 (4): container,xe_tai→truck(3); oto→car(1); xe_khach→bus(2);
xe_may→motorcycle(0); xe_ba_gac → bỏ cả ảnh (R1).

Usage:
    conda activate yolov8_ft
    python scripts/data_prep/build_v5plus_clean.py            # build cả 3 biến thể
    python scripts/data_prep/verify_v5plus_clean.py           # kiểm tra + sinh datasheet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "audit_v10"))
from validate_labels import check_label_file  # noqa: E402
from v10_common import REPO_ROOT, label_path_of, load_records  # noqa: E402
from check_perceptual_duplicates_v10 import _hamming_matrix, _phash  # noqa: E402

V5_ROOT = REPO_ROOT / "data" / "merged_v5"
DATA_DIR = REPO_ROOT / "data"
MANIFEST_DIR = REPO_ROOT / "reports" / "v5plus_clean"
DET_CACHE = MANIFEST_DIR / "v10_detector_flags.json"
IMG_EXTS = {".jpg", ".jpeg", ".png"}
SEED = 0

CLASS_MAP = {0: 3, 1: 1, 2: None, 3: 2, 4: 0, 5: 3}
V10_NAMES = ["container", "oto", "xe_ba_gac", "xe_khach", "xe_may", "xe_tai"]
V5_NAMES = ["motorcycle", "car", "bus", "truck"]
BA_GAC = 2

DEDUP_PHASH, DEDUP_DT_S = 4, 2

# Video trong V5 val bị loại khỏi val (cùng camera với train) — xem docstring
VAL_LEAKY_PREFIXES = ("cantho_",)

VARIANTS = {
    "v5base": "merged_v5_cleanval",
    "full": "merged_v5plus_clean",
    "nooverlay": "merged_v5plus_clean_nooverlay",
    "ovmatched": "merged_v5plus_clean_ovmatched",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def v5_split_items(split: str, val_policy: str = "clean") -> list[tuple[Path, Path]]:
    out = []
    for p in sorted((V5_ROOT / "images" / split).iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue  # bỏ cache .npy
        if split == "val" and val_policy == "clean" and p.name.startswith(VAL_LEAKY_PREFIXES):
            continue
        lbl = V5_ROOT / "labels" / split / f"{p.stem}.txt"
        if lbl.exists():
            out.append((p, lbl))
    return out


def eval_fingerprint(root: Path, val_exclude: tuple[str, ...] = ()) -> dict:
    """SHA-256 của (tên ảnh, nội dung nhãn) val/test — chứng minh benchmark không đổi."""
    fp = {}
    for split in ("val", "test"):
        h = hashlib.sha256()
        n = 0
        for p in sorted((root / "images" / split).iterdir()):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            if split == "val" and val_exclude and p.name.startswith(val_exclude):
                continue
            lbl = root / "labels" / split / f"{p.stem}.txt"
            h.update(p.name.encode())
            h.update(lbl.read_bytes() if lbl.exists() else b"<missing>")
            n += 1
        fp[split] = {"n_images": n, "sha256": h.hexdigest()}
    return fp


# ───────────────────────────── lọc v10 ─────────────────────────────
def read_v10_labels(rec) -> tuple[list[list[str]], list[str]]:
    lp = label_path_of(REPO_ROOT / rec.path)
    issues = check_label_file(lp, len(V10_NAMES))
    rows = [l.split() for l in lp.read_text().splitlines() if l.strip()] if lp.exists() else []
    return rows, ([] if issues == ["__empty__"] else issues)


def dedup_groups(recs, H: np.ndarray) -> dict[int, int]:
    """Union-find: trả về map index → root."""
    D = _hamming_matrix(H, H)
    cam = np.array([r.camera_id for r in recs])
    plate = np.array([r.plate or f"__none{i}" for i, r in enumerate(recs)])
    ts = np.array([r.dt.timestamp() for r in recs])
    m = (D <= DEDUP_PHASH) & (cam[:, None] == cam[None, :]) & (
        (plate[:, None] == plate[None, :]) | (np.abs(ts[:, None] - ts[None, :]) <= DEDUP_DT_S))
    m = np.triu(m, 1)
    par = list(range(len(recs)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for i, j in zip(*np.nonzero(m)):
        par[find(i)] = find(j)
    return {i: find(i) for i in range(len(recs))}


def detector_flags(paths: list[str], use_cache: bool) -> dict:
    if use_cache and DET_CACHE.exists():
        c = json.loads(DET_CACHE.read_text())
        if set(c["flags"]) >= set(paths):
            print(f"      dùng cache detector {DET_CACHE.relative_to(REPO_ROOT)}")
            return c
    import check_label_completeness_v10 as comp
    from ultralytics import YOLO
    model = YOLO(str(comp.WEIGHTS))
    rows = comp._scan(model, [REPO_ROOT / p for p in paths])
    flags = {str(Path(r["path"]).relative_to(REPO_ROOT)): {"n_unlabeled": len(r["missing"]),
                                                          "classes": r["missing_cls"]} for r in rows}
    c = {"weights": comp.WEIGHTS.name, "weights_sha256": sha256_file(comp.WEIGHTS),
         "params": {"conf": comp.CONF, "iou": comp.IOU_THR, "ioa": comp.IOA_THR, "min_area": comp.MIN_AREA},
         "flags": flags}
    DET_CACHE.parent.mkdir(parents=True, exist_ok=True)
    DET_CACHE.write_text(json.dumps(c, indent=1, ensure_ascii=False))
    return c


def select_v10(use_det_cache: bool) -> tuple[list[dict], list[dict], dict]:
    recs = load_records()
    print(f"[v10] {len(recs)} ảnh gốc")
    info = []
    for r in recs:
        rows, issues = read_v10_labels(r)
        info.append({"rec": r, "rows": rows, "issues": issues,
                     "classes": [int(x[0]) for x in rows if x and x[0].isdigit()]})

    excluded: list[dict] = []
    alive = list(range(len(recs)))

    def drop(idx_list, reason, extra=None):
        nonlocal alive
        s = set(idx_list)
        for i in idx_list:
            e = {"path": recs[i].path, "reason": reason}
            if extra and i in extra:
                e.update(extra[i])
            excluded.append(e)
        alive = [i for i in alive if i not in s]
        print(f"      {reason}: loại {len(s)} → còn {len(alive)}")

    # R4 nhãn lỗi / rỗng
    drop([i for i in alive if info[i]["issues"] or not info[i]["rows"]], "R4_label_invalid_or_empty")
    # R1 ba gác
    drop([i for i in alive if BA_GAC in info[i]["classes"]], "R1_contains_xe_ba_gac")
    # R2 dedup
    with ProcessPoolExecutor(max_workers=8) as ex:
        H = np.frombuffer(b"".join(ex.map(_phash, [str(REPO_ROOT / recs[i].path) for i in alive], chunksize=64)),
                          dtype=np.uint8).reshape(-1, 8)
    sub = [recs[i] for i in alive]
    root = dedup_groups(sub, H)
    groups = defaultdict(list)
    for k, i in enumerate(alive):
        groups[root[k]].append(i)
    dup, keep_of = [], {}
    for members in groups.values():
        if len(members) < 2:
            continue
        keep = sorted(members, key=lambda i: (-len(info[i]["rows"]), recs[i].path))[0]
        for i in members:
            if i != keep:
                dup.append(i)
                keep_of[i] = {"duplicate_of": recs[keep].path}
    drop(dup, "R2_duplicate", keep_of)
    # R3 detector
    det = detector_flags([recs[i].path for i in alive], use_det_cache)
    flagged = {i: {"n_unlabeled": det["flags"][recs[i].path]["n_unlabeled"],
                   "coco_classes": det["flags"][recs[i].path]["classes"]}
               for i in alive if det["flags"].get(recs[i].path, {}).get("n_unlabeled", 0) > 0}
    drop(list(flagged), "R3_unlabeled_vehicle_by_detector", flagged)

    kept = [{"rec": recs[i], "rows": info[i]["rows"], "overlay": bool(recs[i].vehicle_id)} for i in alive]
    meta = {"detector": {k: v for k, v in det.items() if k != "flags"},
            "n_raw": len(recs), "n_kept": len(kept)}
    return kept, excluded, meta


def pick_variant(kept: list[dict], variant: str) -> tuple[list[dict], dict]:
    if variant == "v5base":
        return [], {}
    if variant == "full":
        return kept, {}
    no = [k for k in kept if not k["overlay"]]
    if variant == "nooverlay":
        return no, {}
    # ovmatched: lấy mẫu ảnh CÓ overlay, khớp số lượng theo từng camera với `nooverlay`
    rng = random.Random(SEED)
    ov_by_cam = defaultdict(list)
    for k in kept:
        if k["overlay"]:
            ov_by_cam[k["rec"].camera_id].append(k)
    need = Counter(k["rec"].camera_id for k in no)
    chosen, shortfall = [], 0
    for cam in sorted(need):
        pool = sorted(ov_by_cam.get(cam, []), key=lambda k: k["rec"].path)
        take = min(need[cam], len(pool))
        chosen += rng.sample(pool, take)
        shortfall += need[cam] - take
    rest = sorted((k for k in kept if k["overlay"] and k not in chosen), key=lambda k: k["rec"].path)
    chosen += rng.sample(rest, shortfall)
    return chosen, {"target_n": len(no), "camera_matched": len(no) - shortfall, "filled_from_other_cameras": shortfall}


def remap(rows: list[list[str]]) -> list[str]:
    out = []
    for p in rows:
        c = CLASS_MAP[int(p[0])]
        assert c is not None  # R1 đã loại ba gác
        out.append(" ".join([str(c)] + p[1:5]))
    return out


def link(src: Path, dst: Path) -> None:
    dst.symlink_to(src.resolve())


def build_variant(variant: str, v10_sel: list[dict], extra: dict, meta: dict, excluded: list[dict],
                  val_policy: str) -> dict:
    dst = DATA_DIR / VARIANTS[variant]
    if dst.exists():
        shutil.rmtree(dst)
    for s in ("train", "val", "test"):
        (dst / "images" / s).mkdir(parents=True)
        (dst / "labels" / s).mkdir(parents=True)

    cls_count = {s: Counter() for s in ("train_v5", "train_v10", "val", "test")}
    n = {}
    for split in ("train", "val", "test"):
        items = v5_split_items(split, val_policy)
        for img, lbl in items:
            link(img, dst / "images" / split / img.name)
            link(lbl, dst / "labels" / split / lbl.name)
            key = "train_v5" if split == "train" else split
            cls_count[key].update(int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip())
        n[f"v5_{split}"] = len(items)

    v10_list = []
    for k in sorted(v10_sel, key=lambda k: k["rec"].path):
        src = REPO_ROOT / k["rec"].path
        stem = f"v10_{src.stem}"
        link(src, dst / "images" / "train" / f"{stem}{src.suffix.lower()}")
        lines = remap(k["rows"])
        (dst / "labels" / "train" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        cls_count["train_v10"].update(int(l.split()[0]) for l in lines)
        v10_list.append({"file": f"{stem}{src.suffix.lower()}", "source": k["rec"].path,
                         "camera_id": k["rec"].camera_id, "overlay": k["overlay"], "n_boxes": len(lines)})
    n["v10_train"] = len(v10_list)
    n["train_total"] = n["v5_train"] + n["v10_train"]

    yaml_path = DATA_DIR / ("data_v5_cleanval.yaml" if variant == "v5base" else
                            f"data_v5plus_clean{'' if variant == 'full' else '_' + variant}.yaml")
    yaml_path.write_text(
        f"# V5+clean [{variant}] — sinh bởi scripts/data_prep/build_v5plus_clean.py\n"
        f"# Test = V5 test nguyên vẹn; val = V5 val {'bỏ Cần Thơ (leak camera)' if val_policy == 'clean' else 'gốc'}. "
        f"Xem reports/v5plus_clean_datasheet.md\n"
        f"path: {dst}\ntrain: images/train\nval:   images/val\ntest:  images/test\n\nnc: 4\nnames:\n"
        + "".join(f"  {i}: {nm}\n" for i, nm in enumerate(V5_NAMES)))

    manifest = {
        "variant": variant, "built_at": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(), "builder": "scripts/data_prep/build_v5plus_clean.py",
        "dataset_root": str(dst.relative_to(REPO_ROOT)), "yaml": str(yaml_path.relative_to(REPO_ROOT)),
        "counts": n,
        "val_policy": val_policy,
        "val_excluded_prefixes": list(VAL_LEAKY_PREFIXES) if val_policy == "clean" else [],
        "v10_pct_of_train": round(100 * n["v10_train"] / n["train_total"], 2),
        "class_boxes": {k: {V5_NAMES[c]: v.get(c, 0) for c in range(4)} for k, v in cls_count.items()},
        "v10_overlay_in_train": sum(e["overlay"] for e in v10_list),
        "eval_fingerprint": eval_fingerprint(dst),
        "v5_eval_fingerprint_reference": eval_fingerprint(
            V5_ROOT, VAL_LEAKY_PREFIXES if val_policy == "clean" else ()),
        "rules": {"R1": "bỏ ảnh có xe_ba_gac", "R2": f"dedup cùng camera + pHash≤{DEDUP_PHASH} + (cùng biển số | Δt≤{DEDUP_DT_S}s)",
                  "R3": "bỏ ảnh có xe chưa gán nhãn theo detector COCO", "R4": "bỏ nhãn lỗi/rỗng"},
        "class_map_v10_to_v5": {V10_NAMES[k]: (V5_NAMES[v] if v is not None else "DROP_IMAGE") for k, v in CLASS_MAP.items()},
        "v10_selection": {**meta, **extra, "seed": SEED},
        "v10_files": v10_list,
    }
    if variant == "full":
        manifest["v10_excluded"] = excluded
        manifest["v10_excluded_by_reason"] = dict(Counter(e["reason"] for e in excluded))
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / f"manifest_{variant}.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    (dst / "split_manifest.json").write_text(json.dumps({k: manifest[k] for k in
                                                        ("variant", "counts", "eval_fingerprint", "rules")},
                                                       indent=1, ensure_ascii=False))
    print(f"[{variant}] train {n['train_total']} (V5 {n['v5_train']} + v10 {n['v10_train']}), "
          f"val {n['v5_val']}, test {n['v5_test']} → {dst.relative_to(REPO_ROOT)}")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--no-detector-cache", action="store_true", help="Chạy lại detector thay vì dùng cache")
    ap.add_argument("--val-policy", choices=["clean", "original"], default="clean",
                    help="clean: bỏ video Cần Thơ khỏi val (cùng camera với train); original: val V5 gốc")
    a = ap.parse_args()
    assert V5_ROOT.exists(), f"Thiếu {V5_ROOT}"
    print("[1/2] Lọc v10 theo R1–R4...")
    kept, excluded, meta = select_v10(not a.no_detector_cache)
    print(f"      v10 giữ {len(kept)} ảnh ({sum(k['overlay'] for k in kept)} có overlay, "
          f"{sum(not k['overlay'] for k in kept)} không overlay)")
    print("[2/2] Build biến thể...")
    for v in a.variants:
        sel, extra = pick_variant(kept, v)
        build_variant(v, sel, extra, meta, excluded, a.val_policy)
    print(f"\nManifest: {MANIFEST_DIR.relative_to(REPO_ROOT)}/manifest_*.json")
    print("Tiếp theo: python scripts/data_prep/verify_v5plus_clean.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
