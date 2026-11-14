"""Kiểm chứng ĐỘC LẬP các dataset V5+clean đã build + sinh datasheet.

Kiểm trên chính thư mục data/merged_v5plus_clean* (không tin vào builder):
  V1  Benchmark không đổi: fingerprint SHA-256 (tên ảnh + nội dung nhãn) test == V5 test;
      val == V5 val (đã bỏ video Cần Thơ nếu val_policy=clean) — tự tính lại từ merged_v5.
  V2  Val/test không chứa ảnh v10; mọi symlink hợp lệ; mỗi ảnh có đúng 1 file nhãn.
  V3  Nhãn hợp lệ (validate_labels.validate_split, nc=4), v10 không có ảnh nhãn rỗng.
  V4  Leak cấp video (audit_leak.audit_dataset): 0% frame val/test từ video có trong train.
  V5  Leak cấp nội dung: pHash train ↔ val/test, Hamming ≤ 5 → phải = 0 cặp (mọi nguồn).
  V8  Leak cấp CAMERA: ảnh nền trung bình mỗi video val/test vs mỗi video/camera train (NCC);
      phải < NCC nhỏ nhất của đối chứng dương (cùng video chia 2 nửa).
  V6  Không còn xe_ba_gac / box rỗng; class id ∈ [0,3].
  V7  Ablation: nooverlay toàn ảnh không-ID, ovmatched toàn ảnh có-ID, cùng kích thước & camera.

Output:
  reports/v5plus_clean/verification.json
  reports/v5plus_clean_datasheet.md

Usage:
    python scripts/data_prep/verify_v5plus_clean.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "audit_v10"))
from audit_leak import audit_dataset, extract_video_id  # noqa: E402
from check_cross_v5_leak_v10 import _features, _ncc  # noqa: E402
from build_v5plus_clean import (DATA_DIR, IMG_EXTS, MANIFEST_DIR, V5_NAMES, V5_ROOT, VARIANTS,  # noqa: E402
                                eval_fingerprint)
from check_perceptual_duplicates_v10 import _hamming_matrix, _phash  # noqa: E402
from validate_labels import validate_split  # noqa: E402
from v10_common import REPO_ROOT  # noqa: E402

OUT_JSON = MANIFEST_DIR / "verification.json"
OUT_MD = REPO_ROOT / "reports" / "v5plus_clean_datasheet.md"
PHASH_THR = 5


def _imgs(root: Path, split: str) -> list[Path]:
    return sorted(p for p in (root / "images" / split).iterdir() if p.suffix.lower() in IMG_EXTS)


def _hashes(paths: list[Path]) -> np.ndarray:
    with ProcessPoolExecutor(max_workers=8) as ex:
        hs = list(ex.map(_phash, [str(p.resolve()) for p in paths], chunksize=64))
    return np.frombuffer(b"".join(hs), dtype=np.uint8).reshape(-1, 8)


def verify_variant(variant: str) -> dict:
    root = DATA_DIR / VARIANTS[variant]
    r: dict = {"root": str(root.relative_to(REPO_ROOT))}
    man0 = json.loads((MANIFEST_DIR / f"manifest_{variant}.json").read_text())
    ref_fp = eval_fingerprint(V5_ROOT, tuple(man0.get("val_excluded_prefixes", [])))
    # V1
    fp = eval_fingerprint(root)
    r["V1_benchmark_unchanged"] = fp == ref_fp
    r["val_policy"] = man0.get("val_policy")
    r["eval_fingerprint"] = fp
    # V2
    bad_links, no_label, v10_in_eval = 0, 0, 0
    for s in ("train", "val", "test"):
        for p in _imgs(root, s):
            if not p.resolve().exists():
                bad_links += 1
            if not (root / "labels" / s / f"{p.stem}.txt").exists():
                no_label += 1
            if s != "train" and p.name.startswith("v10_"):
                v10_in_eval += 1
    r["V2_structure"] = {"ok": bad_links == 0 and no_label == 0 and v10_in_eval == 0,
                         "broken_symlinks": bad_links, "images_without_label": no_label,
                         "v10_in_val_test": v10_in_eval}
    # V3 + V6
    vl = {s: validate_split(root, s, 4) for s in ("train", "val", "test")}
    v10_empty, v10_cls = 0, Counter()
    for p in (root / "labels" / "train").glob("v10_*.txt"):
        lines = [l for l in p.read_text().splitlines() if l.strip()]
        v10_empty += not lines
        v10_cls.update(int(l.split()[0]) for l in lines)
    r["V3_labels"] = {"ok": all(v.get("status") == "ok" for v in vl.values()) and v10_empty == 0,
                      "per_split": {s: {k: v[k] for k in ("status", "n_paired", "n_empty_labels",
                                                           "n_files_with_issues", "class_distribution")}
                                    for s, v in vl.items()},
                      "v10_empty_labels": v10_empty}
    r["V6_v10_classes"] = {"ok": set(v10_cls) <= {0, 1, 2, 3},
                           "boxes": {V5_NAMES[c]: v10_cls.get(c, 0) for c in range(4)}}
    # V4
    a = audit_dataset(root)
    r["V4_video_leak"] = {"ok": a["leak_metrics"]["val"]["pct_leaked"] == 0 and a["leak_metrics"]["test"]["pct_leaked"] == 0,
                          "val_pct": a["leak_metrics"]["val"]["pct_leaked"],
                          "test_pct": a["leak_metrics"]["test"]["pct_leaked"],
                          "leaky_videos": a["leaky_videos"]}
    # V7
    man = json.loads((MANIFEST_DIR / f"manifest_{variant}.json").read_text())
    ov = [e["overlay"] for e in man["v10_files"]]
    r["v10_overlay_share"] = round(100 * sum(ov) / len(ov), 2) if ov else 0
    r["counts"] = man["counts"]
    r["v10_cameras"] = dict(Counter(e["camera_id"] for e in man["v10_files"]))
    return r


def phash_cross(root: Path) -> dict:
    tr = _imgs(root, "train")
    ev = _imgs(root, "val") + _imgs(root, "test")
    Ht, He = _hashes(tr), _hashes(ev)
    pairs = []
    min_v10 = 64
    for st in range(0, len(Ht), 512):
        D = _hamming_matrix(Ht[st:st + 512], He)
        for i, j in zip(*np.nonzero(D <= PHASH_THR)):
            pairs.append({"train": tr[st + i].name, "eval": ev[j].name, "hamming": int(D[i, j]),
                          "train_source": "v10" if tr[st + i].name.startswith("v10_") else "v5"})
        is_v10 = np.array([p.name.startswith("v10_") for p in tr[st:st + 512]])
        if is_v10.any():
            min_v10 = min(min_v10, int(D[is_v10].min()))
    by = Counter(p["train_source"] for p in pairs)
    return {"threshold": PHASH_THR, "n_pairs_v10_train": by.get("v10", 0), "n_pairs_v5_train": by.get("v5", 0),
            "min_hamming_v10_train_vs_eval": min_v10, "pairs_sample": pairs[:50],
            "ok": not pairs}


def _group_key(p: Path) -> str:
    """Nhóm theo camera: v10 → camera_id; V5 → video id (bỏ '_' thừa cuối tên)."""
    if p.name.startswith("v10_"):
        from v10_common import parse_name
        return "v10:" + (parse_name(p.stem[4:])["camera_id"] or "?")
    return extract_video_id(p.name).rstrip("_")


def camera_ncc(root: Path, per_group: int = 120) -> dict:
    import random
    groups: dict = {}
    for s in ("train", "val", "test"):
        for p in _imgs(root, s):
            groups.setdefault((s, _group_key(p)), []).append(str(p.resolve()))
    rng = random.Random(0)
    bg, half = {}, {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for k, ps in sorted(groups.items()):
            ps = rng.sample(ps, min(per_group, len(ps)))
            f = [x[2] for x in ex.map(_features, ps, chunksize=16)]
            bg[k] = np.mean(f, axis=0)
            if len(f) >= 20:
                h = len(f) // 2
                half[k] = _ncc(np.mean(f[:h], axis=0), np.mean(f[h:], axis=0))
    pos_min = min(half.values())
    train = [k for k in bg if k[0] == "train"]
    per_eval = {}
    for k in bg:
        if k[0] == "train":
            continue
        best = max(((round(_ncc(bg[k], bg[t]), 4), t[1]) for t in train))
        per_eval[f"{k[0]}:{k[1]}"] = {"max_ncc": best[0], "closest_train_group": best[1]}
    mx = max(v["max_ncc"] for v in per_eval.values())
    return {"positive_control_min": round(pos_min, 4), "max_cross_ncc": mx,
            "per_eval_video": per_eval, "ok": mx < pos_min}


def _icon(ok: bool) -> str:
    return "✅" if ok else "❌"


def datasheet(res: dict, manifests: dict, ovval: dict) -> str:
    full = manifests["full"]
    vs = list(VARIANTS)
    ex = full["v10_excluded_by_reason"]
    L = []
    w = L.append
    w("# Datasheet — V5+clean (V5 + vietnam_vehicles v10, đã làm sạch)\n")
    w(f"_Sinh tự động bởi [verify_v5plus_clean.py](../scripts/data_prep/verify_v5plus_clean.py) từ dataset build bởi "
      f"[build_v5plus_clean.py](../scripts/data_prep/build_v5plus_clean.py) (git `{full['git_head'][:7]}`, {full['built_at']})._\n")

    all_ok = all(v["ok"] for v in res["summary"].values())
    w("## Kết luận\n")
    w((f"✅ **Cả {len(vs)} biến thể qua toàn bộ kiểm tra — sẵn sàng train.**" if all_ok else
       "❌ **Có kiểm tra FAIL — KHÔNG train cho tới khi xử lý.**") + "\n")
    w("| Kiểm tra | " + " | ".join(f"`{v}`" for v in vs) + " |")
    w("|---|" + "---|" * len(vs))
    for key, label in (("V1_benchmark_unchanged", "V1 Test = V5 test; val = V5 val bỏ Cần Thơ (SHA-256)"),
                       ("V2_structure", "V2 Cấu trúc (symlink, nhãn, không v10 trong eval)"),
                       ("V3_labels", "V3 Nhãn hợp lệ (validate_labels)"),
                       ("V4_video_leak", "V4 Leak cấp video (audit_leak)"),
                       ("V6_v10_classes", "V6 Class v10 sau remap ∈ schema V5")):
        cells = []
        for v in VARIANTS:
            x = res["variants"][v][key]
            cells.append(_icon(x if isinstance(x, bool) else x["ok"]))
        w(f"| {label} | " + " | ".join(cells) + " |")
    pc = res["phash_cross_full"]
    cn = res["camera_ncc_full"]
    def row(label, cell_full, cells_other="⊂ `full`"):
        w(f"| {label} | " + " | ".join(cell_full if v == "full" else cells_other for v in vs) + " |")
    row(f"V5 Leak nội dung: pHash train ↔ val/test ≤{pc['threshold']}",
        f"{_icon(pc['ok'])} {pc['n_pairs_v5_train'] + pc['n_pairs_v10_train']} cặp")
    row("V8 Leak cấp camera: NCC ảnh nền val/test vs train < đối chứng dương",
        f"{_icon(cn['ok'])} {cn['max_cross_ncc']} < {cn['positive_control_min']}")
    w(f"| V7 Ablation hợp lệ (nooverlay toàn không-ID, ovmatched toàn có-ID, cùng cỡ & camera) | – | – | "
      f"{_icon(res['V7_ablation']['ok'])} | {_icon(res['V7_ablation']['ok'])} |")
    w("\n_Train của mọi biến thể là tập con của train `full` (V5 train + v10 đã lọc), val/test giống hệt nhau "
      "→ kết quả V5/V8 trên `full` áp dụng cho cả 4._\n")

    w("## 1. Mục đích\n")
    w("Bổ sung dữ liệu CCTV Việt Nam (Roboflow `visionvn/vietnam_vehicles` v10) vào train của V5 mà **không** làm "
      "sai benchmark: **test giữ nguyên V5** (SOICT24 cam_09/10, camera chưa thấy) để so sánh được với mọi kết quả "
      "trước; **val bỏ phần Cần Thơ** vì cùng camera với train (mục 3). Bản này sửa các vấn đề phát hiện trong audit "
      "([v10_audit_summary.md](v10_audit_summary.md)).\n")

    w("## 2. Thành phần\n")
    w("| Biến thể | YAML | Train (V5 + v10) | % v10 | v10 có overlay | Val | Test | Mục đích |")
    w("|---|---|---|---|---|---|---|---|")
    purpose = {"v5base": "Baseline V5 train lại cùng giao thức", "full": "**Bản chính**",
               "nooverlay": "Ablation: v10 không overlay",
               "ovmatched": "Đối chứng: v10 có overlay, cùng cỡ & camera"}
    for v in VARIANTS:
        m = manifests[v]
        c = m["counts"]
        w(f"| `{v}` | `{m['yaml']}` | {c['train_total']} ({c['v5_train']} + {c['v10_train']}) | {m['v10_pct_of_train']}% | "
          f"{res['variants'][v]['v10_overlay_share']}% | {c['v5_val']} | {c['v5_test']} | {purpose[v]} |")
    w("")
    w("Số box theo class:\n")
    w("| Phần | " + " | ".join(V5_NAMES) + " |")
    w("|---|" + "---|" * 4)
    rows = [("V5 train", full["class_boxes"]["train_v5"])]
    rows += [(f"v10 ({v})", manifests[v]["class_boxes"]["train_v10"]) for v in VARIANTS if v != "v5base"]
    rows += [("Val (V5, bỏ Cần Thơ)", full["class_boxes"]["val"]), ("Test (V5)", full["class_boxes"]["test"])]
    for name, cb in rows:
        w(f"| {name} | " + " | ".join(str(cb[n]) for n in V5_NAMES) + " |")
    w("")

    w("## 3. Sửa val: bỏ video Cần Thơ (leak cấp camera trong V5 gốc)\n")
    w("Kiểm tra pHash train↔eval phát hiện 102 cặp gần trùng giữa **V5 train** và **V5 val**, toàn bộ thuộc Cần Thơ. "
      "Kiểm bằng ảnh nền camera và bằng mắt: 4 video Cần Thơ (2 ở train: CoopMart 8h–8h30, 8h10–8h30; 2 ở val: "
      "CoopMart 8h30–9h00, camDLHB 13h20) là **cùng một camera** (NCC 0.98 / 0.76; đối chứng cùng video ≥ 0.73). "
      "`audit_leak.py` không bắt được vì so theo tên video, không theo camera.")
    w("- Hệ quả: 466/1466 frame val (32%) đến từ camera đã thấy khi train → val lạc quan, ảnh hưởng chọn "
      "checkpoint & early-stop. **Test (SOICT cam_09/10) không bị ảnh hưởng** (pHash gần nhất d=12, NCC ≤ 0.35).")
    w("- Sửa: val = `soict24_cam_07` (1000 frame, camera riêng). 466 frame Cần Thơ **không** chuyển sang train "
      "(giữ V5 train y nguyên). Test giữ nguyên tuyệt đối (fingerprint V1).")
    w("- Vì đổi val → cách chọn checkpoint khác các run cũ; do đó có biến thể `v5base` để **train lại baseline V5** "
      "theo đúng giao thức mới. So sánh khoa học là `v5base` vs `full` (cùng val, cùng test, cùng cấu hình).")
    w("- Kết quả cũ (V5 0.639, V5+ 0.747 mAP50 trên test) vẫn hợp lệ về test nhưng checkpoint được chọn bằng val có leak.\n")
    w("Video val/test và camera train gần nhất (NCC ảnh nền):\n")
    for k, v in sorted(res["camera_ncc_full"]["per_eval_video"].items()):
        w(f"- `{k}` → `{v['closest_train_group']}`: {v['max_ncc']}")
    w("")
    w("## 4. Quy tắc làm sạch v10 (cố định trước khi train)\n")
    w(f"Từ {full['v10_selection']['n_raw']} ảnh v10 → giữ **{full['v10_selection']['n_kept']}**.\n")
    w("| Quy tắc | Lý do | Ảnh loại |")
    w("|---|---|---|")
    w(f"| R4 nhãn lỗi/rỗng | an toàn | {ex.get('R4_label_invalid_or_empty', 0)} |")
    w(f"| R1 bỏ **cả ảnh** có `xe_ba_gac` | V5 không có class này. Bản V5+ cũ chỉ xoá box → ~110 xe ba gác thành "
      f"\"nền\" chưa gán nhãn (false negative có hệ thống) | {ex.get('R1_contains_xe_ba_gac', 0)} |")
    w(f"| R2 gộp ảnh trùng thật: cùng camera + pHash≤4 + (cùng biển số hoặc cách ≤2s) | tránh over-weight. **Không** "
      f"dùng pHash thuần: camera tĩnh → 6/8 cặp pHash≤2 kiểm bằng mắt là xe khác | {ex.get('R2_duplicate', 0)} |")
    det = full["v10_selection"]["detector"]
    w(f"| R3 bỏ ảnh có xe chưa gán nhãn theo detector COCO ({det['weights']}, conf≥{det['params']['conf']}) | giảm "
      f"false negative trong nhãn | {ex.get('R3_unlabeled_vehicle_by_detector', 0)} |")
    w("\nClass remap v10→V5: " + ", ".join(f"`{k}`→{v}" for k, v in full["class_map_v10_to_v5"].items()) + ".")
    w("Danh sách từng ảnh bị loại + lý do: `reports/v5plus_clean/manifest_full.json` (`v10_excluded`).\n")

    w("## 5. Overlay vẽ sẵn & thiết kế ablation\n")
    ho = ovval["holdout"]
    w(f"~83% ảnh v10 có tag/khung của hệ thống tracking gốc vẽ lên pixel, đặt ngay tại xe được gán nhãn → nguy cơ "
      f"model học lối tắt \"tag màu = xe\". Không làm sai số trên test V5 (V5 không có overlay) nhưng có thể làm "
      f"giảm lợi ích của v10.")
    w(f"- Quy tắc **tên file có vehicle ID ⇔ có overlay**: tune {ovval['tuning_sample']['rule_correct']}/"
      f"{ovval['tuning_sample']['n']}; **held-out mù {int(ho['accuracy'] * ho['n'])}/{ho['n']}** "
      f"(nhận định ghi trước khi mở đáp án — `reports/v5plus_clean/overlay_rule_validation.json`).")
    sel = manifests["ovmatched"]["v10_selection"]
    w(f"- `nooverlay` vs `ovmatched`: cùng V5 train, cùng số ảnh v10 ({sel['target_n']}), cùng phân bố camera "
      f"({sel['camera_matched']}/{sel['target_n']} khớp camera, seed {sel['seed']}) → khác biệt chủ yếu là overlay.")
    w("- Giới hạn: ảnh không-ID chụp 01–02/2024, ảnh có-ID cùng camera chụp 03–04/2024 → còn lệch thời điểm/thời tiết; "
      "phân bố class v10 hai bên cũng không trùng khít (bảng mục 2).\n")

    w("## 6. Giao thức train/đánh giá (đề xuất để kết quả có giá trị khoa học)\n")
    w("**Vì sao phải train lại cả baseline:** so sánh cũ V5 0.639 → V5+ 0.747 mAP50 **không quy được cho v10** — "
      "hai run khác nhau đồng thời ở imgsz (640→960), tổng epoch (40→80 do warm-start), augmentation "
      "(mixup/copy_paste/xoay), lr0 và lịch lr. Riêng imgsz 960 đã có thể tăng mạnh AP xe máy nhỏ.\n")
    w("1. **Một recipe chung cho mọi biến thể**, chỉ đổi dữ liệu: init COCO `yolov8s.pt`, imgsz 960, 60 epoch, "
      "không early-stop, cùng augmentation — "
      "[v5plus_clean_train.py](../scripts/training/v5plus_clean_train.py) (`RECIPE`). Không warm-start từ "
      "`train_v8s_v5_honest/best.pt` vì checkpoint đó được chọn bằng val có leak.")
    w("2. **Chọn checkpoint chỉ bằng val sạch** (cam_07); script chạy **test V5 đúng một lần** sau khi train xong "
      "và lưu `reports/v5plus_clean/results/<run>.json`. Không chỉnh quy tắc làm sạch/recipe sau khi nhìn test.")
    w("3. Thứ tự: (a) `v5base` và `full` seed 42 — cặp so sánh chính. (b) `nooverlay` và `ovmatched` seed 42. "
      "(c) Nếu chênh lệch nhỏ (<~1 điểm mAP50) → thêm seed 0 và 1 (`--seed 42 0 1`), báo trung bình ± độ lệch "
      "chuẩn (`--summarize`); một seed không đủ để kết luận khác biệt nhỏ.")
    w("   - Số epoch bằng nhau ⇒ `full` có nhiều iteration hơn `v5base` (2.5× ảnh). Đây là quy ước phổ biến; "
      "nếu cần tách hiệu ứng \"nhiều bước tối ưu hơn\" thì chạy thêm `v5base` với 150 epoch.")
    w("4. Báo cáo mAP50, mAP50-95 và AP theo class; với ablation overlay báo cả kích thước dữ liệu.\n")

    w("## 7. Hạn chế đã biết\n")
    w("- v10 vẫn chiếm ~59% ảnh train bản `full`, khác domain (snapshot vi phạm, 1–3 xe/ảnh) so với test SOICT "
      "(mật độ cao, ~33 box/ảnh).")
    w("- R3 dùng detector COCO → chỉ bắt xe chưa gán mà COCO thấy (cận dưới); không áp cho V5 train (giữ nguyên "
      "bản đã dùng ở mọi kết quả trước để so sánh được).")
    w("- 83% ảnh trong `full` vẫn có overlay; ablation ở mục 4 dùng để đo ảnh hưởng thay vì đoán.")
    w("- `data/merged_v5plus` (bản cũ) được giữ nguyên làm baseline tái lập.\n")

    w("## 8. Tái lập\n")
    w("```bash\nconda activate yolov8_ft\npython scripts/data_prep/build_v5plus_clean.py\n"
      "python scripts/data_prep/verify_v5plus_clean.py\n```")
    w("Manifest từng biến thể (nguồn, camera, overlay của từng ảnh v10; fingerprint val/test): "
      "`reports/v5plus_clean/manifest_{full,nooverlay,ovmatched}.json`.")
    return "\n".join(L) + "\n"


def main() -> int:
    res = {"variants": {}, "summary": {}}
    for v in VARIANTS:
        print(f"[verify] {v}...")
        res["variants"][v] = verify_variant(v)
    print("[verify] pHash train ↔ val/test (full)...")
    res["phash_cross_full"] = phash_cross(DATA_DIR / VARIANTS["full"])
    print("[verify] NCC ảnh nền camera (full)...")
    res["camera_ncc_full"] = camera_ncc(DATA_DIR / VARIANTS["full"])

    manifests = {v: json.loads((MANIFEST_DIR / f"manifest_{v}.json").read_text()) for v in VARIANTS}
    full_ids = {e["file"] for e in manifests["full"]["v10_files"]}
    no = manifests["nooverlay"]["v10_files"]
    ov = manifests["ovmatched"]["v10_files"]
    res["V7_ablation"] = {
        "ok": (all(not e["overlay"] for e in no) and all(e["overlay"] for e in ov) and len(no) == len(ov)
               and Counter(e["camera_id"] for e in no) == Counter(e["camera_id"] for e in ov)
               and {e["file"] for e in no + ov} <= full_ids),
        "n_nooverlay": len(no), "n_ovmatched": len(ov)}

    for v, r in res["variants"].items():
        res["summary"][f"{v}_V1"] = {"ok": r["V1_benchmark_unchanged"]}
        for k in ("V2_structure", "V3_labels", "V4_video_leak", "V6_v10_classes"):
            res["summary"][f"{v}_{k[:2]}"] = {"ok": r[k]["ok"]}
    res["summary"]["V5_phash"] = {"ok": res["phash_cross_full"]["ok"]}
    res["summary"]["V8_camera_ncc"] = {"ok": res["camera_ncc_full"]["ok"]}
    res["summary"]["V7_ablation"] = {"ok": res["V7_ablation"]["ok"]}

    OUT_JSON.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    ovval = json.loads((MANIFEST_DIR / "overlay_rule_validation.json").read_text())
    OUT_MD.write_text(datasheet(res, manifests, ovval))
    for k, v in res["summary"].items():
        print(f"  {_icon(v['ok'])} {k}")
    print(f"[OK] {OUT_JSON.relative_to(REPO_ROOT)}\n[OK] {OUT_MD.relative_to(REPO_ROOT)}")
    return 0 if all(v["ok"] for v in res["summary"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
