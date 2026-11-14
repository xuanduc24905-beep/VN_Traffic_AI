"""(f) Kiểm tra toàn vẹn nhãn + phân phối class của v10 standalone.

Tái sử dụng `check_label_file` từ scripts/data_prep/validate_labels.py
(format 5 giá trị, class_id ∈ [0,nc-1], toạ độ ∈ [0,1], bbox không suy biến,
không vượt biên). Không dùng `validate_split` vì nó giả định layout
images/<split>/ còn v10 là <split>/images/.

Mất cân bằng class: với mỗi class, so tỉ lệ % box ở valid/test với train.
Lệch > 15 điểm phần trăm → FLAG (phân phối eval không đại diện cho train).
Ngoài ra báo tỉ lệ class lớn nhất / nhỏ nhất (imbalance nội tại).

Verdict:
- integrity: PASS nếu 0 file lỗi, 0 ảnh thiếu nhãn, 0 nhãn thiếu ảnh.
- distribution: WARN nếu có class lệch > 15pp giữa các split.

Usage:
    python scripts/data_prep/audit_v10/check_label_integrity_v10.py
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from v10_common import IMG_EXTS, REPORTS_DIR, SPLITS, V10_ROOT, write_json  # noqa: E402
from validate_labels import check_label_file  # noqa: E402

OUT = REPORTS_DIR / "v10_label_integrity.json"
SHIFT_PP = 15.0


def run(root: Path = V10_ROOT, out: Path = OUT, shift_pp: float = SHIFT_PP) -> dict:
    meta = yaml.safe_load((root / "data.yaml").read_text())
    nc, names = int(meta["nc"]), list(meta["names"])

    splits = {}
    for s in SPLITS:
        img_dir, lbl_dir = root / s / "images", root / s / "labels"
        imgs = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
        lbls = {p.stem for p in lbl_dir.iterdir() if p.suffix == ".txt"}
        cls_box, cls_img, issues, n_empty, n_boxes_per_img = Counter(), Counter(), {}, 0, []
        for stem in sorted(imgs & lbls):
            lp = lbl_dir / f"{stem}.txt"
            iss = check_label_file(lp, nc)
            if iss == ["__empty__"]:
                n_empty += 1
                n_boxes_per_img.append(0)
                continue
            if iss:
                issues[stem] = iss
            ids = [int(l.split()[0]) for l in lp.read_text().splitlines()
                   if l.strip() and l.split()[0].isdigit()]
            n_boxes_per_img.append(len(ids))
            cls_box.update(ids)
            cls_img.update(set(ids))
        total = sum(cls_box.values())
        splits[s] = {
            "n_images": len(imgs), "n_labels": len(lbls),
            "images_without_label": len(imgs - lbls), "labels_without_image": len(lbls - imgs),
            "n_empty_labels": n_empty, "n_files_with_issues": len(issues),
            "file_issues_sample": dict(list(issues.items())[:20]),
            "n_boxes": total,
            "mean_boxes_per_image": round(total / max(len(n_boxes_per_img), 1), 2),
            "class_boxes": {names[k]: cls_box.get(k, 0) for k in range(nc)},
            "class_pct": {names[k]: round(100 * cls_box.get(k, 0) / total, 2) if total else 0 for k in range(nc)},
            "class_images": {names[k]: cls_img.get(k, 0) for k in range(nc)},
        }

    # Lệch phân phối giữa split
    shifts = []
    for s in ("valid", "test"):
        for c in names:
            d = splits[s]["class_pct"][c] - splits["train"]["class_pct"][c]
            if abs(d) > shift_pp:
                shifts.append({"split": s, "class": c, "train_pct": splits["train"]["class_pct"][c],
                               "split_pct": splits[s]["class_pct"][c], "delta_pp": round(d, 2)})
    # Imbalance nội tại (toàn bộ)
    all_box = Counter()
    for s in SPLITS:
        all_box.update(splits[s]["class_boxes"])
    nz = [v for v in all_box.values() if v > 0]
    missing_in_eval = [{"split": s, "class": c} for s in ("valid", "test")
                       for c in names if splits[s]["class_boxes"][c] == 0 and all_box[c] > 0]

    integ_ok = all(v["n_files_with_issues"] == 0 and v["images_without_label"] == 0
                   and v["labels_without_image"] == 0 for v in splits.values())
    result = {
        "check": "label_integrity",
        "threshold": f"0 lỗi nhãn; lệch tỉ lệ class giữa split ≤ {shift_pp}pp (WARN)",
        "verdict": "PASS" if integ_ok else "FAIL",
        "distribution_verdict": "PASS" if not shifts and not missing_in_eval else "WARN",
        "nc": nc, "names": names,
        "splits": splits,
        "class_shift_flags": shifts,
        "classes_missing_in_eval": missing_in_eval,
        "overall_class_boxes": dict(all_box),
        "overall_max_min_ratio": round(max(nz) / min(nz), 1) if nz else None,
    }
    write_json(result, out)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=V10_ROOT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--shift-pp", type=float, default=SHIFT_PP)
    a = ap.parse_args()
    r = run(a.root, a.out, a.shift_pp)
    print(f"[{r['verdict']}] integrity | [{r['distribution_verdict']}] distribution: "
          f"{len(r['class_shift_flags'])} class lệch >{a.shift_pp}pp")
    for s in SPLITS:
        print(f"  {s:5s}: {r['splits'][s]['class_pct']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
