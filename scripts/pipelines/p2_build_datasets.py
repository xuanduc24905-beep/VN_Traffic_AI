"""Pipeline 2 (và dựng dataset cho pipeline 1, 3): ghép tập gốc với dữ liệu thêm, cùng ngân sách.

Mỗi "arm" là một file dataset yaml. Tập gốc (base) KHÔNG bị chép lại: yaml liệt kê thư mục
train của base và thư mục train của phần thêm. val và test luôn là của base nên mọi arm đo trên
cùng một tập. Chỉ ảnh của phần thêm được chép vào <out>/<arm>/.

  base   : chỉ tập gốc (đối chứng, không thêm gì)
  hand   : base + N ảnh nhãn tay (vd bộ vietnam_vehicles v10 của Bình Định)
  pseudo : base + N ảnh nhãn giả (ra từ p1_pseudo_small.py)

Ngân sách bằng nhau là điều kiện của pipeline 2: --budget images (cùng số ảnh) hoặc
--budget boxes (cùng tổng số hộp; -n là số hộp). In kèm thống kê để thấy lệch lớp và tỉ lệ hộp nhỏ.

Ví dụ:
  # pipeline 2: nhãn tay và nhãn giả, cùng 3000 ảnh
  python scripts/pipelines/p2_build_datasets.py --base data/merged_v5 --out data/exp2 \\
      --hand data/raw_vietnam_vehicles_v10 --remap-v10 --pseudo data/pseudo_mask \\
      --arms base,hand,pseudo --budget images -n 3000
  # pipeline 1 hoặc 3: chỉ base + nhãn giả
  python scripts/pipelines/p2_build_datasets.py --base data/merged_v5 --out data/exp1 \\
      --pseudo data/pseudo_mask --arms pseudo -n 100000

--hand/--pseudo là thư mục YOLO chứa images/ và labels/ (cấu trúc con tuỳ ý, ảnh nằm dưới một
thư mục tên 'images'). Với bộ Roboflow thô (train/images, valid/images, test/images) thêm --remap-v10.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (box_stats, label_path_for, list_images, place, read_yolo,  # noqa: E402
                    write_multi_yaml, write_yolo)

# Roboflow vietnam_vehicles v10 (6 lớp) -> 4 lớp của dự án (giống scripts/data_prep/build_v5plus_merged.py)
V10_MAP = {0: 3, 1: 1, 2: None, 3: 2, 4: 0, 5: 3}   # container, oto, xe_ba_gac, xe_khach, xe_may, xe_tai


def collect_pairs(root: Path, remap_v10=False, keep_dropped=False):
    """Trả về list (ảnh, mảng nhãn [n,5]). Bỏ ảnh không có file nhãn."""
    out, dropped_imgs = [], 0
    for img in list_images(root):
        try:
            lp = label_path_for(img)
        except ValueError:
            continue
        if not lp.exists():
            continue
        rows = read_yolo(lp)
        if remap_v10:
            new, has_drop = [], False
            for r in rows:
                m = V10_MAP.get(int(r[0]), None)
                if m is None:
                    has_drop = True
                    continue
                new.append([m, *r[1:]])
            if has_drop and not keep_dropped:
                dropped_imgs += 1
                continue
            rows = np.array(new, dtype=np.float64).reshape(-1, 5)
            if len(rows) == 0:
                dropped_imgs += 1
                continue
        out.append((img, rows))
    return out, dropped_imgs


def sample_budget(pairs, n, budget, seed):
    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    if budget == "images":
        return [pairs[i] for i in idx[:n]]
    chosen, total = [], 0
    for i in idx:
        if total >= n:
            break
        chosen.append(pairs[i])
        total += len(pairs[i][1])
    return chosen


def build_extra(chosen, dst_root: Path, prefix: str, link: bool):
    for img, rows in chosen:
        name = f"{prefix}_{img.stem}"
        place(img, dst_root / "images" / "train" / (name + img.suffix.lower()), link)
        write_yolo(dst_root / "labels" / "train" / (name + ".txt"), rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True, help="tập gốc có images/{train,val,test}, vd data/merged_v5")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hand", type=Path, default=None)
    ap.add_argument("--pseudo", type=Path, default=None)
    ap.add_argument("--remap-v10", action="store_true", help="--hand là Roboflow v10 6 lớp, đổi về 4 lớp")
    ap.add_argument("--keep-images-with-dropped", action="store_true",
                    help="giữ ảnh có xe ba gác (mặc định BỎ ảnh đó để ba gác không bị học thành nền)")
    ap.add_argument("--arms", default="base,hand,pseudo", help="danh sách arm, cách nhau dấu phẩy")
    ap.add_argument("--budget", choices=["images", "boxes"], default="images")
    ap.add_argument("-n", type=int, default=3000, help="số ảnh (hoặc số hộp nếu --budget boxes) mỗi arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--link", action="store_true", help="symlink thay vì chép (không dùng trên Drive)")
    a = ap.parse_args()

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    base_train = a.base / "images" / "train"
    base_val, base_test = a.base / "images" / "val", a.base / "images" / "test"
    for d in (base_train, base_val):
        if not d.exists():
            ap.error(f"không thấy {d}")
    a.out.mkdir(parents=True, exist_ok=True)

    manifest = {"args": {k: str(v) for k, v in vars(a).items()}, "arms": {}}
    base_stats = box_stats(a.base / "labels" / "train")
    print(f"[base] {base_stats}")

    for arm in arms:
        yaml_path = a.out / f"{arm}.yaml"
        if arm == "base":
            write_multi_yaml(yaml_path, [base_train], base_val, base_test if base_test.exists() else None,
                             "Đối chứng: chỉ tập gốc")
            manifest["arms"][arm] = {"train": base_stats}
            print(f"[base] {yaml_path}")
            continue
        src = {"hand": a.hand, "pseudo": a.pseudo}.get(arm)
        if arm not in ("hand", "pseudo"):
            ap.error(f"arm không hợp lệ: {arm} (dùng base, hand, pseudo)")
        if src is None:
            ap.error(f"arm '{arm}' cần --{arm}")
        pairs, dropped = collect_pairs(src, a.remap_v10 and arm == "hand", a.keep_images_with_dropped)
        if not pairs:
            ap.error(f"không tìm thấy cặp ảnh-nhãn nào trong {src}")
        chosen = sample_budget(pairs, a.n, a.budget, a.seed)
        if a.budget == "images" and len(chosen) < a.n:
            print(f"[cảnh báo] arm {arm}: chỉ có {len(chosen)} ảnh < {a.n}; các arm sẽ KHÔNG cùng ngân sách")
        dst = a.out / arm
        build_extra(chosen, dst, arm, a.link)
        write_multi_yaml(yaml_path, [base_train, dst / "images" / "train"], base_val,
                         base_test if base_test.exists() else None,
                         f"Arm {arm}: base + {len(chosen)} ảnh từ {src}")
        st = box_stats(dst / "labels" / "train")
        manifest["arms"][arm] = {"extra": st, "pool": len(pairs), "dropped_images": dropped}
        print(f"[{arm}] pool {len(pairs)} ảnh (loại {dropped} ảnh có lớp bị bỏ) -> lấy {st['images']} ảnh, "
              f"{st['boxes']} hộp, {st['boxes_per_image']} hộp/ảnh, hộp nhỏ {st['small_box_share']:.1%}, "
              f"theo lớp {st['per_class']}")
        print(f"        {yaml_path}")

    (a.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[xong] {a.out}. Huấn luyện từng arm bằng scripts/pipelines/train.py --data {a.out}/<arm>.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
