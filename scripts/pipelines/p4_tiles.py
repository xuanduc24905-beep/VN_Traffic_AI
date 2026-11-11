"""Pipeline 4 (một phần): huấn luyện trên ô cắt từ ảnh lớn, để xe máy xa chiếm tỉ lệ lớn hơn.

Mỗi ảnh lớn hơn kích thước ô (mặc định 640) được cắt thành các ô chồng lấn. Nhãn được chuyển
sang toạ độ ô; hộp chỉ giữ nếu phần nhìn thấy trong ô >= --min-visible (mặc định 50%) diện tích
hộp gốc. Ảnh nhỏ hơn ô (vd ảnh Cần Thơ 416x416) không bị cắt.

val và test KHÔNG cắt: yaml dùng nguyên val/test của tập gốc, nên mô hình huấn luyện trên ô vẫn
được đo trên ảnh đầy đủ, cùng điều kiện với các mô hình khác.

Ví dụ:
  python scripts/pipelines/p4_tiles.py --src data/merged_v5 --out data/merged_v5_tiles
  python scripts/pipelines/train.py --data data/merged_v5_tiles/tiles.yaml --name tiles --recipe v5plus

Mặc định train gồm cả ảnh đầy đủ của tập gốc lẫn các ô (--no-full để chỉ dùng ô).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (box_stats, label_path_for, list_images, place, read_yolo,  # noqa: E402
                    write_multi_yaml, write_yolo)
from no_retrain_experiments import tiles as tile_grid  # noqa: E402


def tile_labels(rows, W, H, tile_box, min_visible=0.5, min_px=4):
    """rows [n,5] chuẩn hoá trên ảnh WxH -> nhãn trong ô tile_box=(x0,y0,x1,y1). Trả về [m,5] chuẩn hoá theo ô."""
    x0, y0, x1, y1 = tile_box
    tw, th = x1 - x0, y1 - y0
    out = []
    for c, xc, yc, w, h in rows:
        bx1, by1 = (xc - w / 2) * W, (yc - h / 2) * H
        bx2, by2 = (xc + w / 2) * W, (yc + h / 2) * H
        ix1, iy1, ix2, iy2 = max(bx1, x0), max(by1, y0), min(bx2, x1), min(by2, y1)
        iw, ih = ix2 - ix1, iy2 - iy1
        if iw < min_px or ih < min_px:
            continue
        area = max((bx2 - bx1) * (by2 - by1), 1e-9)
        if (iw * ih) / area < min_visible:
            continue
        out.append([c, ((ix1 + ix2) / 2 - x0) / tw, ((iy1 + iy2) / 2 - y0) / th, iw / tw, ih / th])
    return np.array(out, dtype=np.float64).reshape(-1, 5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="tập gốc có images/{train,val,test}")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--min-visible", type=float, default=0.5)
    ap.add_argument("--keep-empty", type=float, default=0.1, help="xác suất giữ ô không có hộp nào")
    ap.add_argument("--no-full", action="store_true", help="train chỉ gồm ô, không kèm ảnh đầy đủ của tập gốc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--link", action="store_true")
    a = ap.parse_args()

    src_train = a.src / "images" / "train"
    if not src_train.exists():
        ap.error(f"không thấy {src_train}")
    rng = random.Random(a.seed)
    (a.out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (a.out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    n_img_tiled = n_img_small = n_tiles = n_empty_kept = 0

    for img_p in list_images(src_train):
        lp = label_path_for(img_p)
        rows = read_yolo(lp)
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        H, W = img.shape[:2]
        grid = tile_grid(W, H, a.tile, a.overlap)
        if not grid:                                   # ảnh nhỏ hơn ô: không cắt
            n_img_small += 1
            if a.no_full:                              # nếu không kèm ảnh đầy đủ thì phải chép sang để không mất
                place(img_p, a.out / "images" / "train" / img_p.name, a.link)
                write_yolo(a.out / "labels" / "train" / (img_p.stem + ".txt"), rows)
            continue
        n_img_tiled += 1
        for k, tb in enumerate(grid):
            tr = tile_labels(rows, W, H, tb, a.min_visible)
            if len(tr) == 0:
                if rng.random() >= a.keep_empty:
                    continue
                n_empty_kept += 1
            x0, y0, x1, y1 = tb
            name = f"{img_p.stem}_t{k}"
            cv2.imwrite(str(a.out / "images" / "train" / (name + ".jpg")), img[y0:y1, x0:x1],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_yolo(a.out / "labels" / "train" / (name + ".txt"), tr)
            n_tiles += 1

    train_dirs = [a.out / "images" / "train"]
    if not a.no_full:
        train_dirs = [src_train] + train_dirs
    val = a.src / "images" / "val"
    test = a.src / "images" / "test"
    write_multi_yaml(a.out / "tiles.yaml", train_dirs, val, test if test.exists() else None,
                     f"Train: {'ảnh đầy đủ + ' if not a.no_full else ''}ô {a.tile}px; val/test: ảnh đầy đủ của tập gốc")
    full = box_stats(a.src / "labels" / "train")
    til = box_stats(a.out / "labels" / "train")
    summ = {"images_tiled": n_img_tiled, "images_not_tiled": n_img_small, "tiles_written": n_tiles,
            "empty_tiles_kept": n_empty_kept, "small_box_share_full": full["small_box_share"],
            "small_box_share_tiles": til["small_box_share"], "tile_boxes": til["boxes"],
            "tile_boxes_per_class": til["per_class"]}
    (a.out / "tiles_manifest.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=2))
    print("[ghi chú] 'small_box_share' tính theo diện tích chuẩn hoá < 0,25%: trên ô, cùng một xe chiếm "
          "tỉ lệ lớn hơn nên tỉ lệ hộp nhỏ giảm - đó là mục đích của việc cắt ô.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
