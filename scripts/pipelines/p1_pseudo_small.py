"""Pipeline 1 (và 3): sinh nhãn giả có kiểm soát hộp nhỏ.

Giáo viên dự đoán trên ảnh không nhãn. Ba chế độ xử lý hộp nhỏ, để so riêng tác dụng của chúng:

  drop : giống V6. Giữ hộp có conf >= ngưỡng lớp VÀ diện tích >= --small-frac (mặc định 0,25%
         diện tích ảnh). Hộp nhỏ hơn bị bỏ, nên xe máy xa bị học thành NỀN.
  keep : không lọc theo kích thước. Hộp nhỏ cũng chỉ cần conf >= ngưỡng (hoặc --small-thr).
  mask : như keep, và thêm: hộp "nghi ngờ" (conf từ --low-thr đến dưới ngưỡng, diện tích
         <= --mask-max-frac) bị TÔ XÁM trong ảnh lưu ra, để mô hình không học chúng thành nền.
         Hộp nghi ngờ trùng một hộp đã giữ (IoU > 0,3 hoặc nằm trọn trong nó) thì không tô.

Ảnh không còn hộp nào sau lọc bị loại. Có nhiều --teacher thì ghép bằng WBF (pipeline 3).

Ví dụ:
  python scripts/pipelines/p1_pseudo_small.py --images data/raw/frames/hcm \\
      --teacher runs/detect/train_v8s_v5plus_960/weights/best.pt \\
      --mode mask --out data/pseudo_mask --device 0
  # giáo viên ensemble (pipeline 3):
  ... --teacher .../v5plus/best.pt --teacher .../v6/best.pt --mode mask
  # chỉ đếm ảnh sẽ được xét, không chạy mô hình:
  ... --dry-run

Ra: <out>/images/train, <out>/labels/train, <out>/manifest.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (NAME_TO_ID, NAMES, camera_id, list_images, load_camera_set,  # noqa: E402
                    load_models, place, teacher_predict, write_yolo)

DEFAULT_THR = {"motorcycle": 0.60, "car": 0.45, "bus": 0.50, "truck": 0.50}  # như pipeline round 1
GRAY = 114


def parse_thr(s: str) -> dict[int, float]:
    out = {NAME_TO_ID[k]: v for k, v in DEFAULT_THR.items()}
    if s:
        for kv in s.split(","):
            k, _, v = kv.partition("=")
            if k.strip() not in NAME_TO_ID:
                raise SystemExit(f"--thr: lớp không hợp lệ '{k}'. Dùng: {', '.join(NAME_TO_ID)}")
            out[NAME_TO_ID[k.strip()]] = float(v)
    return out


def _iou_inside(a, b):
    """IoU của a [4] với từng hộp trong b [n,4]; và tỉ lệ diện tích a nằm trong từng b."""
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa + ab - inter, 1e-9), inter / aa


def select_boxes(xyxy, conf, cls, w, h, mode, thr, small_frac=0.0025, small_thr=None,
                 low_thr=0.15, mask_max_frac=0.01):
    """Trả về (rows [k,5] nhãn YOLO giữ lại, mask_boxes [m,4] pixel cần tô xám, thống kê)."""
    n = len(conf)
    if n == 0:
        return np.zeros((0, 5)), np.zeros((0, 4)), {"small_kept": 0, "masked": 0}
    frac = ((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])) / float(w * h)
    need = np.array([thr.get(int(c), 0.5) for c in cls])
    is_small = frac < small_frac
    if mode != "drop" and small_thr is not None:
        need = np.where(is_small, small_thr, need)
    keep = conf >= need
    if mode == "drop":
        keep &= ~is_small
    kept_xyxy = xyxy[keep]
    rows = np.zeros((int(keep.sum()), 5))
    if len(rows):
        b = kept_xyxy
        rows[:, 0] = cls[keep]
        rows[:, 1] = ((b[:, 0] + b[:, 2]) / 2) / w
        rows[:, 2] = ((b[:, 1] + b[:, 3]) / 2) / h
        rows[:, 3] = (b[:, 2] - b[:, 0]) / w
        rows[:, 4] = (b[:, 3] - b[:, 1]) / h
    mask_boxes = []
    if mode == "mask":
        unsure = (~keep) & (conf >= low_thr) & (frac <= mask_max_frac)
        for i in np.where(unsure)[0]:
            if len(kept_xyxy):
                iou, inside = _iou_inside(xyxy[i], kept_xyxy)
                if (iou > 0.3).any() or (inside > 0.5).any():
                    continue
            mask_boxes.append(xyxy[i])
    mb = np.array(mask_boxes, dtype=np.float32).reshape(-1, 4)
    return rows, mb, {"small_kept": int((keep & is_small).sum()), "masked": len(mb)}


def apply_masks(img, mask_boxes, pad=2):
    h, w = img.shape[:2]
    for x1, y1, x2, y2 in mask_boxes:
        a, b = max(int(x1) - pad, 0), max(int(y1) - pad, 0)
        c, d = min(int(x2) + pad, w), min(int(y2) + pad, h)
        img[b:d, a:c] = GRAY
    return img


def generate(images, models, out, mode="mask", thr=None, small_frac=0.0025, small_thr=None,
             low_thr=0.15, mask_max_frac=0.01, imgsz=960, tta=False, device=None, link=False,
             progress_every=200):
    thr = thr or parse_thr("")
    out = Path(out)
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    stats = {"images_in": len(images), "images_kept": 0, "boxes_kept": 0, "small_kept": 0,
             "masked_boxes": 0, "per_class": {v: 0 for v in NAMES.values()}}
    t0 = time.time()
    used = set()
    for i, p in enumerate(images):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        xyxy, conf, cls = teacher_predict(models, img, imgsz=imgsz, conf_floor=min(low_thr, 0.10),
                                          device=device, tta=tta)
        rows, mb, st = select_boxes(xyxy, conf, cls, w, h, mode, thr, small_frac, small_thr,
                                    low_thr, mask_max_frac)
        if len(rows) == 0:
            continue
        stem = Path(p).stem
        if stem in used:                      # trùng tên ở thư mục con khác: thêm tên thư mục cha
            stem = f"{Path(p).parent.name}_{stem}"
        used.add(stem)
        dst_img = out / "images" / "train" / (stem + ".jpg")
        if len(mb):
            cv2.imwrite(str(dst_img), apply_masks(img, mb), [cv2.IMWRITE_JPEG_QUALITY, 95])
        elif Path(p).suffix.lower() in (".jpg", ".jpeg"):
            place(Path(p), dst_img, link)
        else:
            cv2.imwrite(str(dst_img), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        write_yolo(out / "labels" / "train" / (stem + ".txt"), rows)
        stats["images_kept"] += 1
        stats["boxes_kept"] += len(rows)
        stats["small_kept"] += st["small_kept"]
        stats["masked_boxes"] += st["masked"]
        for c in rows[:, 0].astype(int):
            stats["per_class"][NAMES.get(int(c), str(int(c)))] += 1
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  {i + 1}/{len(images)} ảnh, giữ {stats['images_kept']}, "
                  f"{(time.time() - t0) / (i + 1):.2f} s/ảnh", flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True, help="thư mục ảnh không nhãn (tìm đệ quy)")
    ap.add_argument("--teacher", action="append", default=[], help="best.pt giáo viên; lặp lại để ghép WBF")
    ap.add_argument("--mode", choices=["drop", "keep", "mask"], default="mask")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--thr", default="", help="ngưỡng theo lớp, vd motorcycle=0.6,car=0.45,bus=0.5,truck=0.5")
    ap.add_argument("--small-frac", type=float, default=0.0025, help="diện tích hộp/ảnh nhỏ hơn mức này là hộp nhỏ")
    ap.add_argument("--small-thr", type=float, default=None, help="ngưỡng riêng cho hộp nhỏ (chế độ keep, mask)")
    ap.add_argument("--low-thr", type=float, default=0.15, help="conf tối thiểu của hộp nghi ngờ (chế độ mask)")
    ap.add_argument("--mask-max-frac", type=float, default=0.01, help="chỉ tô hộp nghi ngờ nhỏ hơn mức này")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--tta", action="store_true", help="bật TTA khi sinh nhãn (chậm hơn)")
    ap.add_argument("--blacklist", default=None, help="yaml/danh sách camera bị loại (configs/hcm_cameras_blacklist.yaml)")
    ap.add_argument("--exclude-cameras", default=None, help="camera của tập test: danh sách, hoặc file mỗi dòng một mã")
    ap.add_argument("--limit", type=int, default=None, help="lấy ngẫu nhiên N ảnh (seed cố định)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--link", action="store_true", help="symlink thay vì chép ảnh gốc (không dùng trên Drive)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    thr = parse_thr(a.thr)
    imgs = list_images(a.images)
    n0 = len(imgs)
    bad = load_camera_set(a.blacklist) | load_camera_set(a.exclude_cameras)
    if bad:
        imgs = [p for p in imgs if camera_id(p.name) not in bad]
    if a.limit and a.limit < len(imgs):
        imgs = sorted(random.Random(a.seed).sample(imgs, a.limit))
    cams = {camera_id(p.name) for p in imgs} - {None}
    print(f"ảnh tìm thấy {n0} | sau loại camera ({len(bad)} mã) và giới hạn: {len(imgs)} | "
          f"số camera: {len(cams)} | chế độ: {a.mode} | ngưỡng: "
          f"{ {NAMES[k]: v for k, v in thr.items()} }")
    if a.dry_run:
        return 0
    if not a.teacher:
        ap.error("cần ít nhất một --teacher (hoặc dùng --dry-run)")
    for t in a.teacher:
        if not Path(t).exists():
            ap.error(f"không thấy trọng số giáo viên: {t}")
    models = load_models(a.teacher)
    stats = generate(imgs, models, a.out, a.mode, thr, a.small_frac, a.small_thr, a.low_thr,
                     a.mask_max_frac, a.imgsz, a.tta, a.device, a.link)
    manifest = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                "thr": {NAMES[k]: v for k, v in thr.items()}, "stats": stats,
                "note": "Số hộp giữ theo lớp cho thấy mất cân bằng; kiểm tay 300-500 ảnh để đo độ chính xác nhãn."}
    (a.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[xong] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
