"""Self-training round 1 — pseudo-label vn_cctv_train bằng V5+.

Workflow:
1. Load V5+ best.pt (train_v8s_v5plus_960)
2. Run inference trên data/raw_vn_cctv/train/images (26k frames)
3. Filter per-image: giữ chỉ frame có >=1 box với conf > CONF_MIN
4. Deduplicate: vn_cctv là 1 video → subsample stride để lấy ~500 frames đa dạng
5. Save YOLO labels + symlink images sang data/pseudo_vn_cctv/
6. Report distribution class để check pseudo-label có hợp lý không

IMPORTANT: vn_cctv_test (2919 frames) HOLD OUT, KHÔNG chạm → OOD eval sau này.

Output:
- data/pseudo_vn_cctv/images/*.jpg  (symlink)
- data/pseudo_vn_cctv/labels/*.txt  (pseudo-labels)
- data/pseudo_vn_cctv/manifest.json (reproducibility: conf threshold, kept count, class dist)
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path("/home/xuand/vn_traffic_ai").resolve()
V5PLUS_WEIGHTS = PROJECT_ROOT / "runs/detect/train_v8s_v5plus_960/weights/best.pt"
SRC_IMAGES = PROJECT_ROOT / "data/raw_vn_cctv/train/images"
DST = PROJECT_ROOT / "data/pseudo_vn_cctv"
NAMES = ["motorcycle", "car", "bus", "truck"]

# Config
CONF_MIN = 0.75            # tối thiểu confidence để accept box
IOU_NMS  = 0.5             # NMS threshold
IMGSZ    = 960             # same as V5+ training
STRIDE   = 50              # vn_cctv 1 video → stride 50 ≈ 525 frames

FRAME_NUM = re.compile(r"_(\d+)_jpg\.rf\.|_(\d+)\.jpg$")


def extract_frame_num(name: str) -> int:
    m = FRAME_NUM.search(name)
    if m:
        return int(m.group(1) or m.group(2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=CONF_MIN)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--max_frames", type=int, default=None,
                    help="limit nếu test nhanh")
    args = ap.parse_args()

    assert V5PLUS_WEIGHTS.exists(), f"Thiếu {V5PLUS_WEIGHTS}. Train V5+ trước."
    assert SRC_IMAGES.exists(), f"Thiếu {SRC_IMAGES}"

    # Dọn + tạo dir
    if DST.exists():
        shutil.rmtree(DST)
    (DST / "images").mkdir(parents=True)
    (DST / "labels").mkdir(parents=True)

    # Load all images, sort by frame_num để subsample đều
    all_imgs = sorted(
        [p for p in SRC_IMAGES.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: extract_frame_num(p.name),
    )
    print(f"[INFO] Found {len(all_imgs)} frames trong {SRC_IMAGES}")

    # Subsample TRƯỚC khi inference để tiết kiệm compute
    sampled = all_imgs[::args.stride]
    if args.max_frames:
        sampled = sampled[:args.max_frames]
    print(f"[INFO] Subsample stride={args.stride} → {len(sampled)} frames để infer")

    print(f"\n[1/2] Load V5+ model: {V5PLUS_WEIGHTS}")
    model = YOLO(str(V5PLUS_WEIGHTS))

    print(f"[2/2] Inference + filter conf>={args.conf}...")
    kept_imgs = 0
    kept_boxes = 0
    class_dist: Counter = Counter()
    skipped_empty = 0

    # Batched inference
    BATCH = 16
    for batch_start in range(0, len(sampled), BATCH):
        batch_imgs = sampled[batch_start:batch_start + BATCH]
        batch_paths = [str(p) for p in batch_imgs]

        results = model.predict(
            batch_paths,
            imgsz=IMGSZ,
            conf=args.conf,
            iou=IOU_NMS,
            verbose=False,
            device=0,
        )

        for img_path, r in zip(batch_imgs, results):
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                skipped_empty += 1
                continue

            # Lấy class, conf, xywhn (normalized)
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()
            xywhn = boxes.xywhn.cpu().numpy()  # [N, 4] normalized

            # Build YOLO label lines (chỉ box qua filter)
            lines = []
            for i in range(len(cls)):
                if conf[i] < args.conf:
                    continue
                c = int(cls[i])
                x, y, w, h = xywhn[i]
                lines.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                class_dist[c] += 1

            if not lines:
                skipped_empty += 1
                continue

            # Prefix "vncctv_" vào tên để tránh clash video_id trong tương lai
            new_name = f"vncctv_pseudo_{img_path.name}"
            img_dst = DST / "images" / new_name
            lbl_dst = DST / "labels" / (Path(new_name).stem + ".txt")
            img_dst.symlink_to(img_path.resolve())
            lbl_dst.write_text("\n".join(lines))

            kept_imgs += 1
            kept_boxes += len(lines)

        if (batch_start // BATCH) % 10 == 0:
            print(f"  Processed {min(batch_start + BATCH, len(sampled))}/{len(sampled)}, "
                  f"kept {kept_imgs} imgs so far")

    # Report
    print(f"\n[DONE] Pseudo-labeling xong.")
    print(f"  Images kept: {kept_imgs}/{len(sampled)} "
          f"({100*kept_imgs/len(sampled):.1f}%)")
    print(f"  Boxes kept: {kept_boxes}")
    print(f"  Skipped (empty after filter): {skipped_empty}")
    print(f"\n  Class distribution:")
    total = sum(class_dist.values())
    for cid in range(4):
        cnt = class_dist[cid]
        pct = 100 * cnt / total if total else 0
        print(f"    {NAMES[cid]:<12}  {cnt:5d}  ({pct:.1f}%)")

    # Manifest
    manifest = {
        "source": str(SRC_IMAGES),
        "weights": str(V5PLUS_WEIGHTS),
        "conf_threshold": args.conf,
        "iou_nms": IOU_NMS,
        "imgsz": IMGSZ,
        "stride": args.stride,
        "total_frames_source": len(all_imgs),
        "frames_sampled": len(sampled),
        "frames_kept": kept_imgs,
        "boxes_kept": kept_boxes,
        "skipped_empty": skipped_empty,
        "class_distribution": {NAMES[c]: n for c, n in class_dist.items()},
    }
    (DST / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n[OK] Manifest: {DST / 'manifest.json'}")
    print(f"[OK] Pseudo dataset: {DST}")
    print(f"\nNext: chạy scripts/data_prep/build_v6_merged.py để merge V5 + pseudo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
