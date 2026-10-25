"""Benchmark chi tiết baseline vs improved trên video mới.

Đo:
- Số detection mỗi frame (per class)
- Confidence trung bình
- Số frame có motorbike bị bắt
- Total detections (KHÔNG chỉ counting qua line — chỉ đo detection thô)
"""
from __future__ import annotations
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

VIDEO = "/tmp/vn_traffic_ai_uploads/Screen Recording 2026-09-19 231332.mp4"

MODELS = {
    "baseline_yolov8s_coco": ROOT / "weights/pretrained/yolov8s.pt",
    "improved_ft_vnv3":      ROOT / "runs/detect/train_v8s_ft_vnv3/weights/best.pt",
}

COCO_VEHICLE = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def run(name: str, weights: Path, video: str, conf: float = 0.25) -> dict:
    """Chạy detect trên toàn bộ video, thu stats detection thô."""
    model = YOLO(str(weights))
    is_coco = model.model.nc == 80
    names = model.names

    n_frames = 0
    n_det_total = 0
    per_class_det = Counter()          # tổng số detection mỗi class qua toàn video
    per_class_conf_sum = defaultdict(float)  # để tính conf trung bình
    per_class_frame_count = Counter()  # số frame có ít nhất 1 det class đó

    t0 = time.time()
    for res in model.predict(video, conf=conf, imgsz=640, verbose=False,
                             stream=True, device=0):
        n_frames += 1
        if res.boxes is None or len(res.boxes) == 0:
            continue
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()

        classes_this_frame = set()
        for cid, cf in zip(cls_ids, confs):
            if is_coco:
                if cid not in COCO_VEHICLE:
                    continue
                cname = COCO_VEHICLE[cid]
            else:
                cname = names[int(cid)]
            per_class_det[cname] += 1
            per_class_conf_sum[cname] += float(cf)
            classes_this_frame.add(cname)
            n_det_total += 1
        for c in classes_this_frame:
            per_class_frame_count[c] += 1
    dt = time.time() - t0

    avg_conf = {c: (per_class_conf_sum[c] / per_class_det[c])
                for c in per_class_det}

    return {
        "name": name,
        "weights": str(weights.relative_to(ROOT)),
        "n_frames": n_frames,
        "runtime": round(dt, 2),
        "fps": round(n_frames / dt, 1) if dt > 0 else 0,
        "total_detections": n_det_total,
        "det_per_frame": round(n_det_total / n_frames, 2) if n_frames else 0,
        "per_class_total_bbox": dict(per_class_det),
        "per_class_avg_conf": {c: round(v, 3) for c, v in avg_conf.items()},
        "per_class_frames_present": dict(per_class_frame_count),
        "pct_frames_with_motorcycle": round(
            per_class_frame_count.get("motorcycle", 0) / n_frames * 100, 1
        ) if n_frames else 0,
    }


def main():
    print(f"Video: {VIDEO}")
    results = {}
    for name, wp in MODELS.items():
        print(f"\n=== {name} ===")
        r = run(name, wp, VIDEO, conf=0.25)
        results[name] = r
        print(f"  frames             : {r['n_frames']}")
        print(f"  runtime            : {r['runtime']}s ({r['fps']} FPS)")
        print(f"  total detections   : {r['total_detections']} ({r['det_per_frame']}/frame)")
        print(f"  per class bbox     : {r['per_class_total_bbox']}")
        print(f"  per class avg conf : {r['per_class_avg_conf']}")
        print(f"  frames w/ mbike    : {r['per_class_frames_present'].get('motorcycle', 0)}"
              f" ({r['pct_frames_with_motorcycle']}%)")

    print("\n=== SO SÁNH ===")
    bl = results["baseline_yolov8s_coco"]
    im = results["improved_ft_vnv3"]
    for cls in ["motorcycle", "car", "bus", "truck"]:
        bl_bb = bl["per_class_total_bbox"].get(cls, 0)
        im_bb = im["per_class_total_bbox"].get(cls, 0)
        bl_conf = bl["per_class_avg_conf"].get(cls, 0)
        im_conf = im["per_class_avg_conf"].get(cls, 0)
        bl_fr = bl["per_class_frames_present"].get(cls, 0)
        im_fr = im["per_class_frames_present"].get(cls, 0)
        diff = im_bb - bl_bb
        print(f"\n  [{cls}]")
        print(f"    bbox tổng     : baseline {bl_bb:>5}  vs  improved {im_bb:>5}  (Δ {diff:+d})")
        print(f"    conf trung bình: baseline {bl_conf:>5}  vs  improved {im_conf:>5}")
        print(f"    frames có class: baseline {bl_fr:>5}  vs  improved {im_fr:>5}"
              f"  /{bl['n_frames']} frames")

    import json
    out = ROOT / "results/tables/bench_new_video.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
