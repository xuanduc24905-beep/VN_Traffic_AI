#!/usr/bin/env python3
"""BƯỚC 4 — Counting eval.

Runs the pipeline (YOLO detection → ByteTrack → line counter) on every available
(video, GT) pair in data/test_videos/ with the final r_clspw_v2 checkpoint. Reports:

- Pred vs GT counts per class.
- Per-video MAE/MAPE.
- Overall MAE and MAPE.
- FPS end-to-end (frames / wall-time).
- Tracker config (bytetrack.yaml).
- Line position.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/xuand/vn_traffic_ai")
sys.path.insert(0, str(ROOT))

from src.tracking.counter import Line, Counter  # noqa: E402
from src.tracking.track import track_stream  # noqa: E402

WEIGHTS = "runs/detect/r_clspw_v2/weights/best.pt"
TRACKER = "bytetrack.yaml"


def run_one(video: Path, gt: dict, line_cfg: dict) -> dict:
    line = Line(
        name=line_cfg.get("name", "mid"),
        p1=tuple(line_cfg["p1"]),
        p2=tuple(line_cfg["p2"]),
        count_direction=line_cfg.get("direction", "both"),
    )
    counter = Counter(lines=[line])
    pred = defaultdict(int)
    class_names = None
    n_frames = 0
    t0 = time.time()
    for frame_idx, res in enumerate(track_stream(
            WEIGHTS, str(video), tracker=TRACKER,
            conf=0.3, iou=0.5, imgsz=960, half=True)):
        n_frames += 1
        if class_names is None:
            class_names = res.names
        if res.boxes.id is None:
            continue
        boxes = res.boxes.xyxy.cpu().numpy()
        ids = res.boxes.id.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy()
        evs = counter.update(frame_idx, boxes, ids, classes)
        for ev in evs:
            cid = int(ev["class_id"])
            pred[class_names[cid]] += 1
    dt = time.time() - t0
    # Build per-class comparison
    classes = sorted(set(gt) | set(pred))
    per_class = {}
    abs_err_sum = 0
    for c in classes:
        g = int(gt.get(c, 0))
        p = int(pred.get(c, 0))
        per_class[c] = {"pred": p, "gt": g, "abs_err": abs(p - g)}
        abs_err_sum += abs(p - g)
    n_cls = max(len(classes), 1)
    mae = abs_err_sum / n_cls
    total_gt = sum(gt.values())
    total_pred = sum(pred.values())
    mape = (100.0 * abs(total_pred - total_gt) / total_gt) if total_gt > 0 else None
    return {
        "video": str(video.relative_to(ROOT)),
        "gt": dict(gt),
        "pred": dict(pred),
        "per_class": per_class,
        "total_gt": total_gt,
        "total_pred": total_pred,
        "MAE_per_class": round(mae, 3),
        "overall_MAPE_%": round(mape, 2) if mape is not None else None,
        "frames": n_frames,
        "runtime_sec": round(dt, 2),
        "fps_end_to_end": round(n_frames / dt, 2) if dt > 0 else None,
        "line": line_cfg,
        "weights": WEIGHTS,
        "tracker": TRACKER,
        "imgsz": 960, "conf": 0.3, "iou": 0.5, "half": True,
    }


def main():
    gt_dir = ROOT / "data/test_videos/ground_truth"
    video_dir = ROOT / "data/test_videos/raw"
    gt_files = sorted(gt_dir.glob("*.json"))
    if not gt_files:
        print("KHÔNG CÓ DỮ LIỆU: no GT files in data/test_videos/ground_truth/")
        return

    results = []
    for gt_path in gt_files:
        gt_doc = json.loads(gt_path.read_text())
        video_name = gt_doc["video"]
        video_path = video_dir / video_name
        if not video_path.exists():
            print(f"skip: {video_name} not found")
            continue
        # default line = horizontal middle
        iw, ih = gt_doc["resolution"]
        line_cfg = {"name": "mid", "p1": [100, ih // 2], "p2": [iw - 100, ih // 2], "direction": "both"}
        # override if GT has 'line'
        if "line" in gt_doc:
            line_cfg = gt_doc["line"]
        print(f"→ {video_name}  (GT total={sum(gt_doc['counts_by_class'].values())})")
        r = run_one(video_path, gt_doc["counts_by_class"], line_cfg)
        results.append(r)
        for c, v in r["per_class"].items():
            print(f"    {c:10s} pred={v['pred']:3d} gt={v['gt']:3d} err={v['abs_err']:3d}")
        print(f"    MAE/class={r['MAE_per_class']}  overall MAPE={r['overall_MAPE_%']}%  FPS={r['fps_end_to_end']}")

    payload = json.dumps(results, indent=2, ensure_ascii=False)
    (ROOT / "outputs/verify/counting_eval.json").write_text(payload)
    (ROOT / "outputs/verify/counting_eval_r_clspw_v2.json").write_text(payload)
    print("Wrote outputs/verify/counting_eval.json")
    print("Wrote outputs/verify/counting_eval_r_clspw_v2.json")


if __name__ == "__main__":
    main()
