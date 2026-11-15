#!/usr/bin/env python3
"""BƯỚC 5 — Speed benchmark.

(a) Model-only latency for r_clspw YOLO at imgsz=960, FP16/FP32, batch=1,
    with 20-image warmup and ≥200-image measurement.
(b) Model-only latency for FRCNN at same input size (resize min=800, max=1333)
    FP32, batch=1.
(c) End-to-end FPS for the demo_traffic.mp4 pipeline with r_clspw + ByteTrack
    + Counter, measured with and without writing an annotated output video.
(d) Tracker-only cost (ms/frame): measured by comparing (detection-only ms)
    vs (detection + tracking ms) on the same frames.

Outputs:
  outputs/verify/speed_benchmark.json
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/xuand/vn_traffic_ai")
sys.path.insert(0, str(ROOT))


def bench_yolo_model_only(ckpt: str, imgsz: int, half: bool,
                          warmup: int = 20, measure: int = 200) -> dict:
    import torch
    from ultralytics import YOLO

    model = YOLO(str(ROOT / ckpt))
    # use random images from v5plus test
    all_imgs = sorted((ROOT / "data/merged_v5plus/images/test").glob("*.jpg"))
    rng = random.Random(42)
    paths = [str(rng.choice(all_imgs)) for _ in range(warmup + measure)]
    # warmup
    for p in paths[:warmup]:
        model.predict(source=p, imgsz=imgsz, half=half, device=0, verbose=False)
    torch.cuda.synchronize()
    # measure
    lat = []
    for p in paths[warmup:]:
        t0 = time.perf_counter()
        _ = model.predict(source=p, imgsz=imgsz, half=half, device=0, verbose=False)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000.0)
    lat = np.array(lat)
    return {
        "ckpt": ckpt, "imgsz": imgsz, "precision": "FP16" if half else "FP32",
        "device": torch.cuda.get_device_name(0),
        "warmup": warmup, "measured": measure,
        "mean_ms": round(float(lat.mean()), 3),
        "median_ms": round(float(np.median(lat)), 3),
        "p95_ms": round(float(np.percentile(lat, 95)), 3),
        "fps_mean": round(1000.0 / lat.mean(), 2),
        "fps_total": round(measure / (lat.sum() / 1000.0), 2),
        "note": "model.predict(): preprocess + inference + NMS; no I/O",
    }


def bench_frcnn_model_only(warmup: int = 20, measure: int = 200) -> dict:
    """Load the trained FRCNN best.pt and time per-image inference."""
    import torch
    import torchvision
    from torchvision.transforms import functional as F
    import cv2

    ckpt = ROOT / "weights/trained/frcnn/best.pt"
    device = torch.device("cuda:0")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    # Typical save shape: {"model": state_dict, "num_classes": ...} or just state_dict
    if isinstance(state, dict) and "model" in state:
        num_classes = state.get("num_classes", 5)  # 4 + background
        state_dict = state["model"]
    else:
        num_classes = 5
        state_dict = state
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=91)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    all_imgs = sorted((ROOT / "data/merged_v5plus/images/test").glob("*.jpg"))
    rng = random.Random(42)
    paths = [str(rng.choice(all_imgs)) for _ in range(warmup + measure)]

    def load(p):
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = F.to_tensor(img).to(device)
        return t

    # warmup
    with torch.inference_mode():
        for p in paths[:warmup]:
            t = load(p)
            _ = model([t])
    torch.cuda.synchronize()
    lat = []
    with torch.inference_mode():
        for p in paths[warmup:]:
            t = load(p)
            t0 = time.perf_counter()
            _ = model([t])
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000.0)
    lat = np.array(lat)
    return {
        "ckpt": "weights/trained/frcnn/best.pt",
        "input": "orig (variable)",
        "precision": "FP32",
        "device": torch.cuda.get_device_name(0),
        "warmup": warmup, "measured": measure,
        "mean_ms": round(float(lat.mean()), 3),
        "median_ms": round(float(np.median(lat)), 3),
        "p95_ms": round(float(np.percentile(lat, 95)), 3),
        "fps_mean": round(1000.0 / lat.mean(), 2),
        "fps_total": round(measure / (lat.sum() / 1000.0), 2),
        "note": "FRCNN model-only forward pass; no NMS post-filter",
    }


def bench_pipeline_end_to_end(write_video: bool) -> dict:
    """End-to-end on demo_traffic.mp4: detect+track+count (+ optional draw/write)."""
    import cv2
    import torch

    from src.tracking.counter import Line, Counter
    from src.tracking.track import track_stream

    video = str(ROOT / "data/test_videos/raw/demo_traffic.mp4")
    gt_doc = json.loads((ROOT / "data/test_videos/ground_truth/demo_traffic.json").read_text())
    iw, ih = gt_doc["resolution"]
    line = Line(name="mid", p1=(100, ih // 2), p2=(iw - 100, ih // 2), count_direction="both")
    counter = Counter(lines=[line])
    weights = "runs/detect/r_clspw/weights/best.pt"

    out_path = ROOT / "outputs/verify/pipeline_render.mp4" if write_video else None
    writer = None

    n_frames = 0
    t0 = time.perf_counter()
    for frame_idx, res in enumerate(track_stream(
            weights, video, tracker="bytetrack.yaml",
            conf=0.3, iou=0.5, imgsz=960, half=True)):
        n_frames += 1
        if res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy()
            counter.update(frame_idx, boxes, ids, classes)
        if write_video:
            frame = res.orig_img.copy()
            if res.boxes.id is not None:
                for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
                    x1, y1, x2, y2 = (int(x) for x in b)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(frame, line.p1, line.p2, (0, 0, 255), 2)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out_path),
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         gt_doc.get("fps", 30), (w, h))
            writer.write(frame)
    if writer is not None:
        writer.release()
    dt = time.perf_counter() - t0
    return {
        "video": str(Path(video).relative_to(ROOT)),
        "write_video": write_video,
        "frames": n_frames,
        "wall_time_sec": round(dt, 3),
        "fps_end_to_end": round(n_frames / dt, 2),
        "output_video": str(out_path.relative_to(ROOT)) if out_path else None,
    }


def main():
    out = {}
    # (a) YOLO model-only FP16
    print("[a1] r_clspw model-only FP16 960")
    out["r_clspw_fp16_960"] = bench_yolo_model_only("runs/detect/r_clspw/weights/best.pt", 960, True)
    print("[a2] r_clspw model-only FP32 960")
    out["r_clspw_fp32_960"] = bench_yolo_model_only("runs/detect/r_clspw/weights/best.pt", 960, False)
    # (b) FRCNN model-only FP32
    print("[b] FRCNN model-only FP32")
    try:
        out["frcnn_fp32"] = bench_frcnn_model_only()
    except Exception as e:
        out["frcnn_fp32"] = {"ERROR": repr(e)}
    # (c) pipeline end-to-end no write
    print("[c1] pipeline end-to-end no write")
    out["pipeline_no_write"] = bench_pipeline_end_to_end(write_video=False)
    print("[c2] pipeline end-to-end with write")
    out["pipeline_with_write"] = bench_pipeline_end_to_end(write_video=True)

    (ROOT / "outputs/verify/speed_benchmark.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )
    print("Wrote outputs/verify/speed_benchmark.json")


if __name__ == "__main__":
    main()
