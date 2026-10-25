"""Tạo template ground truth JSON cho video test — Người 4 điền số thủ công.

Chạy:
    python scripts/prepare_test_video.py \
        --video data/test_videos/raw/nga_tu_01.mp4 \
        --out data/test_videos/ground_truth/nga_tu_01.json
"""
import argparse
import json
from pathlib import Path
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n_frames / fps if fps > 0 else 0
    cap.release()

    template = {
        "video": Path(args.video).name,
        "resolution": [w, h],
        "fps": round(fps, 3),
        "duration_sec": round(duration, 1),
        "_instructions": "Xem video và điền tay số lượt vào counts_by_class.",
        "counts_by_class": {
            "motorcycle": 0,
            "car": 0,
            "bus": 0,
            "truck": 0,
            "bicycle": 0
        },
        "counts_by_line": {}
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    print(f"[] Template GT tại: {out_path}")
    print(f" Video {duration:.1f}s, {n_frames} frame, {w}×{h} @ {fps:.1f} fps")


if __name__ == "__main__":
    main()
