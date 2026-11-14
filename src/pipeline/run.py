"""Pipeline tích hợp end-to-end: video detect+track count CSV events + video annotated.

Có thể dùng qua CLI (main) hoặc import function `run_pipeline` từ code khác (Streamlit).

CLI:
    python -m src.pipeline.run \
        --video data/test_videos/raw/nga_tu_01.mp4 \
        --weights runs/detect/train/weights/best.pt \
        --tracker bytetrack.yaml \
        --counting-config configs/counting_zones.json \
        --video-key nga_tu_01 \
        --out-csv results/tables/nga_tu_01_events.csv \
        --out-video results/videos/nga_tu_01_out.mp4
"""
import argparse
import csv
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
import cv2

from src.tracking.track import track_stream
from src.tracking.counter import Counter, load_lines


_DIR_COLOR = {"ltr": (0, 255, 255), "rtl": (255, 0, 255)}


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    width: int
    height: int
    total_frames: int

    @property
    def duration_sec(self) -> float:
        return self.total_frames / self.fps if self.total_frames > 0 else 0.0


def probe_video(video: str) -> VideoMetadata:
    """Read and validate video metadata before starting expensive inference."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Không mở được video: {video}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        cap.release()
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
    if width <= 0 or height <= 0:
        raise ValueError(f"Video có kích thước không hợp lệ: {width}x{height}")
    return VideoMetadata(fps=fps, width=width, height=height,
                         total_frames=total_frames)


def _class_name(class_names, class_id: int) -> str:
    if class_names is None:
        return str(class_id)
    try:
        return str(class_names[int(class_id)])
    except (KeyError, IndexError, TypeError):
        return str(class_id)


def _write_events_csv(events: list[dict], out_csv: str) -> None:
    """Atomically replace the destination so a failed write leaves no partial CSV."""
    destination = Path(out_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
                "w", newline="", encoding="utf-8-sig",
                dir=destination.parent, prefix=f".{destination.name}.",
                suffix=".tmp", delete=False) as tmp:
            tmp_name = tmp.name
            writer = csv.DictWriter(
                tmp,
                fieldnames=["frame", "time_sec", "line", "direction",
                            "track_id", "class_name"],
            )
            writer.writeheader()
            writer.writerows(events)
        os.replace(tmp_name, destination)
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def draw_overlay(frame, boxes_xyxy, ids, classes, lines, counter, class_names):
    for xyxy, tid, cls in zip(boxes_xyxy, ids, classes):
        if tid is None:
            continue
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{_class_name(class_names, int(cls))}#{int(tid)}"
        cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    for line in lines:
        p1 = tuple(round(v) for v in line.p1)
        p2 = tuple(round(v) for v in line.p2)
        cv2.line(frame, p1, p2, (0, 0, 255), 2)
        tag = line.name + (f" [{line.count_direction}]"
                           if line.count_direction != "both" else "")
        cv2.putText(frame, tag, p1,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    y = 30
    for ln_name, dirs in counter.counts.items():
        total = sum(sum(c.values()) for c in dirs.values())
        cv2.putText(frame, f"{ln_name}: {total}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 22

        # Tổng theo class (cộng gộp mọi direction) — hàng ưu tiên hiển thị
        per_class_total = {}
        for direction, per_cls in dirs.items():
            for cid, n in per_cls.items():
                per_class_total[cid] = per_class_total.get(cid, 0) + n
        if per_class_total:
            for cid, n in sorted(per_class_total.items()):
                cname = _class_name(class_names, int(cid))
                cv2.putText(frame, f"  {cname}: {n}", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                y += 18

        # Theo direction (tổng)
        for direction, per_cls in dirs.items():
            sub = sum(per_cls.values())
            color = _DIR_COLOR.get(direction, (200, 200, 200))
            cv2.putText(frame, f"  {direction}: {sub}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 16
    return frame


def run_pipeline(video: str, weights: str, lines: list,
                 tracker: str = "bytetrack.yaml",
                 conf: float = 0.25, iou: float = 0.5, imgsz: int = 640,
                 half: bool = False,
                 out_csv: str | None = None,
                 out_video: str | None = None,
                 progress_cb=None,
                 device: str | int | None = None,
                 anchor_mode: str = "bottom_center",
                 crossing_hysteresis: float = 3.0,
                 max_track_gap: int = 30,
                 classes: list[int] | None = None):
    """Chạy pipeline. Trả về (events list, counter.to_dict(), class_names).

    events là list dict với schema:
        frame, time_sec, line, direction, track_id, class_name
    """
    metadata = probe_video(video)
    fps = metadata.fps
    w, h = metadata.width, metadata.height
    total_frames = metadata.total_frames

    counter = Counter(
        lines=lines,
        anchor_mode=anchor_mode,
        hysteresis_px=crossing_hysteresis,
        max_gap_frames=max_track_gap,
        state_ttl_frames=max(300, max_track_gap),
    )

    writer = None
    if out_video:
        Path(out_video).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            writer.release()
            raise OSError(f"Không tạo được video output: {out_video}")

    events = []
    class_names = None

    # FP16 is not supported by the CPU inference path.
    if isinstance(device, str) and device.lower() == "cpu":
        half = False

    try:
        for frame_idx, res in enumerate(track_stream(
                weights, video, tracker=tracker,
                conf=conf, iou=iou, imgsz=imgsz, half=half,
                device=device, classes=classes)):

            if class_names is None:
                class_names = res.names

            frame = res.orig_img.copy() if writer is not None else None
            boxes = res.boxes.xyxy.cpu().numpy()
            detected_classes = res.boxes.cls.cpu().numpy()
            confidences = res.boxes.conf.cpu().numpy()
            if res.boxes.id is None:
                ids = [None] * len(boxes)
            else:
                ids = res.boxes.id.cpu().numpy()

            frame_events = counter.update(
                frame_idx, boxes, ids, detected_classes, confidences)
            for event in frame_events:
                events.append({
                    "frame": frame_idx,
                    "time_sec": round(frame_idx / fps, 3),
                    "line": event["line"],
                    "direction": event["direction"],
                    "track_id": event["track_id"],
                    "class_name": _class_name(class_names, event["class_id"]),
                })

            # Lines and accumulated counts stay visible even on frames with no
            # confirmed tracking IDs.
            if writer is not None and frame is not None:
                frame = draw_overlay(frame, boxes, ids, detected_classes,
                                     lines, counter, class_names)
                writer.write(frame)

            if progress_cb is not None:
                progress_cb(frame_idx + 1, total_frames)
    finally:
        if writer is not None:
            writer.release()

    if out_csv:
        _write_events_csv(events, out_csv)

    return events, counter.to_dict(), class_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--counting-config", default="configs/counting_zones.json")
    ap.add_argument("--video-key", required=True, help="key trong counting_zones.json")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto", help="auto, cpu, mps, 0, 0,1, ...")
    ap.add_argument("--half", action="store_true", help="FP16 inference (nhanh 1.3-1.5x trên GPU Ada/Ampere)")
    ap.add_argument("--anchor", default="bottom_center",
                    choices=["center", "bottom_center"])
    ap.add_argument("--crossing-hysteresis", type=float, default=3.0,
                    help="Dead-band quanh line, tính bằng pixel")
    ap.add_argument("--max-track-gap", type=int, default=30,
                    help="Không nối trajectory nếu ID biến mất quá số frame này")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-video", default=None)
    args = ap.parse_args()

    metadata = probe_video(args.video)
    lines = load_lines(args.counting_config, args.video_key,
                       video_size=(metadata.width, metadata.height))
    device = None if args.device == "auto" else args.device
    events, totals, _ = run_pipeline(
        video=args.video, weights=args.weights, lines=lines,
        tracker=args.tracker, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
        half=args.half,
        device=device, anchor_mode=args.anchor,
        crossing_hysteresis=args.crossing_hysteresis,
        max_track_gap=args.max_track_gap,
        out_csv=args.out_csv, out_video=args.out_video,
    )
    print(f"[] {len(events)} events {args.out_csv}")
    print(f"[] Totals: {totals}")


if __name__ == "__main__":
    main()
