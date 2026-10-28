"""
Bước 6: Chạy model đã train trên ảnh / video mới.

Script có 2 mode:
    1. --image abc.jpg       → predict 1 ảnh, lưu ảnh có bbox overlay
    2. --video abc.mp4       → predict từng frame, lưu video output

Lưu ý:
    - Trong production, ta không chỉ detect mỗi frame độc lập — còn cần TRACKING
      để gán ID bền (xe máy đi qua 10 frame liên tiếp phải cùng ID 7, không nhảy ID).
    - Ultralytics tích hợp sẵn ByteTrack / BoT-SORT: thay model.predict() bằng
      model.track() là có tracking.
    - Pipeline đầy đủ (detect → track → đếm qua line → export CSV) ở
      src/pipeline/run.py — phức tạp hơn.

Chạy:
    python tutorial/6_inference.py --image data/merged_v5/images/test/some.jpg
    python tutorial/6_inference.py --video path/to/video.mp4
"""

import argparse       # parse command line args
from pathlib import Path
import cv2            # OpenCV để đọc/ghi ảnh/video
from ultralytics import YOLO


# ─── Config mặc định ─────────────────────────────────────────────────────────

WEIGHTS = "runs/detect/tutorial_train_v8s/weights/best.pt"
IMGSZ = 960

# Confidence threshold — box có conf < này bị loại.
# Thấp = bắt nhiều (có thể có box rác). Cao = ít FP nhưng dễ miss.
# 0.25 là mặc định ultralytics. Demo đẹp dùng 0.3-0.4.
CONF = 0.25

# IoU threshold cho Non-Maximum Suppression.
# Khi 2 box chồng ≥ ngưỡng → giữ box conf cao hơn. 0.7 là mặc định.
IOU = 0.7

# Device
DEVICE = 0


def predict_image(model: YOLO, img_path: str, out_dir: Path):
    """
    Chạy predict trên 1 ảnh, lưu ảnh có bbox overlay.
    """
    print(f"\n[Image] Predict: {img_path}")

    # model(path) ngắn gọn cho model.predict(path). Return list results (1 item mỗi ảnh).
    results = model.predict(
        source=img_path,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        device=DEVICE,
        verbose=False,
        save=False,      # Ta tự save ở dưới để kiểm soát path
    )

    r = results[0]  # vì chỉ truyền 1 ảnh

    # r.boxes là object chứa tất cả bbox detected trong ảnh.
    # Attributes:
    #   r.boxes.xyxy    — tensor [N, 4]: toạ độ (x1, y1, x2, y2) ở pixel space
    #   r.boxes.conf    — tensor [N]: confidence của mỗi box
    #   r.boxes.cls     — tensor [N]: class id (0/1/2/3)
    #   r.boxes.xywh    — center x, center y, width, height
    #   r.boxes.xyxyn   — toạ độ normalize [0, 1]
    n_boxes = len(r.boxes)
    print(f"  Phát hiện {n_boxes} object(s)")

    if n_boxes > 0:
        # In chi tiết từng box
        for i in range(n_boxes):
            cls_id = int(r.boxes.cls[i])
            cls_name = model.names[cls_id]
            conf = float(r.boxes.conf[i])
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            print(f"    [{i+1}] {cls_name:12s} conf={conf:.2f} "
                  f"bbox=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    # Vẽ bbox lên ảnh. r.plot() return ảnh numpy array (BGR, như OpenCV)
    # có sẵn bbox + label + confidence overlay. Ultralytics tự chọn màu theo class.
    annotated = r.plot()

    out_path = out_dir / f"predict_{Path(img_path).stem}.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"  → Lưu: {out_path}")


def predict_video(model: YOLO, video_path: str, out_dir: Path):
    """
    Chạy predict trên video, lưu video output có bbox overlay.

    Nếu muốn tracking (ID bền giữa các frame) → đổi model.predict() thành model.track().
    """
    print(f"\n[Video] Predict: {video_path}")

    # Mở video với OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ Không mở được video: {video_path}")
        return

    # Lấy thông tin video
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Info: {w}x{h} @ {fps:.1f} FPS, {n_frames} frames "
          f"(~{n_frames/fps:.1f}s)")

    # Setup video writer. mp4v codec, output .mp4.
    # Lưu ý: mp4v chạy OK trên desktop nhưng không play được trên trình duyệt.
    # Nếu cần streaming: transcode qua H.264 bằng ffmpeg (xem src/pipeline/run.py).
    out_path = out_dir / f"predict_{Path(video_path).stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    # Loop qua từng frame
    frame_idx = 0
    total_detections = 0

    # Dùng model.track() thay vì predict() để có ID bền giữa các frame.
    # ByteTrack sẽ match box dựa trên IoU + confidence qua các frame.
    # persist=True nghĩa là giữ track state giữa các lần gọi (cho streaming).
    while True:
        ret, frame = cap.read()  # Đọc 1 frame. ret=False khi hết video.
        if not ret:
            break

        # Chạy model + tracking
        results = model.track(
            source=frame,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            device=DEVICE,
            persist=True,     # Giữ track ID qua các frame
            tracker="bytetrack.yaml",  # hoặc "botsort.yaml" (chính xác hơn, chậm hơn)
            verbose=False,
        )

        r = results[0]
        total_detections += len(r.boxes)

        # Vẽ overlay
        annotated = r.plot()

        # Ghi vào output
        writer.write(annotated)

        # Progress mỗi 30 frame (~1 giây video)
        frame_idx += 1
        if frame_idx % 30 == 0:
            pct = 100 * frame_idx / n_frames if n_frames else 0
            print(f"  [{frame_idx:>5}/{n_frames}] ({pct:4.1f}%) "
                  f"detections tổng: {total_detections}")

    # Dọn dẹp
    cap.release()
    writer.release()

    print(f"\n  ✓ Xong. Tổng {frame_idx} frames, {total_detections} detections.")
    print(f"  → Output: {out_path}")


def main():
    # Parse arguments. argparse là module chuẩn để nhận --flag từ command line.
    parser = argparse.ArgumentParser(description="Inference YOLOv8 trên ảnh hoặc video")
    parser.add_argument("--image", type=str, help="Đường dẫn 1 ảnh để predict")
    parser.add_argument("--video", type=str, help="Đường dẫn video để predict")
    parser.add_argument("--weights", type=str, default=WEIGHTS,
                        help=f"Model weights (default: {WEIGHTS})")
    parser.add_argument("--out-dir", type=str, default="results/tutorial_inference",
                        help="Thư mục output")
    args = parser.parse_args()

    # Phải cho --image hoặc --video, không cả hai, không trống
    if not args.image and not args.video:
        parser.error("Phải truyền --image hoặc --video")

    # Tạo thư mục output
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Load model
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"❌ Không tìm thấy weights: {args.weights}")
        print("   Chạy bước 4 trước, hoặc truyền --weights path/to/best.pt")
        return
    print(f"Model: {args.weights}")

    model = YOLO(args.weights)

    # Chạy mode tương ứng
    if args.image:
        predict_image(model, args.image, out_dir)
    if args.video:
        predict_video(model, args.video, out_dir)

    print("\n→ Chạy tiếp bước 7 để vẽ biểu đồ training:")
    print("    python tutorial/7_plot_results.py")


if __name__ == "__main__":
    main()
