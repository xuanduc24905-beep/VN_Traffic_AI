"""
CHU DE 10: DEM VA PHAN LOAI PHUONG TIEN GIAO THONG (STARTER CODE)

File goc do giao vien cung cap. Day la baseline don gian dung yolov8s.pt
COCO pretrained + line counting 1-line + counted_ids set chong dem trung.
(Nhom dung v8s de fair so voi model fine-tune cung kien truc v8s.)

Yeu cau: pip install ultralytics opencv-python matplotlib

Chay:
    python baseline/10_traffic_counting.py

Cac goi y mo rong o cuoi file (nhom da lam trong src/):
    1. Nhieu counting line + direction (ltr/rtl)  -> src/tracking/counter.py
    2. Thong ke theo bucket thoi gian             -> src/stats/aggregator.py
    3. So sanh accuracy voi manual count          -> src/evaluation/
    4. Fine-tune YOLOv8 cho xe may VN             -> src/detection/train.py
"""
import cv2
from ultralytics import YOLO
from collections import defaultdict
import matplotlib.pyplot as plt

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle"}


def main(video_path, model_path="yolov8s.pt", line_y=400,
         out_video=None, show_window=True):
    """Baseline goc — chi 1 line ngang tai y=line_y, khong direction, khong log time."""
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    counted_ids = set()
    class_counts = defaultdict(int)
    track_history = {}  # {track_id: last_cy}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if out_video:
        writer = cv2.VideoWriter(out_video,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # model.track() tu dong gan track_id (dung ByteTrack mac dinh)
        results = model.track(frame, persist=True, conf=0.4, verbose=False)[0]
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)

        if results.boxes.id is not None:
            for box, track_id in zip(results.boxes, results.boxes.id):
                cls_name = model.names[int(box.cls[0])]
                if cls_name not in VEHICLE_CLASSES:
                    continue
                track_id = int(track_id)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cy = (y1 + y2) // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls_name} #{track_id}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Dem khi tam xe di qua duong ke (line_y) va chua duoc dem
                last_cy = track_history.get(track_id)
                if (last_cy is not None
                        and last_cy < line_y <= cy
                        and track_id not in counted_ids):
                    counted_ids.add(track_id)
                    class_counts[cls_name] += 1
                track_history[track_id] = cy

        # Hien thi bang thong ke tren frame
        y_offset = 30
        cv2.putText(frame, f"Tong: {sum(class_counts.values())}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)
        for cls_name, count in class_counts.items():
            y_offset += 25
            cv2.putText(frame, f"{cls_name}: {count}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2)

        if writer is not None:
            writer.write(frame)

        if show_window:
            cv2.imshow("Dem phuong tien giao thong - Nhan Q de thoat", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()

    # Xuat bieu do thong ke
    if class_counts:
        plt.bar(class_counts.keys(), class_counts.values(), color="skyblue")
        plt.title("Thong ke luu luong phuong tien (baseline)")
        plt.xlabel("Loai phuong tien")
        plt.ylabel("So luong")
        plt.tight_layout()
        plt.savefig("baseline/traffic_stats.png")
        plt.close()
        print("Da luu bieu do vao baseline/traffic_stats.png")

    print("Ket qua dem cuoi cung:", dict(class_counts))
    return dict(class_counts)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/test_videos/raw/demo_traffic.mp4")
    ap.add_argument("--model", default="yolov8s.pt",
                    help="mac dinh yolov8s.pt COCO pretrained (baseline fair vs v8s fine-tune)")
    ap.add_argument("--line-y", type=int, default=474,
                    help="Y ngang tai giua frame 948px cao")
    ap.add_argument("--out-video", default="baseline/baseline_out.mp4")
    ap.add_argument("--no-show", action="store_true",
                    help="Chay headless (khong hien window)")
    args = ap.parse_args()

    main(video_path=args.video, model_path=args.model, line_y=args.line_y,
         out_video=args.out_video, show_window=not args.no_show)


# ============================================================
# GOI Y MO RONG CHO SINH VIEN — nhom da lam:
# ------------------------------------------------------------
# 1. Nhieu counting line + phan direction (ltr/rtl) bang cross-product
#    -> src/tracking/counter.py
# 2. Thong ke theo bucket thoi gian (30s, 1p, 5p...), flow rate, peak
#    -> src/stats/aggregator.py
# 3. Evaluation 2 tang: mAP tren test set + counting acc vs GT thu cong
#    -> src/evaluation/eval_full.py
# 4. Fine-tune YOLOv8s tren merged dataset VN (68k UA-DETRAC + 674 CanTho)
#    -> src/detection/train.py + scripts/utils/train.sh
# ============================================================
