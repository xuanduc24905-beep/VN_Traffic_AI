"""Chạy V5 model inference trên 1463 ảnh snapshot -> pseudo-label YOLO format.

Input : data/raw/frames/hcm/
Output: data/pseudo_v6/round1/predict/labels/*.txt   (YOLO format)
        data/pseudo_v6/round1/predict/images/*.jpg   (ảnh gốc được giữ)

Format mỗi dòng .txt:  <class_id> <x_c> <y_c> <w> <h> <conf>

Chạy:
  python scripts/pseudo/04_inference.py
  python scripts/pseudo/04_inference.py --conf 0.30    # lấy rộng hơn
  python scripts/pseudo/04_inference.py --model runs/detect/train_v8s_v6_pseudo/weights/best.pt
"""

import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO

MODEL_DEFAULT = "runs/detect/train_v8s_v5_honest/weights/best.pt"
FRAMES_DIR = Path("data/raw/frames/hcm")
OUT_DIR = Path("data/pseudo_v6/round1")


def main(model_path: str, conf: float, imgsz: int, device: str, batch: int):
    assert Path(model_path).exists(), f"Không thấy model: {model_path}"
    assert FRAMES_DIR.exists(), f"Không thấy frames: {FRAMES_DIR}"

    frame_paths = sorted(str(p) for p in FRAMES_DIR.rglob("*.jpg"))
    n_frames = len(frame_paths)
    print(f"[load] model: {model_path}")
    print(f"[load] frames: {n_frames} ảnh trong {FRAMES_DIR}")
    print(f"[config] conf={conf} imgsz={imgsz} device={device} batch={batch}")

    model = YOLO(model_path)
    print(f"[info] classes: {model.names}")

    # Clear old output
    predict_dir = OUT_DIR / "predict"
    if predict_dir.exists():
        print(f"[clean] xoá {predict_dir}")
        shutil.rmtree(predict_dir)

    # Run inference — save_txt tự ghi YOLO format
    print(f"\n[run] inference...")
    results = model.predict(
        source=frame_paths,
        conf=conf,
        iou=0.5,
        imgsz=imgsz,
        save=False,         # không cần ảnh visualize ở bước này
        save_txt=True,
        save_conf=True,
        project=str(OUT_DIR.resolve()),   # absolute: Ultralytics không prepend runs/detect/
        name="predict",
        exist_ok=True,
        stream=True,
        verbose=False,
        batch=batch,
        device=device,
        half=(device != "cpu"),
    )

    # Consume generator + count
    n_processed = 0
    n_with_detection = 0
    n_boxes_total = 0
    class_counts = {name: 0 for name in model.names.values()}

    for r in results:
        n_processed += 1
        if r.boxes is not None and len(r.boxes) > 0:
            n_with_detection += 1
            n_boxes_total += len(r.boxes)
            for cls_id in r.boxes.cls.cpu().numpy().astype(int):
                class_counts[model.names[cls_id]] += 1
        if n_processed % 100 == 0:
            print(f"  [{n_processed}/{n_frames}] with_det={n_with_detection} boxes={n_boxes_total}")

    # Report
    print(f"\n{'='*50}")
    print(f"[DONE]")
    print(f"  Processed: {n_processed} ảnh")
    print(f"  Có ≥1 box: {n_with_detection} ({100*n_with_detection/max(1,n_processed):.1f}%)")
    print(f"  Không có box: {n_processed - n_with_detection}")
    print(f"  Tổng box : {n_boxes_total}")
    print(f"  TB box/ảnh (ảnh có box): {n_boxes_total/max(1,n_with_detection):.1f}")
    print(f"\n  Phân bố class:")
    for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(1, n_boxes_total)
        print(f"    {name:12s}: {count:5d} ({pct:.1f}%)")
    print(f"\n  Output: {predict_dir}/labels/*.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--conf", type=float, default=0.30,
                    help="ngưỡng confidence (lấy rộng, lọc chặt ở B4)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0", help='"0" cho GPU, "cpu" cho CPU')
    ap.add_argument("--batch", type=int, default=4,
                    help="batch size (giảm nếu GPU OOM)")
    args = ap.parse_args()
    main(args.model, args.conf, args.imgsz, args.device, args.batch)
