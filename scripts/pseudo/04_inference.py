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
import gc
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch

MODEL_DEFAULT = "runs/detect/train_v8s_v5_honest/weights/best.pt"
FRAMES_DIR = Path("data/raw/frames/hcm")
OUT_DIR = Path("data/pseudo_v6/round1")


def run_chunk(model, chunk_paths, conf, imgsz, device, batch, out_dir, name, class_counts):
    """Chạy inference 1 chunk, trả (n_processed, n_with_det, n_boxes)."""
    results = model.predict(
        source=chunk_paths,
        conf=conf,
        iou=0.5,
        imgsz=imgsz,
        save=False,
        save_txt=True,
        save_conf=True,
        project=str(out_dir.resolve()),
        name=name,
        exist_ok=True,
        stream=True,
        verbose=False,
        batch=batch,
        device=device,
        half=(device != "cpu"),
        workers=0,
    )

    n_processed = n_with_det = n_boxes = 0
    for r in results:
        n_processed += 1
        if r.boxes is not None and len(r.boxes) > 0:
            n_with_det += 1
            n_boxes += len(r.boxes)
            for cls_id in r.boxes.cls.cpu().numpy().astype(int):
                class_counts[model.names[cls_id]] += 1
    return n_processed, n_with_det, n_boxes


def main(model_path: str, conf: float, imgsz: int, device: str, batch: int,
         chunk_size: int, resume: bool):
    assert Path(model_path).exists(), f"Không thấy model: {model_path}"
    assert FRAMES_DIR.exists(), f"Không thấy frames: {FRAMES_DIR}"

    all_frame_paths = sorted(str(p) for p in FRAMES_DIR.rglob("*.jpg"))
    n_all = len(all_frame_paths)
    print(f"[load] model: {model_path}")
    print(f"[load] frames: {n_all} ảnh trong {FRAMES_DIR}")
    print(f"[config] conf={conf} imgsz={imgsz} device={device} batch={batch} "
          f"chunk={chunk_size} resume={resume}")

    model = YOLO(model_path)
    print(f"[info] classes: {model.names}")

    predict_dir = OUT_DIR / "predict"
    labels_dir = predict_dir / "labels"

    # ─── Resume logic ───
    if resume and labels_dir.exists():
        # Lấy stem của những ảnh đã có label
        done_stems = {p.stem for p in labels_dir.glob("*.txt")}
        frame_paths = [p for p in all_frame_paths
                       if Path(p).stem not in done_stems]
        n_frames = len(frame_paths)
        print(f"\n[resume] đã có {len(done_stems)} label → bỏ qua")
        print(f"[resume] còn {n_frames} ảnh chưa inference")
        if n_frames == 0:
            print("[resume] TẤT CẢ đã xong, không có gì để làm")
            return
    else:
        # Fresh start — xoá output cũ
        if predict_dir.exists():
            print(f"[clean] xoá {predict_dir}")
            shutil.rmtree(predict_dir)
        frame_paths = all_frame_paths
        n_frames = n_all

    # Chia thành chunks
    chunks = [frame_paths[i:i + chunk_size] for i in range(0, n_frames, chunk_size)]
    print(f"\n[plan] chia {n_frames} ảnh thành {len(chunks)} chunks (mỗi chunk {chunk_size} ảnh)")

    n_processed = 0
    n_with_detection = 0
    n_boxes_total = 0
    class_counts = {name: 0 for name in model.names.values()}

    for i, chunk in enumerate(chunks, 1):
        print(f"\n[chunk {i}/{len(chunks)}] processing {len(chunk)} ảnh...")
        np_, nd_, nb_ = run_chunk(
            model, chunk, conf, imgsz, device, batch,
            OUT_DIR, "predict", class_counts
        )
        n_processed += np_
        n_with_detection += nd_
        n_boxes_total += nb_
        print(f"  [chunk {i}] processed={np_} with_det={nd_} boxes={nb_} | "
              f"cum: {n_processed}/{n_frames}")

        # Clear GPU cache giữa chunk để tránh fragment/leak
        if device != "cpu":
            torch.cuda.empty_cache()
        gc.collect()

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
    ap.add_argument("--batch", type=int, default=1,
                    help="batch size (giảm nếu GPU OOM)")
    ap.add_argument("--chunk-size", type=int, default=500,
                    help="số ảnh/chunk — clear GPU cache giữa chunk, giảm nếu OOM")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="skip ảnh đã có label (mặc định bật)")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="tắt resume, chạy lại từ đầu + xoá output cũ")
    args = ap.parse_args()
    main(args.model, args.conf, args.imgsz, args.device, args.batch,
         args.chunk_size, args.resume)
