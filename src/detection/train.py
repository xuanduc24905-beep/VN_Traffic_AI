"""Người 1 — Train YOLOv8 trên merged dataset 5 lớp.

Chạy:
    # Train từ đầu
    python -m src.detection.train --data data/data.yaml --epochs 50 --batch 16

    # Train tiếp từ checkpoint (dừng giữa chừng do tắt máy / Ctrl+C)
    python -m src.detection.train --resume --name train_v8s_50e

Cơ chế resume:
    - Ultralytics tự save `last.pt` mỗi epoch vào `runs/detect/{name}/weights/`
    - Khi --resume: bỏ qua --pretrained, load thẳng last.pt, tiếp tục epoch dừng
    - Tổng epochs, batch, imgsz... đọc từ args.yaml của run cũ (không cần khai báo lại)
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data.yaml")
    ap.add_argument("--pretrained", default="weights/pretrained/yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="train")
    ap.add_argument("--cache", default="disk", choices=["ram", "disk", "false"])
    ap.add_argument("--save-period", type=int, default=-1,
                    help="Save checkpoint riêng mỗi N epoch (epoch5.pt, epoch10.pt...). "
                         "-1 = chỉ giữ last.pt + best.pt (tiết kiệm ổ đĩa, đủ để resume).")
    ap.add_argument("--resume", action="store_true",
                    help="Tiếp tục train từ last.pt của run cùng --name (dừng do tắt máy/Ctrl+C).")
    args = ap.parse_args()

    if args.resume:
        # Ultralytics có thể save vào nhiều vị trí do settings.runs_dir tự prepend.
        # Glob tất cả last.pt trùng --name, chọn cái mới nhất.
        matches = sorted(
            Path(".").rglob(f"{args.name}/weights/last.pt"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if not matches:
            raise FileNotFoundError(
                f"Không tìm thấy last.pt cho run '{args.name}'. "
                f"Chạy `find . -name last.pt` để kiểm tra."
            )
        last_pt = matches[0]
        if len(matches) > 1:
            print(f"[!] Nhiều last.pt cho '{args.name}':")
            for m in matches:
                print(f" {m} ({m.stat().st_mtime})")
            print(f"[i] Chọn cái mới nhất: {last_pt}")
        print(f"[↻] Resume từ: {last_pt}")
        model = YOLO(str(last_pt))
        model.train(resume=True)
        print(f"[] Best weight: {last_pt.parent / 'best.pt'}")
        return

    model = YOLO(args.pretrained)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        cache=(args.cache if args.cache != "false" else False),
        save=True,
        save_period=args.save_period,
        seed=42,
        exist_ok=True,
    )
    print(f"[] Best weight: {Path(args.project) / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
