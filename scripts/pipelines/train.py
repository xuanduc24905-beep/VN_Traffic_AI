"""Huấn luyện dùng chung cho các pipeline 1-4: một công thức = một tên, cấu hình nền là V5+.

Công thức nền (V5+): yolov8s, imgsz 960, 40 epoch, batch 16, lr0 0,005, cosine LR, SGD, mixup 0,1,
seed 42. copy_paste không có tác dụng với nhãn hộp nên đặt 0. Mỗi công thức chỉ đổi một nhóm
tham số so với nền, để so sánh được.

  v5plus      công thức V5+ từ trọng số COCO (yolov8s.pt)
  finetune    tiếp tục từ best.pt của V5+: 20 epoch, lr0 0,0005 (giá trị khởi điểm, chưa kiểm chứng)
  imgsz1280   imgsz 1280, batch 8
  multiscale  multi_scale 0,25 (đổi kích thước ảnh khi train)
  clspw       cls_pw 0,5 (tăng trọng số lớp hiếm: bus, truck)
  scale       scale 0,7 và translate 0,15 (xe ở xa)
  p2          kiến trúc yolov8s-p2 (thêm đầu stride 4), nạp trọng số COCO phần khớp
  progressive 2 giai đoạn: 960 rồi 1280 (giai đoạn 2 bắt đầu từ best.pt của giai đoạn 1)

Chưng cất đặc trưng (pipeline 3): thêm --distill-model <giáo_viên.pt> [--dis 6.0]. Giáo viên và học
sinh phải cùng họ YOLO (cả hai YOLOv8). Cần bản Ultralytics có tham số distill_model.

Ví dụ:
  python scripts/pipelines/train.py --data data/exp1/pseudo.yaml --name p1_mask --recipe finetune --device 0
  python scripts/pipelines/train.py --data data/data_v5.yaml --name r_clspw --recipe clspw --device 0
  python scripts/pipelines/train.py --data data/data_v5.yaml --name smoke --recipe v5plus --fraction 0.05 --epochs 2
  python scripts/pipelines/train.py ... --dry-run          # chỉ in cấu hình, không huấn luyện
  python scripts/pipelines/train.py ... --name p1_mask --resume   # chạy tiếp khi Colab bị ngắt

Sau huấn luyện tự đánh giá trên val và ghi vào runs/detect/pipelines_summary.csv.
Đánh giá test chỉ làm một lần, sau khi đã chốt mô hình (xem HUONG_DAN.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO  # noqa: E402
from v8_experiments import BASE, evaluate, local_data_yaml  # noqa: E402

V5P_BEST = REPO / "runs/detect/train_v8s_v5plus_960/weights/best.pt"

NEUTRAL = dict(BASE, copy_paste=0.0)
RECIPES = {
    "v5plus": dict(over={}, arch="yolov8s.pt", init="yolov8s.pt"),
    "finetune": dict(over=dict(epochs=20, lr0=0.0005, lrf=0.1, close_mosaic=10, patience=10),
                     arch=None, init=str(V5P_BEST)),
    "imgsz1280": dict(over=dict(imgsz=1280, batch=8), arch="yolov8s.pt", init="yolov8s.pt"),
    "multiscale": dict(over=dict(multi_scale=0.25), arch="yolov8s.pt", init="yolov8s.pt"),
    "clspw": dict(over=dict(cls_pw=0.5), arch="yolov8s.pt", init="yolov8s.pt"),
    "scale": dict(over=dict(scale=0.7, translate=0.15), arch="yolov8s.pt", init="yolov8s.pt"),
    "p2": dict(over={}, arch="yolov8s-p2.yaml", init="yolov8s.pt"),
    "progressive": dict(over={}, arch="yolov8s.pt", init="yolov8s.pt"),
}


def build_cfg(args, data_yaml: Path, runs_dir: Path, name: str, recipe: dict) -> dict:
    cfg = dict(NEUTRAL)
    cfg.update(recipe["over"])
    for k in ("epochs", "batch", "imgsz", "fraction", "seed"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.distill_model:
        cfg["distill_model"] = args.distill_model
        cfg["dis"] = args.dis
    for kv in args.set or []:
        k, _, v = kv.partition("=")
        if not k or not _:
            raise SystemExit(f"--set cần dạng khoá=giá_trị, nhận: {kv}")
        cfg[k] = yaml.safe_load(v)
    cfg.update(data=str(data_yaml), project=str(runs_dir), name=name, exist_ok=True, verbose=True,
               device=args.device)
    return cfg


def check_keys(cfg: dict) -> None:
    from ultralytics.cfg import DEFAULT_CFG_DICT
    bad = [k for k in cfg if k not in DEFAULT_CFG_DICT]
    if bad:
        import ultralytics
        raise SystemExit(f"Bản Ultralytics {ultralytics.__version__} không có tham số {bad}. "
                         f"Cập nhật: pip install -U ultralytics (hoặc bỏ tham số đó).")


def append_row(path: Path, row: dict) -> None:
    import pandas as pd
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--name", required=True, help="tên lần chạy; kết quả ở runs/detect/<name>")
    ap.add_argument("--recipe", choices=list(RECIPES), default="v5plus")
    ap.add_argument("--init", default=None, help="trọng số khởi tạo (mặc định theo công thức)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--epochs2", type=int, default=15, help="số epoch giai đoạn 2 của 'progressive'")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--fraction", type=float, default=None, help="dùng một phần tập train để chạy thử")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--distill-model", default=None)
    ap.add_argument("--dis", type=float, default=6.0)
    ap.add_argument("--set", nargs="*", metavar="khoá=giá_trị", help="ghi đè tham số Ultralytics bất kỳ")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    recipe = RECIPES[a.recipe]
    runs_dir = REPO / "runs" / "detect"
    runs_dir.mkdir(parents=True, exist_ok=True)
    if not a.data.exists():
        ap.error(f"không thấy {a.data}")
    data_yaml = local_data_yaml(a.data.resolve(), REPO, runs_dir)
    cfg = build_cfg(a, data_yaml, runs_dir, a.name, recipe)
    init = a.init or recipe["init"]
    check_keys(cfg)

    plan = {"recipe": a.recipe, "name": a.name, "init": init, "arch": recipe["arch"], "data": str(data_yaml),
            "khác_nền": {k: v for k, v in cfg.items() if NEUTRAL.get(k, "∅") != v
                         and k not in ("data", "project", "name", "exist_ok", "verbose", "device")}}
    if a.recipe == "progressive":
        plan["giai_đoạn_2"] = {"imgsz": 1280, "batch": 8, "epochs": a.epochs2, "lr0": 0.002, "init": f"{a.name}_s1/best.pt"}
    print(json.dumps(plan, ensure_ascii=False, indent=2, default=str))
    if a.dry_run:
        return 0
    if not Path(init).exists() and not init.endswith(".pt"):
        ap.error(f"không thấy trọng số khởi tạo: {init}")
    if recipe["init"] == str(V5P_BEST) and not Path(init).exists():
        ap.error(f"công thức finetune cần {V5P_BEST} (tải ở GitHub Releases v1.0-weights) hoặc truyền --init")

    from ultralytics import YOLO
    best = runs_dir / a.name / "weights" / "best.pt"

    if a.resume:
        last = runs_dir / a.name / "weights" / "last.pt"
        if not last.exists():
            ap.error(f"không có {last} để chạy tiếp")
        YOLO(str(last)).train(resume=True)
    elif a.recipe == "progressive":
        s1 = dict(cfg, name=f"{a.name}_s1")
        YOLO(init).train(**s1)
        s1_best = runs_dir / f"{a.name}_s1" / "weights" / "best.pt"
        s2 = dict(cfg, imgsz=1280, batch=8, epochs=a.epochs2, lr0=0.002, close_mosaic=5)
        YOLO(str(s1_best)).train(**s2)
    else:
        arch = recipe["arch"]
        model = YOLO(arch).load(init) if (arch and arch.endswith(".yaml")) else YOLO(init)
        model.train(**cfg)

    if not a.no_eval:
        res = evaluate(best, data_yaml, "val", 1280 if a.recipe == "progressive" else cfg["imgsz"], a.device)
        row = {"name": a.name, "recipe": a.recipe, "data": str(a.data), "init": init, **res}
        append_row(runs_dir / "pipelines_summary.csv", row)
        print("[VAL]", res)
    print(f"[xong] {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
