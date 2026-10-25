"""Chạy 2 pipeline trên demo_traffic.mp4 và lưu kết quả JSON để nhét vào báo cáo.

Baseline : yolov8s.pt COCO (80 lớp — filter 4 vehicle)
Improved : train_v8s_ft_vnv3/weights/best.pt (4 lớp VN)
"""
from __future__ import annotations
import json
import time
from collections import defaultdict
from pathlib import Path

from src.tracking.counter import Line, Counter
from src.tracking.track import track_stream
from src.evaluation.metrics import report as eval_report

ROOT = Path(__file__).resolve().parent.parent
DEMO = str(ROOT / "data/test_videos/raw/demo_traffic.mp4")
GT = json.loads((ROOT / "data/test_videos/ground_truth/demo_traffic.json").read_text())["counts_by_class"]

LINE = Line(name="mid", p1=(100, 474), p2=(1664, 474), count_direction="both")
COCO_VEH = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def run(weights: str, remap_coco: bool) -> dict:
    counter = Counter(lines=[LINE])
    class_names = None
    pred = defaultdict(int)
    t0 = time.time()
    n_frames = 0
    for frame_idx, res in enumerate(track_stream(
            weights, DEMO, tracker="bytetrack.yaml",
            conf=0.3, iou=0.5, imgsz=640, half=True)):
        n_frames += 1
        if class_names is None:
            class_names = res.names
        if res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy()
            evs = counter.update(frame_idx, boxes, ids, classes)
            for ev in evs:
                cid = int(ev["class_id"])
                if remap_coco:
                    if cid in COCO_VEH:
                        pred[COCO_VEH[cid]] += 1
                else:
                    pred[class_names[int(cid)]] += 1
    dt = time.time() - t0
    return {
        "weights": weights,
        "pred": dict(pred),
        "runtime_sec": round(dt, 2),
        "frames": n_frames,
        "fps": round(n_frames / dt, 1),
    }


def main():
    print("[1/2] Baseline (yolov8s.pt COCO)...")
    bl = run("weights/pretrained/yolov8s.pt", remap_coco=True)
    bl["report"] = eval_report(bl["pred"], GT)

    print("[2/2] Improved (train_v8s_ft_vnv3/best.pt)...")
    im = run("runs/detect/train_v8s_ft_vnv3/weights/best.pt", remap_coco=False)
    im["report"] = eval_report(im["pred"], GT)

    out = ROOT / "results/tables/eval_counting_2pipelines.json"
    out.write_text(json.dumps({
        "gt": GT,
        "baseline": bl,
        "improved": im,
    }, indent=2, ensure_ascii=False))
    print(f"[save] {out}")

    for name, r in [("Baseline", bl), ("Improved", im)]:
        print(f"\n{name}:")
        print(f"  pred    : {r['pred']}")
        print(f"  runtime : {r['runtime_sec']}s ({r['fps']} FPS)")
        rep = r["report"]
        print(f"  MAE     : {rep['MAE']}")
        print(f"  MAPE(%) : {rep['MAPE_percent']:.2f}")
        print(f"  overall acc: {rep['overall_accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()
