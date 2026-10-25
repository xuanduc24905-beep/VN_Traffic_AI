"""Chạy tracking proxy metrics trên 2 pipeline (baseline vs improved)
với video demo_traffic.mp4 và events CSV đã có sẵn.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import tracking_proxy_from_events

ROOT = Path(__file__).resolve().parent.parent
GT = json.loads((ROOT / "data/test_videos/ground_truth/demo_traffic.json").read_text())["counts_by_class"]

CANDIDATES = {
    "baseline_v1_baseline_detrac4": ROOT / "results/tables/demo_traffic_events.csv",
}

# Chạy lại nhanh trên demo video cho baseline COCO + improved fine-tune để lấy events csv
from src.tracking.counter import Line, Counter
from src.tracking.track import track_stream

LINE = Line(name="mid", p1=(100, 474), p2=(1664, 474), count_direction="both")
COCO_VEH = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
DEMO = str(ROOT / "data/test_videos/raw/demo_traffic.mp4")

def gen_events(weights: str, out_csv: str, remap_coco: bool) -> None:
    import csv as _csv
    counter = Counter(lines=[LINE])
    class_names = None
    rows = []
    for frame_idx, res in enumerate(track_stream(
            weights, DEMO, tracker="bytetrack.yaml",
            conf=0.3, iou=0.5, imgsz=640, half=True)):
        if class_names is None:
            class_names = res.names
        if res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy()
            fps = 30.0
            for ev in counter.update(frame_idx, boxes, ids, classes):
                cid = int(ev["class_id"])
                if remap_coco:
                    if cid not in COCO_VEH:
                        continue
                    cname = COCO_VEH[cid]
                else:
                    cname = class_names[cid]
                rows.append({
                    "frame": frame_idx,
                    "time_sec": round(frame_idx / fps, 3),
                    "line": ev["line"],
                    "direction": ev["direction"],
                    "track_id": int(ev["track_id"]),
                    "class_name": cname,
                })
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=["frame", "time_sec", "line",
                                            "direction", "track_id", "class_name"])
        w.writeheader()
        w.writerows(rows)
    print(f"  [save] {out_csv} ({len(rows)} rows)")


def main():
    print("[gen events] baseline (COCO)...")
    gen_events("weights/pretrained/yolov8s.pt",
               "results/tables/proxy_baseline_events.csv",
               remap_coco=True)

    print("[gen events] improved (fine-tune)...")
    gen_events("runs/detect/train_v8s_ft_vnv3/weights/best.pt",
               "results/tables/proxy_improved_events.csv",
               remap_coco=False)

    print("\n=== TRACKING PROXY METRICS trên demo_traffic.mp4 ===")
    print(f"GT: {GT}")
    for name, csv_path in [
        ("Baseline (COCO)",  "results/tables/proxy_baseline_events.csv"),
        ("Improved (v8s ft)", "results/tables/proxy_improved_events.csv"),
    ]:
        m = tracking_proxy_from_events(csv_path, GT)
        print(f"\n{name}:")
        print(f"  Overall:  track_count={m['overall']['track_count']}"
              f"  gt={m['overall']['gt']}"
              f"  overcount_rate={m['overall']['overcount_rate']}"
              f"  id_stability={m['overall']['id_stability']}"
              f"  avg_track_lifetime={m['overall']['avg_track_lifetime_frames']} frames")
        for cls, d in m["per_class"].items():
            print(f"  [{cls}] tracks={d['track_count']}"
                  f"  gt={d['gt']}"
                  f"  ovr={d['overcount_rate']}"
                  f"  stability={d['id_stability']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
