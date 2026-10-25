"""Evaluation 2 tầng cho model detection + counting.

Tầng A — Detection quality:
    YOLO.val() trên test split có label sẵn mAP@0.5, mAP@0.5:0.95, P, R per-class.
    Đây là standard ML benchmark, không cần GT thủ công.

Tầng B — Counting quality (task-level):
    Chạy pipeline (detect+track+count) trên video có GT thủ công,
    so bằng eval_report (accuracy/MAE/MAPE per-class).

Chạy:
    # So 1 model (mặc định: v8s 50e vừa train + baseline_detrac4)
    python -m src.evaluation.eval_full

    # Tuỳ chọn model list
    python -m src.evaluation.eval_full --models weights/trained/v1_baseline.pt \\
        runs/detect/train_v8s_50e/weights/best.pt

    # Bỏ tầng A (không val trên test set — nhanh)
    python -m src.evaluation.eval_full --skip-mAP
"""
import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO

from src.tracking.counter import Line
from src.pipeline.run import run_pipeline
from src.evaluation.metrics import report as eval_report


DEFAULT_MODELS = [
    "weights/trained/v1_baseline.pt",
    "weights/v3_vn_best.pt",
]

DEFAULT_VIDEO_GT = [
    ("data/test_videos/raw/demo_traffic.mp4",
     "data/test_videos/ground_truth/demo_traffic.json"),
]

DEFAULT_LINE = Line(name="mid", p1=(100, 474), p2=(1664, 474),
                    count_direction="both")


def tang_a_mAP(weights: str, data_yaml: str, imgsz: int = 640) -> dict:
    """Tầng A — mAP trên test split. Chỉ chạy nếu số class model khớp data.yaml."""
    model = YOLO(weights)
    import yaml
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    if len(model.names) != cfg["nc"]:
        return {"skipped": f"Schema mismatch: model {len(model.names)} class vs data {cfg['nc']} class"}
    m = model.val(data=data_yaml, split="test", imgsz=imgsz,
                  verbose=False, plots=False)
    return {
        "mAP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "per_class": {model.names[i]: {"mAP50": float(m.box.ap50[i]),
                                       "mAP50_95": float(m.box.ap[i])}
                      for i in range(len(model.names))
                      if i < len(m.box.ap50)},
    }


def tang_b_counting(weights: str, video: str, gt_json: str,
                    line: Line = DEFAULT_LINE,
                    conf: float = 0.3, imgsz: int = 416, half: bool = True) -> dict:
    """Tầng B — counting accuracy vs GT thủ công."""
    gt = json.load(open(gt_json))
    gt_counts = {k: v for k, v in gt["counts_by_class"].items() if v > 0}

    t0 = time.time()
    events, totals, names = run_pipeline(
        video=video, weights=weights, lines=[line],
        tracker="bytetrack.yaml",
        conf=conf, iou=0.5, imgsz=imgsz, half=half,
        out_csv=None, out_video=None,
    )
    dt = time.time() - t0

    per_class = {}
    for _, dirs in totals.items():
        for _, cls in dirs.items():
            for cid, cnt in cls.items():
                cname = names[int(cid)]
                per_class[cname] = per_class.get(cname, 0) + cnt

    rep = eval_report(per_class, gt_counts)
    return {"pred": per_class, "gt": gt_counts, "runtime_sec": round(dt, 2),
            "report": rep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--data", default="data/data.yaml")
    ap.add_argument("--skip-mAP", action="store_true")
    ap.add_argument("--out", default="results/tables/eval_full.json")
    args = ap.parse_args()

    results = {}
    for m in args.models:
        if not Path(m).exists():
            print(f"[!] Skip {m}: không tồn tại")
            continue
        print(f"\n{'=' * 60}\n>>> {m}\n{'=' * 60}")
        model_res = {"weights": m}

        if not args.skip_mAP:
            print("[A] mAP trên test set...")
            model_res["mAP"] = tang_a_mAP(m, args.data)
            if "skipped" in model_res["mAP"]:
                print(f" skip: {model_res['mAP']['skipped']}")
            else:
                a = model_res["mAP"]
                print(f" mAP@0.5={a['mAP50']:.3f} mAP@0.5-0.95={a['mAP50_95']:.3f} "
                      f"P={a['precision']:.3f} R={a['recall']:.3f}")

        print("[B] Counting trên video GT...")
        model_res["counting"] = []
        for video, gt in DEFAULT_VIDEO_GT:
            print(f" - {Path(video).name}")
            r = tang_b_counting(m, video, gt)
            model_res["counting"].append({"video": video, **r})
            rep = r["report"]
            print(f" pred={r['pred']} gt={r['gt']}")
            print(f" overall_acc={rep['overall_accuracy']:.3f} "
                  f"MAE={rep['MAE']:.2f} MAPE={rep['MAPE_percent']:.1f}% "
                  f"({r['runtime_sec']}s)")

        results[m] = model_res

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[] Lưu: {args.out}")


if __name__ == "__main__":
    main()
