"""Fair comparison: eval nhiều model trên CÙNG V5 test (unseen videos, no leak).

Models:
- V5 honest       : train trên V5 (video-aware, 2.8k frames)
- Old DETRAC+Cantho: train trên data/merged (DETRAC 60k + Cantho VN, có leak + van→truck issue)
- (TODO) v8s COCO zero-shot: pretrained COCO, remap

Output:
- Bảng mAP overall + per-class
- Confusion matrix mỗi model (save ra results/v5_cross_eval/)
"""
import json
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path("/home/xuand/vn_traffic_ai").resolve()
DATA_YAML = PROJECT_ROOT / "data" / "data_v5.yaml"
OUT_DIR = PROJECT_ROOT / "results" / "v5_cross_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "v5_honest": PROJECT_ROOT / "runs/detect/train_v8s_v5_honest/weights/best.pt",
    "old_detrac_cantho": PROJECT_ROOT / "weights/trained/best.pt",
}

NAMES = ["motorcycle", "car", "bus", "truck"]


def eval_one(label: str, weights: Path) -> dict:
    print(f"\n{'='*60}\n  EVAL: {label}  ({weights})\n{'='*60}")
    model = YOLO(str(weights))
    # imgsz=640 same cho fair comparison (ignore training imgsz)
    results = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=16,
        project=str(OUT_DIR),
        name=label,
        exist_ok=True,
        verbose=False,
        plots=True,  # saves confusion_matrix.png
    )
    rb = results.box
    # Per-class mAP50 (array indexed by class id)
    per_class = {}
    for i, name in enumerate(NAMES):
        try:
            per_class[name] = {
                "mAP50": float(rb.ap50[i]) if i < len(rb.ap50) else None,
                "mAP50-95": float(rb.ap[i]) if i < len(rb.ap) else None,
            }
        except Exception:
            per_class[name] = {"mAP50": None, "mAP50-95": None}
    return {
        "weights": str(weights),
        "mAP50": float(rb.map50),
        "mAP50-95": float(rb.map),
        "precision": float(rb.mp),
        "recall": float(rb.mr),
        "per_class": per_class,
    }


def print_summary(results: dict[str, dict]) -> None:
    print("\n\n" + "=" * 90)
    print("SUMMARY — tất cả eval trên V5 test (unseen videos, no leak)")
    print("=" * 90)
    print(f"{'Model':<22} {'mAP50':>8} {'mAP50-95':>10} {'P':>7} {'R':>7}  |  " +
          "  ".join(f"{n[:5]:>7}" for n in NAMES) + "  (mAP50 per class)")
    print("-" * 100)
    for label, r in results.items():
        pc = r["per_class"]
        pc_str = "  ".join(
            f"{(pc[n]['mAP50'] if pc[n]['mAP50'] is not None else float('nan')):>7.3f}"
            for n in NAMES
        )
        print(f"{label:<22} {r['mAP50']:>8.4f} {r['mAP50-95']:>10.4f} "
              f"{r['precision']:>7.3f} {r['recall']:>7.3f}  |  {pc_str}")


def main() -> int:
    results = {}
    for label, weights in MODELS.items():
        if not weights.exists():
            print(f"[SKIP] {label}: {weights} không tồn tại")
            continue
        results[label] = eval_one(label, weights)

    print_summary(results)

    (OUT_DIR / "summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    print(f"\n[OK] Chi tiết: {OUT_DIR / 'summary.json'}")
    print(f"[OK] Confusion matrices: {OUT_DIR}/<model>/confusion_matrix_normalized.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
