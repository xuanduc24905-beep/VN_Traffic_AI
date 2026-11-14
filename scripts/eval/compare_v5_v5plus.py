"""So sánh V5 baseline vs V5+ warm-start trên cùng V5 test set.

Eval 2 model trên data_v5plus.yaml (test giống V5) → in bảng mAP + AP per class.
Lưu kết quả JSON vào docs/report/assets/v5_vs_v5plus.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path("/home/xuand/vn_traffic_ai").resolve()
DATA_YAML = PROJECT_ROOT / "data" / "data_v5plus.yaml"
OUT_JSON = PROJECT_ROOT / "docs" / "report" / "assets" / "v5_vs_v5plus.json"

MODELS = [
    ("V5 baseline",    PROJECT_ROOT / "runs" / "detect" / "train_v8s_v5_honest" / "weights" / "best.pt"),
    ("V5+ warm-start", PROJECT_ROOT / "runs" / "detect" / "train_v8s_v5plus_warm" / "weights" / "best.pt"),
]
CLASSES = ["motorcycle", "car", "bus", "truck"]


def main() -> int:
    results: dict[str, dict] = {}

    for name, weights in MODELS:
        if not weights.exists():
            print(f"[SKIP] {name}: {weights} không tồn tại")
            continue
        print(f"\n--- Eval {name} on V5 test ---")
        m = YOLO(str(weights)).val(
            data=str(DATA_YAML),
            split="test",
            imgsz=960,
            batch=16,
            verbose=False,
        )
        results[name] = {
            "weights": str(weights),
            "mAP50": float(m.box.map50),
            "mAP50_95": float(m.box.map),
            "AP50": {
                cls: float(m.box.ap50[i]) if i < len(m.box.ap50) else None
                for i, cls in enumerate(CLASSES)
            },
        }

    # Bảng so sánh
    print("\n" + "=" * 80)
    header = f"{'Model':<20} {'mAP50':>10} {'mAP50-95':>10} " + " ".join(f"{c:>10}" for c in CLASSES)
    print(header)
    print("-" * 80)
    for name, r in results.items():
        ap = r["AP50"]
        def fmt(x): return f"{x:.3f}" if x is not None else "n/a"
        row = f"{name:<20} {r['mAP50']:>10.4f} {r['mAP50_95']:>10.4f} "
        row += " ".join(f"{fmt(ap[c]):>10}" for c in CLASSES)
        print(row)
    print("=" * 80)

    # Delta (nếu có đủ 2 model)
    if len(results) == 2:
        v5 = results["V5 baseline"]
        v5p = results["V5+ warm-start"]
        print(f"\nΔ V5+ − V5:")
        print(f"  mAP50:    {(v5p['mAP50'] - v5['mAP50']):+.4f}")
        print(f"  mAP50-95: {(v5p['mAP50_95'] - v5['mAP50_95']):+.4f}")
        for c in CLASSES:
            if v5["AP50"][c] is not None and v5p["AP50"][c] is not None:
                print(f"  AP50 {c:<12}: {(v5p['AP50'][c] - v5['AP50'][c]):+.4f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nĐã lưu: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
