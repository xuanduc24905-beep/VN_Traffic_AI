"""Eval yolov8s.pt COCO (filter vehicle classes) vs V3 trên cùng val VN-only.

Fair comparison:
  - yolov8s.pt (COCO 80 classes) → filter COCO {2:car, 3:motorcycle, 5:bus, 7:truck}
    → remap canonical {motorcycle=0, car=1, bus=2, truck=3}
  - V3 → canonical sẵn

Compute mAP50 + mAP50-95 per class bằng torchmetrics.
"""
from pathlib import Path

import torch
from torchmetrics.detection import MeanAveragePrecision
from ultralytics import YOLO

ROOT = Path("/home/xuand/vn_traffic_ai")
VAL_IMG_DIR = ROOT / "data/merged_v3/images/val"
VAL_LBL_DIR = ROOT / "data/merged_v3/labels/val"

# COCO cls → canonical cls
COCO_TO_CANON = {2: 1, 3: 0, 5: 2, 7: 3}
CANON_NAMES = ["motorcycle", "car", "bus", "truck"]

MODELS = {
    "yolov8s_coco":  str(ROOT / "yolov8s.pt"),  # move sang weights/pretrained/ sau khi train xong
    "V1_baseline":   str(ROOT / "weights/trained/v1_baseline.pt"),
    "V2_fix":        str(ROOT / "weights/trained/v2_fix.pt"),
    "V3_vn":         str(ROOT / "weights/v3_vn_best.pt"),  # sẽ move trained/ sau khi train 1280 xong
}


def load_gt(lbl_path: Path, img_w: int, img_h: int):
    """YOLO txt → xyxy pixel + labels."""
    boxes, labels = [], []
    if not lbl_path.exists():
        return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long)
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        cls = int(parts[0])
        xc, yc, w, h = map(float, parts[1:5])
        x1 = (xc - w/2) * img_w
        y1 = (yc - h/2) * img_h
        x2 = (xc + w/2) * img_w
        y2 = (yc + h/2) * img_h
        boxes.append([x1, y1, x2, y2])
        labels.append(cls)
    return torch.tensor(boxes), torch.tensor(labels, dtype=torch.long)


def eval_model(name: str, pt_path: str, is_coco: bool) -> dict:
    print(f"\n{'='*60}\n{name}  ({pt_path})\n{'='*60}")
    model = YOLO(pt_path)

    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

    img_paths = sorted(p for p in VAL_IMG_DIR.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    batch_size = 32
    for i in range(0, len(img_paths), batch_size):
        batch = img_paths[i:i+batch_size]
        results = model.predict(
            [str(p) for p in batch],
            imgsz=640, conf=0.001, iou=0.6,
            verbose=False, device=0,
        )
        preds_batch, targets_batch = [], []
        for img_path, r in zip(batch, results):
            h, w = r.orig_shape
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu()
                scores = r.boxes.conf.cpu()
                labels = r.boxes.cls.cpu().long()
                if is_coco:
                    mask = torch.tensor([int(c) in COCO_TO_CANON for c in labels])
                    boxes, scores, labels = boxes[mask], scores[mask], labels[mask]
                    labels = torch.tensor([COCO_TO_CANON[int(c)] for c in labels],
                                          dtype=torch.long)
            else:
                boxes = torch.zeros((0, 4))
                scores = torch.zeros((0,))
                labels = torch.zeros((0,), dtype=torch.long)
            preds_batch.append({"boxes": boxes, "scores": scores, "labels": labels})

            lbl_path = VAL_LBL_DIR / f"{img_path.stem}.txt"
            gt_boxes, gt_labels = load_gt(lbl_path, w, h)
            targets_batch.append({"boxes": gt_boxes, "labels": gt_labels})
        metric.update(preds_batch, targets_batch)
        if (i // batch_size) % 10 == 0:
            print(f"  processed {i+len(batch)}/{len(img_paths)}")

    res = metric.compute()
    out = {
        "mAP50":    float(res["map_50"]),
        "mAP50-95": float(res["map"]),
        "per_class": {},
    }
    mapc = res.get("map_per_class", None)
    for i, cls in enumerate(CANON_NAMES):
        ap = float(mapc[i]) if (mapc is not None and len(mapc) > i) else None
        out["per_class"][cls] = {"mAP50-95": ap}
    print(f"  → mAP50={out['mAP50']:.4f}  mAP50-95={out['mAP50-95']:.4f}")
    for cls, v in out["per_class"].items():
        s = f"{v['mAP50-95']:.4f}" if v['mAP50-95'] is not None else "N/A"
        print(f"      {cls}: mAP50-95={s}")
    return out


def main():
    all_results = {}
    for name, pt in MODELS.items():
        is_coco = (name == "yolov8s_coco")
        all_results[name] = eval_model(name, pt, is_coco=is_coco)

    print("\n\n" + "="*70)
    print(f"{'Model':<16} {'mAP50':>7} {'mAP50-95':>9}  "
          f"{'motor95':>8} {'car95':>7} {'bus95':>7} {'truck95':>8}  (per-class mAP50-95)")
    print("-"*80)
    for name, r in all_results.items():
        pc = r["per_class"]
        def fmt(x): return f"{x:.3f}" if x is not None else "N/A"
        print(f"{name:<16} {r['mAP50']:>7.3f} {r['mAP50-95']:>9.3f}  "
              f"{fmt(pc['motorcycle']['mAP50-95']):>8} {fmt(pc['car']['mAP50-95']):>7} "
              f"{fmt(pc['bus']['mAP50-95']):>7} {fmt(pc['truck']['mAP50-95']):>8}")

    out_json = ROOT / "results/tables/eval_coco_vs_v3.json"
    import json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
