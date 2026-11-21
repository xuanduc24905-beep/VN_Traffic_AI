"""Cham lai baseline YOLOv8s-COCO (zero-shot) voi ma lop dung (0->3, 1->2, 2->5, 3->7)."""
from __future__ import annotations
import argparse, json, shutil, tempfile
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GROUP_TO_COCO = {0: 3, 1: 2, 2: 5, 3: 7}
COCO_TO_NAME = {3: "motorcycle", 2: "car", 5: "bus", 7: "truck"}
CLASS_ORDER = ["motorcycle", "car", "bus", "truck"]
DATASETS = [
    ("V5+ test (CCTV, 1.500)", "data/data_v5plus.yaml",   "test"),
    ("VN Aerial test (149)",   "data/data_vn_aerial.yaml", "test"),
    ("VN Aerial val (486)",    "data/data_vn_aerial.yaml", "val"),
    ("Extended test (2.135)",  "data/data_eval_extended.yaml", "test"),
]
IMGSZ = 960

def resolve_root(yaml_path):
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    if not root.exists():
        root = PROJECT_ROOT / "data" / root.name
    return root, cfg

def make_coco_copy(src_root, split_dir, dst_root, coco_names):
    img_src = src_root / split_dir
    lbl_src = src_root / split_dir.replace("images", "labels", 1)
    split = Path(split_dir).name
    img_dst, lbl_dst = dst_root / "images" / split, dst_root / "labels" / split
    shutil.copytree(img_src, img_dst)
    lbl_dst.mkdir(parents=True)
    n_before = {c: 0 for c in GROUP_TO_COCO}
    for f in lbl_src.glob("*.txt"):
        out = []
        for line in f.read_text().splitlines():
            p = line.split()
            if not p:
                continue
            g = int(p[0])
            if g not in GROUP_TO_COCO:
                raise ValueError(f"{f}: ma lop la {g}")
            n_before[g] += 1
            p[0] = str(GROUP_TO_COCO[g])
            out.append(" ".join(p))
        (lbl_dst / f.name).write_text("\n".join(out) + ("\n" if out else ""))
    rel = f"images/{split}"
    names = "\n".join(f"  {i}: {n}" for i, n in coco_names.items())
    (dst_root / "data.yaml").write_text(
        f"path: {dst_root.resolve().as_posix()}\ntrain: {rel}\nval: {rel}\ntest: {rel}\n"
        f"nc: {len(coco_names)}\nnames:\n{names}\n", encoding="utf-8")
    return {"images": sum(1 for _ in img_dst.iterdir()),
            "instances": {COCO_TO_NAME[GROUP_TO_COCO[g]]: n for g, n in n_before.items()}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="results/coco_baseline_remap")
    args = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(args.weights)
    coco_names = model.names
    assert len(coco_names) == 80 and all(coco_names[c] == n for c, n in COCO_TO_NAME.items())
    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for dname, data_yaml, split in DATASETS:
        src_root, cfg = resolve_root(PROJECT_ROOT / data_yaml)
        if not (src_root / cfg[split]).exists():
            print(f"[SKIP] {dname}: khong co {src_root / cfg[split]}")
            continue
        with tempfile.TemporaryDirectory(prefix="coco_remap_") as tmp:
            dst = Path(tmp) / "ds"
            stats = make_coco_copy(src_root, cfg[split], dst, coco_names)
            print(f"\n=== {dname}: {stats['images']} anh, instances {stats['instances']}")
            m = model.val(data=str(dst / "data.yaml"), split="test", imgsz=IMGSZ, batch=16,
                          device=args.device, classes=list(COCO_TO_NAME), plots=False, verbose=True)
            r = {"images": stats["images"], "instances": stats["instances"],
                 "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
                 "precision": float(m.box.mp), "recall": float(m.box.mr),
                 "AP50": {COCO_TO_NAME[int(c)]: float(v) for c, v in zip(m.box.ap_class_index, m.box.ap50)},
                 "AP50_95": {COCO_TO_NAME[int(c)]: float(v) for c, v in zip(m.box.ap_class_index, m.box.ap)}}
            results[dname] = r
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    lines = ["# YOLOv8s-COCO zero-shot - cham lai voi ma lop dung", "",
             "| Bo test | mAP50 | mAP50-95 | " + " | ".join(f"AP50 {c}" for c in CLASS_ORDER) + " |",
             "|---|---|---|" + "---|" * len(CLASS_ORDER)]
    for dname, r in results.items():
        cells = " | ".join(f"{r['AP50'].get(c, float('nan')):.4f}" for c in CLASS_ORDER)
        lines.append(f"| {dname} | {r['mAP50']:.4f} | {r['mAP50_95']:.4f} | {cells} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))

if __name__ == "__main__":
    main()
