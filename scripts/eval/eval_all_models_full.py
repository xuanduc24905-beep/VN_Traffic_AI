"""Full comparative eval — tất cả model trên tất cả test set.

Models:
- v8s_coco_zeroshot (yolov8s.pt pretrained, label remap V5→COCO)
- old_detrac_cantho (weights/trained/best.pt, DETRAC+Cantho merge)
- v5_honest (train_v8s_v5_honest/best.pt, 640)
- v5_plus_960 (train_v8s_v5plus_960/best.pt, 960)
- v6_pseudo (train_v8s_v6_pseudo/best.pt, V5+pseudo vn_cctv)

Test sets:
- v5_test: 1500 frames VN urban (unseen videos)
- vn_cctv_test: 2919 frames Đà Nẵng CCTV (OOD, held out hoàn toàn)
- detrac_test: subsample DETRAC Chinese highway (cross-domain)

Skip model nào không tồn tại (vd V5+ chưa train) để script chạy được từng phase.
Output: results/v5_cross_eval/FINAL_COMPARISON.md + per-model JSON.
"""
from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path("/home/xuand/vn_traffic_ai").resolve()
OUT_DIR = PROJECT_ROOT / "results" / "v5_cross_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)
NAMES = ["motorcycle", "car", "bus", "truck"]

MODELS = {
    "v5_honest":        PROJECT_ROOT / "runs/detect/train_v8s_v5_honest/weights/best.pt",
    "v5_plus_960":      PROJECT_ROOT / "runs/detect/train_v8s_v5plus_960/weights/best.pt",
    "v6_pseudo":        PROJECT_ROOT / "runs/detect/train_v8s_v6_pseudo/weights/best.pt",
    "old_detrac_cantho":PROJECT_ROOT / "weights/trained/best.pt",
}

# yolov8s COCO zero-shot cần dataset remapped tạm
V5_IMAGES = PROJECT_ROOT / "data/merged_v5/images/test"
V5_LABELS = PROJECT_ROOT / "data/merged_v5/labels/test"
V5_YAML   = PROJECT_ROOT / "data/data_v5.yaml"
VN_CCTV_IMAGES = PROJECT_ROOT / "data/raw_vn_cctv/test/images"
VN_CCTV_LABELS = PROJECT_ROOT / "data/raw_vn_cctv/test/labels"
DETRAC_IMAGES = Path("/home/xuand/yolov8_finetune/data/detrac/images/test")
DETRAC_LABELS = Path("/home/xuand/yolov8_finetune/data/detrac/labels/test")

V5_TO_COCO = {0: 3, 1: 2, 2: 5, 3: 7}
DETRAC_TO_V5 = {0: 1, 1: 2}
VIDEO_RE_DETRAC = re.compile(r"^(MVI_\d+)_")

COCO_80 = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]


# ---------- Helpers to build temporary eval datasets ----------

def build_vncctv_test_tmp() -> Path:
    """Prepare vn_cctv_test as a YOLO val dataset."""
    tmp = Path("/tmp/vn_cctv_test_eval")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "images" / "test").mkdir(parents=True)
    (tmp / "labels" / "test").mkdir(parents=True)
    for img in VN_CCTV_IMAGES.iterdir():
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl = VN_CCTV_LABELS / (img.stem + ".txt")
        if not lbl.exists():
            continue
        (tmp / "images" / "test" / img.name).symlink_to(img.resolve())
        (tmp / "labels" / "test" / (img.stem + ".txt")).symlink_to(lbl.resolve())
    (tmp / "data.yaml").write_text(
        f"path: {tmp}\ntrain: images/test\nval: images/test\ntest: images/test\n"
        f"nc: 4\nnames:\n  0: motorcycle\n  1: car\n  2: bus\n  3: truck\n"
    )
    return tmp / "data.yaml"


def build_detrac_test_tmp(stride: int = 50) -> Path:
    tmp = Path("/tmp/detrac_test_v5schema_eval")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "images" / "test").mkdir(parents=True)
    (tmp / "labels" / "test").mkdir(parents=True)
    per_video = defaultdict(list)
    for img in sorted(DETRAC_IMAGES.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        m = VIDEO_RE_DETRAC.match(img.name)
        if m:
            per_video[m.group(1)].append(img)
    for vid, frames in per_video.items():
        for img in frames[::stride]:
            lbl = DETRAC_LABELS / (img.stem + ".txt")
            if not lbl.exists():
                continue
            new_lines = []
            for line in lbl.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                c = int(parts[0])
                if c not in DETRAC_TO_V5:
                    continue
                new_lines.append(f"{DETRAC_TO_V5[c]} " + " ".join(parts[1:]))
            if not new_lines:
                continue
            (tmp / "images" / "test" / img.name).symlink_to(img.resolve())
            (tmp / "labels" / "test" / (img.stem + ".txt")).write_text(
                "\n".join(new_lines)
            )
    (tmp / "data.yaml").write_text(
        f"path: {tmp}\ntrain: images/test\nval: images/test\ntest: images/test\n"
        f"nc: 4\nnames:\n  0: motorcycle\n  1: car\n  2: bus\n  3: truck\n"
    )
    return tmp / "data.yaml"


def build_v5test_cocoids_tmp() -> Path:
    tmp = Path("/tmp/v5_test_cocoids_eval")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "images" / "test").mkdir(parents=True)
    (tmp / "labels" / "test").mkdir(parents=True)
    for img in V5_IMAGES.iterdir():
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        (tmp / "images" / "test" / img.name).symlink_to(img.resolve())
    for lbl in V5_LABELS.iterdir():
        if lbl.suffix.lower() != ".txt":
            continue
        new_lines = []
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            v = int(parts[0])
            if v not in V5_TO_COCO:
                continue
            new_lines.append(f"{V5_TO_COCO[v]} " + " ".join(parts[1:]))
        (tmp / "labels" / "test" / lbl.name).write_text("\n".join(new_lines))
    (tmp / "data.yaml").write_text(
        f"path: {tmp}\ntrain: images/test\nval: images/test\ntest: images/test\n"
        f"nc: 80\nnames: {COCO_80}\n"
    )
    return tmp / "data.yaml"


# ---------- Eval wrappers ----------

def eval_model(label: str, weights: Path, yaml_path: Path,
               imgsz: int = 640, tta: bool = False) -> dict:
    print(f"\n>>> EVAL {label} on {yaml_path.parent.name} (imgsz={imgsz}, tta={tta})")
    model = YOLO(str(weights))
    r = model.val(
        data=str(yaml_path),
        split="test",
        imgsz=imgsz,
        batch=16,
        project=str(OUT_DIR),
        name=f"{label}__on__{yaml_path.parent.name}",
        exist_ok=True,
        verbose=False,
        plots=True,
        augment=tta,
    )
    rb = r.box
    per_class = {}
    for i, cid in enumerate(rb.ap_class_index):
        try:
            name = NAMES[int(cid)]
        except Exception:
            name = f"class_{int(cid)}"
        per_class[name] = {
            "mAP50": float(rb.ap50[i]),
            "mAP50-95": float(rb.ap[i]),
        }
    return {
        "overall_mAP50": float(rb.map50),
        "overall_mAP50_95": float(rb.map),
        "precision": float(rb.mp),
        "recall": float(rb.mr),
        "per_class": per_class,
    }


def eval_v8s_zeroshot(yaml_coco_path: Path, imgsz: int = 640) -> dict:
    """Zero-shot v8s COCO, trích mAP cho 4 class COCO quan tâm."""
    print(f"\n>>> EVAL v8s_coco_zeroshot on v5_test (imgsz={imgsz})")
    model = YOLO(str(PROJECT_ROOT / "yolov8s.pt"))
    r = model.val(
        data=str(yaml_coco_path),
        split="test",
        imgsz=imgsz,
        batch=16,
        project=str(OUT_DIR),
        name="v8s_coco_zeroshot__on__v5_test",
        exist_ok=True,
        verbose=False,
        plots=False,
    )
    rb = r.box
    interest = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    per_class = {}
    for coco_id, name in interest.items():
        try:
            idx = list(rb.ap_class_index).index(coco_id)
            per_class[name] = {
                "mAP50": float(rb.ap50[idx]),
                "mAP50-95": float(rb.ap[idx]),
            }
        except ValueError:
            per_class[name] = {"mAP50": 0.0, "mAP50-95": 0.0}
    maps50 = [v["mAP50"] for v in per_class.values()]
    maps = [v["mAP50-95"] for v in per_class.values()]
    return {
        "overall_mAP50": sum(maps50) / len(maps50),
        "overall_mAP50_95": sum(maps) / len(maps),
        "precision": None,
        "recall": None,
        "per_class": per_class,
    }


# ---------- Reporting ----------

def write_markdown(all_results: dict) -> None:
    """all_results = {model_label: {testset_label: result_dict}}"""
    md = ["# Final comparison — all models × all test sets",
          "",
          "Mỗi cell là `mAP50 / mAP50-95`.",
          "",
          "## Overall",
          "",
          "| Model | V5 test (VN urban) | vn_cctv test (OOD CCTV) | DETRAC test (cross-domain) |",
          "|---|---|---|---|"]

    def cell(r):
        if r is None:
            return "—"
        return f"**{r['overall_mAP50']:.3f}** / {r['overall_mAP50_95']:.3f}"

    for model, by_test in all_results.items():
        md.append(f"| {model} | {cell(by_test.get('v5_test'))} | "
                  f"{cell(by_test.get('vn_cctv_test'))} | "
                  f"{cell(by_test.get('detrac_test'))} |")

    md += ["", "## Per-class mAP50 trên V5 test", "",
           "| Model | motorcycle | car | bus | truck |",
           "|---|---|---|---|---|"]
    for model, by_test in all_results.items():
        r = by_test.get("v5_test")
        if r is None:
            md.append(f"| {model} | — | — | — | — |")
            continue
        pc = r.get("per_class", {})
        row = [model]
        for c in NAMES:
            v = pc.get(c, {}).get("mAP50")
            row.append(f"{v:.3f}" if v is not None else "—")
        md.append("| " + " | ".join(row) + " |")

    (OUT_DIR / "FINAL_COMPARISON.md").write_text("\n".join(md))


def main() -> int:
    all_results: dict = {}

    # Build test yamls
    print("[SETUP] Build temporary test datasets...")
    v5_yaml = V5_YAML
    vncctv_yaml = build_vncctv_test_tmp() if VN_CCTV_IMAGES.exists() else None
    detrac_yaml = build_detrac_test_tmp() if DETRAC_IMAGES.exists() else None
    v5_coco_yaml = build_v5test_cocoids_tmp()

    # v8s COCO zero-shot (chỉ trên V5 test)
    try:
        all_results["v8s_coco_zeroshot"] = {
            "v5_test": eval_v8s_zeroshot(v5_coco_yaml),
        }
    except Exception as e:
        print(f"[WARN] v8s_coco_zeroshot fail: {e}")

    # Các trained model
    for label, weights in MODELS.items():
        if not weights.exists():
            print(f"[SKIP] {label}: {weights} chưa tồn tại")
            continue
        imgsz = 960 if ("v5plus" in str(weights) or "v6_pseudo" in str(weights)) else 640
        model_results = {}
        # V5 test (chính)
        try:
            model_results["v5_test"] = eval_model(label, weights, v5_yaml, imgsz=imgsz)
        except Exception as e:
            print(f"[WARN] {label} on v5_test fail: {e}")
        # vn_cctv test (OOD)
        if vncctv_yaml:
            try:
                model_results["vn_cctv_test"] = eval_model(
                    label, weights, vncctv_yaml, imgsz=imgsz
                )
            except Exception as e:
                print(f"[WARN] {label} on vn_cctv fail: {e}")
        # DETRAC test (cross-domain)
        if detrac_yaml:
            try:
                model_results["detrac_test"] = eval_model(
                    label, weights, detrac_yaml, imgsz=imgsz
                )
            except Exception as e:
                print(f"[WARN] {label} on detrac fail: {e}")
        all_results[label] = model_results

    # Save
    (OUT_DIR / "FINAL_COMPARISON.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False)
    )
    write_markdown(all_results)
    print(f"\n[OK] Chi tiết: {OUT_DIR / 'FINAL_COMPARISON.json'}")
    print(f"[OK] Markdown: {OUT_DIR / 'FINAL_COMPARISON.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
