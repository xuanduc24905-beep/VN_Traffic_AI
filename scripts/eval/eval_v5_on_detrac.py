"""Strategy D — External eval: V5 honest trên DETRAC test (cross-domain).

DETRAC là highway Trung Quốc, class khác VN:
  DETRAC 0 car   → V5 1 car
  DETRAC 1 bus   → V5 2 bus
  DETRAC 2 van   → DROP (van ≠ truck VN)
  DETRAC 3 others → DROP

Chỉ eval 2 class COMMON (car, bus) để fair.
Subsample DETRAC test (video-aware, stride 50) để nhanh (~1000 frames).

Mục đích: xem V5 (train VN) generalize sang domain highway TQ thế nào.
Nếu V5 vẫn detect được car/bus trên DETRAC → chứng minh model học
general vehicle features, không overfit VN domain.
"""
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path("/home/xuand/vn_traffic_ai").resolve()
DETRAC_IMAGES = Path("/home/xuand/yolov8_finetune/data/detrac/images/test")
DETRAC_LABELS = Path("/home/xuand/yolov8_finetune/data/detrac/labels/test")
TMP_ROOT = Path("/tmp/detrac_test_v5schema")
OUT_DIR = PROJECT_ROOT / "results" / "v5_cross_eval" / "v5_on_detrac"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# DETRAC class → V5 class (DROP van + others)
DETRAC_TO_V5 = {0: 1, 1: 2}   # car→car, bus→bus
STRIDE = 50  # subsample

VIDEO_RE = re.compile(r"^(MVI_\d+)_")


def build_tmp_dataset() -> Path:
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)
    (TMP_ROOT / "images" / "test").mkdir(parents=True)
    (TMP_ROOT / "labels" / "test").mkdir(parents=True)

    # Group by video, apply stride
    per_video: dict[str, list[Path]] = defaultdict(list)
    for img in sorted(DETRAC_IMAGES.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        m = VIDEO_RE.match(img.name)
        if m:
            per_video[m.group(1)].append(img)

    kept_imgs = 0
    kept_boxes = 0
    for vid, frames in per_video.items():
        selected = frames[::STRIDE]
        for img in selected:
            label_src = DETRAC_LABELS / (img.stem + ".txt")
            if not label_src.exists():
                continue
            # Remap labels
            new_lines = []
            for line in label_src.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                if cls not in DETRAC_TO_V5:
                    continue
                new_lines.append(f"{DETRAC_TO_V5[cls]} " + " ".join(parts[1:]))
                kept_boxes += 1
            if not new_lines:
                continue  # skip empty
            # Symlink image, write remapped label
            (TMP_ROOT / "images" / "test" / img.name).symlink_to(img.resolve())
            (TMP_ROOT / "labels" / "test" / (img.stem + ".txt")).write_text(
                "\n".join(new_lines)
            )
            kept_imgs += 1

    yaml_body = f"""path: {TMP_ROOT}
train: images/test
val:   images/test
test:  images/test
nc: 4
names:
  0: motorcycle
  1: car
  2: bus
  3: truck
"""
    (TMP_ROOT / "data.yaml").write_text(yaml_body)
    print(f"[OK] DETRAC subsample: {len(per_video)} videos → {kept_imgs} frames, "
          f"{kept_boxes} boxes (car+bus only)")
    return TMP_ROOT / "data.yaml"


def main() -> int:
    yaml_path = build_tmp_dataset()

    model = YOLO(str(PROJECT_ROOT / "runs/detect/train_v8s_v5_honest/weights/best.pt"))
    r = model.val(
        data=str(yaml_path),
        split="test",
        imgsz=640,
        batch=16,
        project=str(OUT_DIR.parent),
        name="v5_on_detrac",
        exist_ok=True,
        verbose=False,
        plots=True,
    )
    rb = r.box

    print("\n" + "=" * 60)
    print("V5 honest trên DETRAC test (cross-domain, car+bus only)")
    print("=" * 60)
    print(f"Overall mAP50:    {float(rb.map50):.4f}")
    print(f"Overall mAP50-95: {float(rb.map):.4f}")
    print(f"Precision:        {float(rb.mp):.4f}")
    print(f"Recall:           {float(rb.mr):.4f}")

    # Per-class
    print(f"\nPer-class (car+bus, 2 class common):")
    for i, idx in enumerate(rb.ap_class_index):
        name = ["motorcycle", "car", "bus", "truck"][int(idx)]
        print(f"  {name:<12}  mAP50={float(rb.ap50[i]):.4f}  mAP50-95={float(rb.ap[i]):.4f}")

    summary = {
        "model": "V5 honest best.pt",
        "test_set": f"DETRAC test subsample 1/{STRIDE} (car+bus only)",
        "mAP50": float(rb.map50),
        "mAP50-95": float(rb.map),
        "precision": float(rb.mp),
        "recall": float(rb.mr),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"\n[OK] {OUT_DIR / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
