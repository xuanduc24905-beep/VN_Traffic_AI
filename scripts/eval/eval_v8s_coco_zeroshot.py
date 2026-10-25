"""Benchmark yolov8s.pt (COCO zero-shot) trên V5 test.

Cách làm: tạo bộ eval tạm trong /tmp/ với
- images: symlink V5 test images
- labels: REMAP V5 class ID → COCO class ID
    V5 0 motorcycle → COCO 3 motorcycle
    V5 1 car        → COCO 2 car
    V5 2 bus        → COCO 5 bus
    V5 3 truck      → COCO 7 truck
- data.yaml: 80 COCO classes

Run `yolov8s.pt val` → extract per-class mAP cho 4 class quan tâm → mean.

Note: Đo zero-shot nên không có finetune. Số sẽ cho thấy baseline COCO
yếu/mạnh thế nào trên domain VN, delta với V5 honest = contribution của
VN domain adaptation.
"""
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path("/home/xuand/vn_traffic_ai").resolve()
V5_IMAGES = PROJECT_ROOT / "data/merged_v5/images/test"
V5_LABELS = PROJECT_ROOT / "data/merged_v5/labels/test"
TMP_ROOT = Path("/tmp/v5_test_cocoids")
OUT_DIR = PROJECT_ROOT / "results" / "v5_cross_eval" / "v8s_coco_zeroshot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# V5 class ID → COCO class ID
V5_TO_COCO = {0: 3, 1: 2, 2: 5, 3: 7}
# COCO class ID → semantic name (chỉ cần cho 4 class quan tâm)
COCO_INTEREST = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Full COCO 80 classes (standard order)
COCO_80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def build_tmp_dataset() -> Path:
    """Create /tmp/v5_test_cocoids/ với labels remapped to COCO IDs."""
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)
    (TMP_ROOT / "images" / "test").mkdir(parents=True)
    (TMP_ROOT / "labels" / "test").mkdir(parents=True)

    # Symlink images
    for img in V5_IMAGES.iterdir():
        if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            (TMP_ROOT / "images" / "test" / img.name).symlink_to(img.resolve())

    # Remap labels
    kept = dropped = 0
    for lbl in V5_LABELS.iterdir():
        if lbl.suffix.lower() != ".txt":
            continue
        new_lines = []
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            v5_id = int(parts[0])
            if v5_id not in V5_TO_COCO:
                dropped += 1
                continue
            new_cls = V5_TO_COCO[v5_id]
            new_lines.append(f"{new_cls} " + " ".join(parts[1:]))
            kept += 1
        (TMP_ROOT / "labels" / "test" / lbl.name).write_text("\n".join(new_lines))

    # Build YAML
    yaml_path = TMP_ROOT / "data.yaml"
    yaml_body = f"""path: {TMP_ROOT}
test: images/test
val:  images/test
train: images/test  # dummy, không dùng
nc: 80
names: {COCO_80}
"""
    yaml_path.write_text(yaml_body)

    print(f"[OK] Tmp dataset tại {TMP_ROOT}")
    print(f"     Labels remapped: {kept} boxes giữ, {dropped} dropped")
    return yaml_path


def main() -> int:
    print("=== Build tmp COCO-ID dataset ===")
    yaml_path = build_tmp_dataset()

    print("\n=== Run yolov8s.pt zero-shot val ===")
    model = YOLO(str(PROJECT_ROOT / "yolov8s.pt"))
    r = model.val(
        data=str(yaml_path),
        split="test",
        imgsz=640,
        batch=16,
        project=str(OUT_DIR.parent),
        name="v8s_coco_zeroshot",
        exist_ok=True,
        verbose=False,
        plots=False,  # confusion matrix 80x80 không hữu ích
    )
    rb = r.box

    # Extract per-class mAP cho 4 COCO classes quan tâm
    per_class = {}
    for coco_id, name in COCO_INTEREST.items():
        # rb.ap_class_index chứa list class IDs mà có data
        # rb.ap50 song song với ap_class_index
        try:
            idx_in_results = list(rb.ap_class_index).index(coco_id)
            per_class[name] = {
                "mAP50": float(rb.ap50[idx_in_results]),
                "mAP50-95": float(rb.ap[idx_in_results]),
            }
        except (ValueError, IndexError):
            per_class[name] = {"mAP50": 0.0, "mAP50-95": 0.0, "note": "no detections"}

    # Mean cho 4 class quan tâm (fair comparison với V5 schema)
    maps50 = [v["mAP50"] for v in per_class.values()]
    maps = [v["mAP50-95"] for v in per_class.values()]
    mean4_mAP50 = sum(maps50) / len(maps50) if maps50 else 0
    mean4_mAP = sum(maps) / len(maps) if maps else 0

    print("\n" + "=" * 60)
    print("YOLOv8s COCO zero-shot — mAP cho 4 class quan tâm")
    print("=" * 60)
    print(f"{'Class':<15}  {'mAP50':>8}  {'mAP50-95':>10}")
    print("-" * 40)
    for name, m in per_class.items():
        print(f"{name:<15}  {m['mAP50']:>8.4f}  {m['mAP50-95']:>10.4f}")
    print("-" * 40)
    print(f"{'MEAN (4cls)':<15}  {mean4_mAP50:>8.4f}  {mean4_mAP:>10.4f}")
    print(f"\n(Toàn bộ 80 class mAP50: {float(rb.map50):.4f})")

    summary = {
        "model": "yolov8s.pt (COCO 80cls, zero-shot)",
        "test_set": "V5 test (1500 frames unseen VN videos)",
        "mean_mAP50_4cls": mean4_mAP50,
        "mean_mAP50_95_4cls": mean4_mAP,
        "overall_80cls_mAP50": float(rb.map50),
        "per_class_4": per_class,
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"\n[OK] Chi tiết: {OUT_DIR / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
