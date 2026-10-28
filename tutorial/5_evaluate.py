"""
Bước 5: Evaluate model đã train — chấm điểm mAP trên val + test set.

Flow:
    1. Load best.pt (model đã train xong ở bước 4)
    2. Chạy model.val() trên val set → ra mAP tổng + per-class
    3. Chạy model.val() trên test set → đây mới là con số "thật" đánh giá model
    4. In bảng so sánh val vs test (test cao hơn val đáng ngờ, test thấp hơn val là bình thường)

Bên trong model.val() Ultralytics làm gì?
    1. Load tất cả ảnh val/test
    2. Với mỗi ảnh: forward pass → ra list bbox dự đoán (có conf, class)
    3. So sánh với ground truth (file .txt) → tính TP/FP/FN
    4. Tính AP (Area under PR curve) cho mỗi class, với 10 ngưỡng IoU (0.5 → 0.95)
    5. mAP = mean của AP

LƯU Ý VỀ TEST SET:
    Nếu bạn đã train nhiều config (lr khác, imgsz khác, aug khác) và mỗi lần
    đều eval trên test → thông tin test đã "leak" vào quyết định của bạn.
    → Số mAP test không còn hoàn toàn fair.
    → Trong research chuẩn: chỉ eval test đúng 1 lần, sau khi đã chốt model.

Chạy: python tutorial/5_evaluate.py
"""

from pathlib import Path
from ultralytics import YOLO


# ─── Config ───────────────────────────────────────────────────────────────────

# Đường dẫn model vừa train xong. Nếu train với tên khác thì sửa.
WEIGHTS = "runs/detect/tutorial_train_v8s/weights/best.pt"

# File config dataset (giống bước 4)
DATA_YAML = "tutorial/3_data.yaml"

# imgsz phải khớp lúc train — nếu train imgsz=960 thì eval cũng 960.
IMGSZ = 960

# Batch cho eval — có thể lớn hơn train vì không có gradient (ít tốn VRAM).
BATCH = 32

DEVICE = 0  # GPU id, hoặc "cpu"


# ─── Code ────────────────────────────────────────────────────────────────────

def evaluate_split(model: YOLO, split: str):
    """
    Chạy eval 1 split và in kết quả.

    Return dict để so sánh val vs test ở cuối.
    """
    print(f"\n{'─' * 60}")
    print(f"Eval trên {split.upper()} set")
    print(f"{'─' * 60}")

    # model.val() tự biết tìm split trong data.yaml.
    # Tham số split: "val" hoặc "test". Nếu không truyền → mặc định "val".
    results = model.val(
        data=DATA_YAML,
        split=split,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        plots=True,       # Vẽ confusion matrix, PR curve, F1 curve
        verbose=False,    # Tắt progress bar cho đỡ rối
        save_json=False,  # Set True nếu muốn export JSON cho pycocotools
        project="runs/detect",
        name=f"tutorial_eval_{split}",
        exist_ok=True,
    )

    # results là object ultralytics.utils.metrics.DetMetrics.
    # Có 3 thuộc tính chính:
    #   results.box.map     — mAP@0.5:0.95 overall
    #   results.box.map50   — mAP@0.5 overall
    #   results.box.maps    — array mAP@0.5:0.95 per class
    # Và nhiều cái khác qua .results_dict

    # In bảng per-class
    print(f"\nPer-class metrics:")
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'mAP@0.5':>10} {'mAP@0.5:0.95':>14}")
    print(f"  {'-' * 60}")

    # names là dict {id: name} từ data.yaml
    names = model.names
    # results.box.p / .r / .ap50 / .ap là array dài nc (số class).
    # Index khớp với class_id.
    for i, name in names.items():
        p = results.box.p[i] if i < len(results.box.p) else 0
        r = results.box.r[i] if i < len(results.box.r) else 0
        ap50 = results.box.ap50[i] if i < len(results.box.ap50) else 0
        ap = results.box.ap[i] if i < len(results.box.ap) else 0
        print(f"  {name:<12} {p:>10.3f} {r:>10.3f} {ap50:>10.3f} {ap:>14.3f}")

    # Overall (mean của tất cả class)
    print(f"  {'-' * 60}")
    print(f"  {'ALL':<12} {results.box.mp:>10.3f} {results.box.mr:>10.3f} "
          f"{results.box.map50:>10.3f} {results.box.map:>14.3f}")

    return {
        "map50": results.box.map50,
        "map50_95": results.box.map,
        "precision": results.box.mp,
        "recall": results.box.mr,
    }


def main():
    print("=" * 70)
    print("EVALUATE MODEL")
    print("=" * 70)

    # Bước 1: Kiểm tra file weights có tồn tại
    weights_path = Path(WEIGHTS)
    if not weights_path.exists():
        print(f"❌ Không tìm thấy weights: {WEIGHTS}")
        print("   Chạy bước 4 trước: python tutorial/4_train.py")
        return

    print(f"Weights: {WEIGHTS}")
    print(f"Data:    {DATA_YAML}")

    # Bước 2: Load model
    model = YOLO(WEIGHTS)

    # Bước 3: Eval val và test
    val_metrics = evaluate_split(model, "val")
    test_metrics = evaluate_split(model, "test")

    # Bước 4: So sánh val vs test
    print("\n" + "=" * 70)
    print("SO SÁNH VAL vs TEST")
    print("=" * 70)
    print(f"  {'Metric':<15} {'Val':>10} {'Test':>10} {'Diff':>10}")
    print(f"  {'-' * 50}")
    for key, label in [
        ("map50", "mAP@0.5"),
        ("map50_95", "mAP@0.5:0.95"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]:
        v = val_metrics[key]
        t = test_metrics[key]
        diff = t - v
        print(f"  {label:<15} {v:>10.3f} {t:>10.3f} {diff:>+10.3f}")

    # Phân tích:
    #   Test > Val đáng kể (> +0.05)   → có thể val set khó bất thường, hoặc test leak
    #   Test ≈ Val                     → generalization tốt
    #   Test < Val đáng kể (< -0.05)   → overfit val, hoặc test từ domain khác
    print()
    gap = test_metrics["map50"] - val_metrics["map50"]
    if abs(gap) < 0.03:
        print("  ✓ Val/Test tương đương → generalization ổn trong-domain")
    elif gap < -0.05:
        print("  ⚠ Test thấp hơn val > 5% → model có thể overfit val, hoặc test khó hơn")
    elif gap > 0.05:
        print("  ⚠ Test cao hơn val > 5% → bất thường, kiểm tra data leak hoặc split không đồng đều")

    print("\n→ Chạy tiếp bước 6 để test model trên 1 ảnh/video thật:")
    print("    python tutorial/6_inference.py --video <path/to/video.mp4>")
    print("=" * 70)


if __name__ == "__main__":
    main()
