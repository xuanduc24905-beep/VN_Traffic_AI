"""
Bước 2: Verify dataset trước khi train.

Script này KHÔNG tạo data mới — nó CHECK data sẵn có ở `data/merged_v5/` có đúng
format YOLOv8 yêu cầu không, và in thống kê để bạn biết dataset gồm gì.

YOLOv8 cần layout:
    data/merged_v5/
    ├── images/
    │   ├── train/  (ảnh .jpg)
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/  (file .txt cùng tên ảnh)
    │   ├── val/
    │   └── test/
    └── (data.yaml ở ngoài — bước 3)

Mỗi file .txt chứa bbox theo format:
    <class_id> <center_x> <center_y> <width> <height>
    - class_id: số nguyên (0, 1, 2, 3) tương ứng với names trong data.yaml
    - 4 số còn lại: normalize [0, 1] theo kích thước ảnh

Chạy: python tutorial/2_prepare_data.py
"""

from pathlib import Path     # module xử lý đường dẫn file (hiện đại hơn os.path)
from collections import Counter  # đếm tần suất — tiện cho đếm class

# ─── Config: đổi đường dẫn nếu dataset ở chỗ khác ─────────────────────────────
DATA_ROOT = Path("/home/xuand/vn_traffic_ai/data/merged_v5")
CLASS_NAMES = ["motorcycle", "car", "bus", "truck"]  # phải khớp với data.yaml


def check_layout(root: Path):
    """
    Verify thư mục đúng format YOLO. Thiếu folder nào sẽ in cảnh báo + dừng.
    """
    print(f"[Check layout] Root: {root}")

    if not root.exists():
        print(f"  ❌ Thư mục không tồn tại: {root}")
        return False

    # YOLO yêu cầu images/{split} và labels/{split}
    required = []
    for split in ["train", "val", "test"]:
        required.append(root / "images" / split)
        required.append(root / "labels" / split)

    all_ok = True
    for p in required:
        if p.exists():
            print(f"  ✓ {p.relative_to(root)}")
        else:
            print(f"  ❌ Thiếu: {p.relative_to(root)}")
            all_ok = False

    return all_ok


def count_split(root: Path, split: str):
    """
    Đếm số ảnh + số bbox mỗi class trong 1 split (train/val/test).

    Return dict: {
        'n_images': int,
        'n_images_with_label': int,  # ảnh có file label tương ứng
        'n_images_empty_label': int, # ảnh có label rỗng (ảnh không có xe)
        'n_images_no_label': int,    # ảnh không có file .txt
        'class_count': {0: ..., 1: ...},
    }
    """
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split

    # Liệt kê tất cả ảnh .jpg (có thể thêm .png nếu dataset có mix)
    # pathlib.Path.glob trả generator → list() để đếm được
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

    class_count = Counter()  # dict mặc định = 0, dễ ++
    n_with_label = 0
    n_empty = 0
    n_no_label = 0

    for img_path in images:
        # Mỗi ảnh "abc.jpg" phải có file "abc.txt" cùng tên trong labels/
        lbl_path = lbl_dir / (img_path.stem + ".txt")  # .stem = tên không có đuôi

        if not lbl_path.exists():
            n_no_label += 1
            continue

        # Đọc file label
        with open(lbl_path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]  # bỏ dòng trống

        if not lines:
            # File tồn tại nhưng rỗng → ảnh không có bbox (negative sample)
            # YOLO vẫn chấp nhận và dùng để học "không có gì"
            n_empty += 1
            continue

        n_with_label += 1
        # Mỗi dòng = 1 bbox. Lấy số đầu = class_id.
        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                # Format sai — bỏ qua, nhưng tốt nhất là cảnh báo
                continue
            cls_id = int(parts[0])
            class_count[cls_id] += 1

    return {
        "n_images": len(images),
        "n_images_with_label": n_with_label,
        "n_images_empty_label": n_empty,
        "n_images_no_label": n_no_label,
        "class_count": dict(class_count),
    }


def print_split_stats(split: str, stats: dict):
    """In thống kê 1 split dưới dạng dễ đọc."""
    print(f"\n[{split.upper()}]")
    print(f"  Tổng ảnh:              {stats['n_images']:>6,}")
    print(f"  Có bbox:               {stats['n_images_with_label']:>6,}")
    print(f"  Có label rỗng (negative): {stats['n_images_empty_label']:>6,}")

    if stats['n_images_no_label'] > 0:
        # Nếu > 0 là bất thường — nên fix trước khi train
        print(f"  ⚠ Thiếu file label:    {stats['n_images_no_label']:>6,} (sẽ bị YOLO bỏ qua)")

    print(f"\n  Số bbox mỗi class:")
    total_bbox = sum(stats['class_count'].values())
    for cls_id, name in enumerate(CLASS_NAMES):
        n = stats['class_count'].get(cls_id, 0)
        pct = 100 * n / total_bbox if total_bbox else 0
        print(f"    {cls_id} {name:12s}: {n:>6,} ({pct:5.1f}%)")
    print(f"  Tổng bbox:             {total_bbox:>6,}")


def check_class_imbalance(stats_train: dict):
    """
    Cảnh báo nếu class quá mất cân đối. Ví dụ motorcycle chỉ chiếm 0.8%
    sẽ khiến model dự đoán kém class này.
    """
    counts = stats_train['class_count']
    if not counts:
        return

    total = sum(counts.values())
    print("\n[Phân tích cân đối class — train set]")

    min_pct = 100.0
    for cls_id, name in enumerate(CLASS_NAMES):
        n = counts.get(cls_id, 0)
        pct = 100 * n / total if total else 0
        if pct < min_pct:
            min_pct = pct

    # Ngưỡng kinh nghiệm: class nào < 5% là "hiếm" → mAP thấp
    if min_pct < 5:
        print(f"  ⚠ Có class dưới 5% ({min_pct:.1f}%) — mAP class đó sẽ thấp.")
        print(f"    Fix: thêm data class hiếm, hoặc class-weighted loss")
    else:
        print(f"  ✓ Phân bố tương đối ổn (class ít nhất: {min_pct:.1f}%)")


def main():
    print("=" * 60)
    print("VERIFY DATASET")
    print("=" * 60)

    # Bước 1: check layout thư mục
    if not check_layout(DATA_ROOT):
        print("\n❌ Layout dataset sai — không thể train. Xem docs/DATA_GUIDE.md")
        return

    # Bước 2: đếm thống kê 3 split
    stats = {}
    for split in ["train", "val", "test"]:
        stats[split] = count_split(DATA_ROOT, split)
        print_split_stats(split, stats[split])

    # Bước 3: cảnh báo nếu class mất cân đối
    check_class_imbalance(stats['train'])

    # Bước 4: cảnh báo nếu val/test quá nhỏ (< 100 ảnh thì mAP không ổn định)
    print("\n[Kiểm tra kích thước val/test]")
    for split in ["val", "test"]:
        n = stats[split]['n_images_with_label']
        if n < 100:
            print(f"  ⚠ {split} chỉ có {n} ảnh — mAP có thể không ổn định (noisy)")
        else:
            print(f"  ✓ {split} có {n} ảnh — đủ để đo mAP")

    print("\n" + "=" * 60)
    print("✓ DATASET OK — chạy tiếp bước 3 (xem file config):")
    print("    cat tutorial/3_data.yaml")
    print("=" * 60)


if __name__ == "__main__":
    main()
