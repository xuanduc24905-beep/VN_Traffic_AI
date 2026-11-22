"""Tạo tập đánh giá 2.135 ảnh bằng symlink, không sao chép hoặc sửa dữ liệu gốc.

Thành phần:
- V5+ test: 1.500 ảnh
- VN Aerial val: 486 ảnh
- VN Aerial test: 149 ảnh

Tên file được thêm prefix theo nguồn để tránh collision. Nhãn giữ nguyên schema
0=motocycle, 1=car, 2=bus, 3=truck đã được kiểm tra ở các dataset nguồn.
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "eval_extended_2135"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCES = [
    ("v5", ROOT / "data" / "merged_v5plus" / "images" / "test"),
    ("aerial_val", ROOT / "data" / "vn_vehicles_aerial" / "images" / "val"),
    ("aerial_test", ROOT / "data" / "vn_vehicles_aerial" / "images" / "test"),
]
EXPECTED = {"v5": 1500, "aerial_val": 486, "aerial_test": 149}


def label_for(image: Path) -> Path:
    parts = list(image.parts)
    parts[-3] = "labels"
    return Path(*parts).with_suffix(".txt")


def link(source: Path, destination: Path) -> None:
    if destination.is_symlink() or destination.exists():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        raise RuntimeError(f"Đích đã tồn tại nhưng không khớp: {destination}")
    destination.symlink_to(source.resolve())


def main() -> None:
    image_out = OUT / "images" / "test"
    label_out = OUT / "labels" / "test"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    counts = {}
    for prefix, folder in SOURCES:
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
        counts[prefix] = len(images)
        for image in images:
            label = label_for(image)
            if not label.exists():
                raise FileNotFoundError(f"Thiếu nhãn cho {image}: {label}")
            name = f"{prefix}__{image.name}"
            link(image, image_out / name)
            link(label, (label_out / name).with_suffix(".txt"))

    if counts != EXPECTED:
        raise RuntimeError(f"Sai số lượng: {counts}, kỳ vọng {EXPECTED}")
    total = sum(counts.values())
    if len(list(image_out.iterdir())) != total or len(list(label_out.glob("*.txt"))) != total:
        raise RuntimeError("Số symlink ảnh/nhãn không khớp")
    print(f"[OK] {counts}; tổng={total}; output={OUT}")


if __name__ == "__main__":
    main()
