"""Vẽ bbox + class label lên ảnh pseudo-label để spot-check bằng file explorer.

Chạy:
  python scripts/pseudo/06_visualize_review.py              # 500 ảnh random
  python scripts/pseudo/06_visualize_review.py --n 200     # 200 ảnh

Sau khi chạy xong:
  - Mở thư mục data/pseudo_v6/review/ bằng file explorer (nautilus, dolphin, VSCode, ...)
  - Xem từng ảnh — box đúng thì không làm gì, box sai thì ghi tên file vào bad_samples.txt
  - Precision = (ảnh đúng hoàn toàn) / (tổng ảnh xem) -> báo cáo vào tiểu luận
"""

import argparse
import random
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ACCEPTED_DIR = Path("data/pseudo_v6/accepted")
OUT_DIR = Path("data/pseudo_v6/review")

CLASS_NAMES = {0: "motorcycle", 1: "car", 2: "bus", 3: "truck"}
CLASS_COLORS = {
    0: (255, 0, 0),      # motorcycle: đỏ
    1: (0, 255, 0),      # car: xanh lá
    2: (255, 165, 0),    # bus: cam
    3: (0, 0, 255),      # truck: xanh dương
}
LINE_WIDTH = 2


def draw_boxes(img_path: Path, label_path: Path, out_path: Path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    for line in label_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        cls = int(parts[0])
        xc, yc, w, h = map(float, parts[1:5])

        # YOLO -> pixel
        x1 = (xc - w / 2) * W
        y1 = (yc - h / 2) * H
        x2 = (xc + w / 2) * W
        y2 = (yc + h / 2) * H

        color = CLASS_COLORS.get(cls, (255, 255, 255))
        name = CLASS_NAMES.get(cls, f"cls{cls}")

        draw.rectangle([x1, y1, x2, y2], outline=color, width=LINE_WIDTH)
        # Label background
        label_text = name
        bbox = draw.textbbox((x1, y1 - 14), label_text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 14), label_text, fill=(255, 255, 255), font=font)

    img.save(out_path, quality=85)


def main(n_sample: int, seed: int):
    images = sorted((ACCEPTED_DIR / "images").glob("*.jpg"))
    assert images, f"Không thấy ảnh trong {ACCEPTED_DIR / 'images'}"

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    # Random sample
    random.seed(seed)
    n = min(n_sample, len(images))
    sampled = random.sample(images, n)
    print(f"[sample] {n} / {len(images)} ảnh (seed={seed})")

    n_ok = 0
    n_skip = 0
    for i, img_path in enumerate(sampled, 1):
        label_path = ACCEPTED_DIR / "labels" / (img_path.stem + ".txt")
        if not label_path.exists():
            n_skip += 1
            continue

        # Đặt tên file theo index để dễ duyệt theo thứ tự
        out_name = f"{i:04d}__{img_path.name}"
        draw_boxes(img_path, label_path, OUT_DIR / out_name)
        n_ok += 1

        if i % 50 == 0:
            print(f"  [{i}/{n}] drawn")

    # Tạo file template để user ghi ảnh sai
    bad_txt = OUT_DIR / "bad_samples.txt"
    bad_txt.write_text(
        "# Ghi tên file ảnh SAI (box lệch / class sai / miss nhiều)\n"
        "# Mỗi dòng 1 file, bỏ comment ở đầu nếu tag đúng\n"
        "# Ví dụ:\n"
        "#   0042__hcm_xxx.jpg   # motorbike bị đánh thành car\n"
        "#   0100__hcm_yyy.jpg   # miss 2 xe trong frame\n"
    )

    # Tạo file template kết quả
    result_txt = OUT_DIR / "RESULT_FORM.md"
    result_txt.write_text(
        f"""# Kết quả spot-check pseudo-label V6 round 1

**Tổng ảnh review**: {n_ok}
**Seed sample**: {seed}

## Đếm thủ công

Sau khi xem hết {n_ok} ảnh, điền vào bảng dưới:

| Phân loại | Số ảnh |
|---|---|
| CORRECT (tất cả box đúng) |   |
| PARTIAL (1-2 box sai, còn lại đúng) |   |
| WRONG (đa số box sai) |   |

## Tính toán precision

- **Precision strict** = CORRECT / {n_ok}  = ___ %
- **Precision loose**  = (CORRECT + PARTIAL) / {n_ok}  = ___ %

## Quyết định

- Nếu precision strict >= 85% → OK, dùng toàn bộ {len(images)} pseudo-label cho train V6
- Nếu 70-85% → tăng threshold conf lên, chạy lại bước infer + filter
- Nếu < 70% → pseudo-label noisy quá, xem lại từ đầu

## Ảnh SAI (nếu cần review lại)
Xem file `bad_samples.txt`.
"""
    )

    print(f"\n[DONE]")
    print(f"  Vẽ xong: {n_ok} ảnh ({n_skip} skip vì thiếu label)")
    print(f"  Output : {OUT_DIR}/")
    print(f"\n  Mở folder bằng file explorer để review:")
    print(f"    nautilus {OUT_DIR}/ &     # Ubuntu/Gnome")
    print(f"    code {OUT_DIR}/           # VSCode")
    print(f"    xdg-open {OUT_DIR}/       # Linux generic")
    print(f"\n  Điền kết quả vào: {OUT_DIR}/RESULT_FORM.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="số ảnh review")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.n, args.seed)
