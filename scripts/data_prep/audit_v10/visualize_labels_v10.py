"""Ve bbox + class label len anh v10 de kiem tra nhan bang mat.

Chay:
  python scripts/data_prep/audit_v10/visualize_labels_v10.py                 # default: train, 200 anh
  python scripts/data_prep/audit_v10/visualize_labels_v10.py --split train --n 200
  python scripts/data_prep/audit_v10/visualize_labels_v10.py --split valid --n 100

Sau khi chay:
  - Mo folder outputs/verify/v10_label_review/<split>/ bang file explorer
  - Luot xem tung anh, ghi file sai vao bad_samples.txt
"""

import argparse
import random
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

V10_ROOT = Path("data/raw_vietnam_vehicles_v10")
OUT_ROOT = Path("outputs/verify/v10_label_review")

# v10 classes: ['container', 'oto', 'xe_ba_gac', 'xe_khach', 'xe_may', 'xe_tai']
CLASS_NAMES = {
    0: "container",
    1: "oto",
    2: "xe_ba_gac",
    3: "xe_khach",
    4: "xe_may",
    5: "xe_tai",
}
CLASS_COLORS = {
    0: (128, 0, 128),    # container: tim
    1: (0, 255, 0),      # oto: xanh la
    2: (255, 255, 0),    # xe_ba_gac: vang
    3: (255, 165, 0),    # xe_khach: cam
    4: (255, 0, 0),      # xe_may: do
    5: (0, 0, 255),      # xe_tai: xanh duong
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

    n_boxes = 0
    if label_path.exists() and label_path.stat().st_size > 0:
        for line in label_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])

            x1 = (xc - w / 2) * W
            y1 = (yc - h / 2) * H
            x2 = (xc + w / 2) * W
            y2 = (yc + h / 2) * H

            color = CLASS_COLORS.get(cls, (255, 255, 255))
            name = CLASS_NAMES.get(cls, f"cls{cls}")

            draw.rectangle([x1, y1, x2, y2], outline=color, width=LINE_WIDTH)
            bbox = draw.textbbox((x1, max(0, y1 - 14)), name, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, max(0, y1 - 14)), name, fill=(255, 255, 255), font=font)
            n_boxes += 1

    img.save(out_path, quality=85)
    return n_boxes


def main(split: str, n_sample: int, seed: int):
    img_dir = V10_ROOT / split / "images"
    lbl_dir = V10_ROOT / split / "labels"
    assert img_dir.exists(), f"Khong thay {img_dir}"
    assert lbl_dir.exists(), f"Khong thay {lbl_dir}"

    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    assert images, f"Khong thay anh trong {img_dir}"

    out_dir = OUT_ROOT / split
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    random.seed(seed)
    n = min(n_sample, len(images))
    sampled = random.sample(images, n)
    print(f"[sample] {n} / {len(images)} anh (split={split}, seed={seed})")

    n_ok = 0
    n_empty = 0
    n_missing = 0
    class_counter = {k: 0 for k in CLASS_NAMES}

    for i, img_path in enumerate(sampled, 1):
        label_path = lbl_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            n_missing += 1
        elif label_path.stat().st_size == 0:
            n_empty += 1

        out_name = f"{i:04d}__{img_path.name}"
        n_boxes = draw_boxes(img_path, label_path, out_dir / out_name)
        if n_boxes == 0 and label_path.exists():
            pass
        n_ok += 1

        if label_path.exists() and label_path.stat().st_size > 0:
            for line in label_path.read_text().strip().split("\n"):
                if line.strip():
                    cls = int(line.split()[0])
                    class_counter[cls] = class_counter.get(cls, 0) + 1

        if i % 50 == 0:
            print(f"  [{i}/{n}] drawn")

    bad_txt = out_dir / "bad_samples.txt"
    bad_txt.write_text(
        "# Ghi ten file anh SAI (box lech / class sai / miss nhieu)\n"
        "# Moi dong 1 file\n"
        "# Vi du:\n"
        "#   0042__xxx.jpg   # xe_may bi danh thanh xe_ba_gac\n"
        "#   0100__yyy.jpg   # miss 2 xe trong frame\n"
    )

    stats_lines = [
        f"# Thong ke sample v10 - split={split}",
        f"",
        f"- Tong anh ve: {n_ok}",
        f"- Seed: {seed}",
        f"- Thieu file .txt: {n_missing}",
        f"- File .txt rong (anh khong co object): {n_empty}",
        f"",
        f"## Phan bo class trong sample",
        f"",
        f"| ID | Class | So bbox |",
        f"|---|---|---|",
    ]
    for cls_id, name in CLASS_NAMES.items():
        stats_lines.append(f"| {cls_id} | {name} | {class_counter[cls_id]} |")
    stats_lines += [
        "",
        "## Mau mau bbox",
        "",
        "- container: tim",
        "- oto: xanh la",
        "- xe_ba_gac: vang",
        "- xe_khach: cam",
        "- xe_may: do",
        "- xe_tai: xanh duong",
        "",
        "## Huong dan review",
        "",
        "1. Mo folder nay bang file explorer (nautilus / VSCode / xdg-open)",
        "2. Luot tung anh, ghi file sai vao bad_samples.txt",
        "3. Danh gia: box lech, class sai, miss object, over-label",
    ]
    (out_dir / "STATS.md").write_text("\n".join(stats_lines))

    print(f"\n[DONE]")
    print(f"  Ve xong: {n_ok} anh")
    print(f"  Thieu .txt: {n_missing}, .txt rong: {n_empty}")
    print(f"  Output: {out_dir}/")
    print(f"\n  Mo folder de review:")
    print(f"    xdg-open {out_dir}/")
    print(f"    code {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "valid", "test"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.split, args.n, args.seed)
