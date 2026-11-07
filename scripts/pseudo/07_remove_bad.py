"""Xoá ảnh/label pseudo-label được đánh dấu SAI sau khi review tay.

Input :
  - data/pseudo_v6/review/bad_samples.txt (tên file sai cụ thể)
  - --exclude-cameras 000605,000617 (xoá toàn bộ ảnh từ camera id xấu)

Action: xoá cặp .jpg + .txt tương ứng khỏi data/pseudo_v6/accepted/

Chạy:
  python scripts/pseudo/07_remove_bad.py                             # dry-run
  python scripts/pseudo/07_remove_bad.py --apply                     # xoá bad_samples
  python scripts/pseudo/07_remove_bad.py --exclude-cameras 000605    # + xoá camera 000605
  python scripts/pseudo/07_remove_bad.py --apply --exclude-cameras 000605,000617
"""

import argparse
import re
from pathlib import Path

BAD_LIST = Path("data/pseudo_v6/review/bad_samples.txt")
ACCEPTED_DIR = Path("data/pseudo_v6/accepted")


def parse_bad_list() -> list[str]:
    """Trả list filenames (không prefix số thứ tự review)."""
    if not BAD_LIST.exists():
        print(f"[err] Không thấy {BAD_LIST}")
        print(f"      Chạy bước review trước (scripts/pseudo/06_visualize_review.py)")
        return []

    names = []
    for line in BAD_LIST.read_text().split("\n"):
        line = line.strip()
        # Bỏ dòng trống + comment
        if not line or line.startswith("#"):
            continue
        # Lấy phần filename trước comment inline (nếu có)
        filename = line.split("#")[0].strip()
        if not filename:
            continue
        # Bỏ prefix số thứ tự review "0042__" nếu có
        m = re.match(r"^\d{4}__(.+)$", filename)
        if m:
            filename = m.group(1)
        names.append(filename)
    return names


def find_files_by_camera(cam_ids: list[str]) -> list[Path]:
    """Tìm tất cả ảnh có camera_id trong tên file."""
    matches = []
    for img in (ACCEPTED_DIR / "images").glob("*.jpg"):
        # Format: hcm_<ward>_<camera_id>_<timestamp>.jpg
        for cam_id in cam_ids:
            if f"_{cam_id}_" in img.name:
                matches.append(img)
                break
    return matches


def main(apply: bool, exclude_cameras: list[str]):
    # ─── 1. Parse bad_samples.txt ───
    bad_names = parse_bad_list()
    n_bad = len(bad_names)
    print(f"[load] {n_bad} file từ bad_samples.txt")

    # ─── 2. Tìm ảnh theo exclude_cameras ───
    cam_victims = []
    if exclude_cameras:
        cam_victims = find_files_by_camera(exclude_cameras)
        print(f"[load] {len(cam_victims)} ảnh thuộc camera xấu {exclude_cameras}")

    # ─── 3. Gom lại các stem cần xoá (unique) ───
    stems_to_remove = set()

    for name in bad_names:
        stem = name.rsplit(".", 1)[0] if "." in name else name
        img = ACCEPTED_DIR / "images" / f"{stem}.jpg"
        lbl = ACCEPTED_DIR / "labels" / f"{stem}.txt"
        if img.exists() or lbl.exists():
            stems_to_remove.add(stem)

    for img in cam_victims:
        stems_to_remove.add(img.stem)

    print(f"\n[plan] Tổng số ảnh sẽ xoá (unique): {len(stems_to_remove)}")

    # ─── 4. Apply ───
    n_removed_img = 0
    n_removed_lbl = 0
    if apply:
        for stem in stems_to_remove:
            img = ACCEPTED_DIR / "images" / f"{stem}.jpg"
            lbl = ACCEPTED_DIR / "labels" / f"{stem}.txt"
            if img.exists():
                img.unlink()
                n_removed_img += 1
            if lbl.exists():
                lbl.unlink()
                n_removed_lbl += 1

    # ─── 5. Report ───
    mode = "[APPLIED]" if apply else "[DRY-RUN]"
    print(f"\n{'='*55}")
    print(f"{mode} Clean pseudo-label V6")
    print(f"{'='*55}")
    print(f"  Từ bad_samples.txt         : {n_bad} files")
    print(f"  Từ exclude-cameras         : {len(cam_victims)} files")
    print(f"  Tổng (unique)              : {len(stems_to_remove)}")

    if apply:
        remain_img = len(list((ACCEPTED_DIR / "images").glob("*.jpg")))
        remain_lbl = len(list((ACCEPTED_DIR / "labels").glob("*.txt")))
        print(f"\n  Đã xoá                     : {n_removed_img} ảnh + {n_removed_lbl} label")
        print(f"  Còn lại trong accepted/    : {remain_img} ảnh + {remain_lbl} label")
    else:
        print("\n  (dry-run — chạy với --apply để xoá thật)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--exclude-cameras", default="",
                    help="comma-separated camera ids to remove entirely (e.g. 000605,000617)")
    args = ap.parse_args()
    cams = [c.strip() for c in args.exclude_cameras.split(",") if c.strip()]
    main(apply=args.apply, exclude_cameras=cams)
