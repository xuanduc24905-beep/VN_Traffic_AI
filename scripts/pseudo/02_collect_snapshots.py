"""Tải snapshot từ danh sách camera HCM -> data/raw/frames/hcm/

Đọc configs từ data/raw/hcm_cameras.yaml (do 01_scrape_camera_urls.py tạo).
Mỗi vòng: duyệt tất cả camera active, tải 1 ảnh, crop overlay 50px trên.

Chạy:
  python scripts/pseudo/02_collect_snapshots.py --duration 60
  python scripts/pseudo/02_collect_snapshots.py --duration 90 --max-cameras 30
"""

import argparse
import time
import yaml
import requests
import random
from io import BytesIO
from pathlib import Path
from datetime import datetime
from PIL import Image

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://giaothong.hochiminhcity.gov.vn/",
}
CAMERAS_YAML = Path("data/raw/hcm_cameras.yaml")
CAMERAS_YAML_COMMITTED = Path("configs/hcm_cameras.yaml")
OUT_DIR = Path("data/raw/frames/hcm")
CROP_TOP = 50
INTERVAL = 20   # giây giữa 2 vòng — nguồn Sở GTVT HCM refresh ~15s


def load_cameras(max_cameras: int | None) -> list[dict]:
    path = CAMERAS_YAML if CAMERAS_YAML.exists() else CAMERAS_YAML_COMMITTED
    with open(path) as f:
        data = yaml.safe_load(f)
    cams = [c for c in data["cameras"].values() if c.get("image_url")]
    print(f"[load] total cameras with URL: {len(cams)}")
    if max_cameras and max_cameras < len(cams):
        random.seed(42)
        cams = random.sample(cams, max_cameras)
        print(f"[load] sampled {len(cams)} cameras")
    return cams


def grab(cam: dict) -> bool:
    try:
        r = requests.get(cam["image_url"], headers=HEADERS, timeout=12)
        if r.status_code != 200:
            return False
        img = Image.open(BytesIO(r.content))
        if img.height > CROP_TOP + 100:
            img = img.crop((0, CROP_TOP, img.width, img.height))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"hcm_{cam['ward']}_{cam['id']}_{ts}.jpg"
        out = OUT_DIR / cam["ward"]
        out.mkdir(parents=True, exist_ok=True)
        img.convert("RGB").save(out / name, quality=90)
        return True
    except Exception as e:
        print(f"  err {cam['id']}: {e}")
        return False


def main(duration_min: int, max_cameras: int | None):
    cams = load_cameras(max_cameras)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    end = time.time() + duration_min * 60
    round_i = 0
    total_ok = 0
    while time.time() < end:
        round_i += 1
        t0 = time.time()
        ok = 0
        for cam in cams:
            if grab(cam):
                ok += 1
            time.sleep(0.3)   # nhẹ nhàng với server
        total_ok += ok
        elapsed = time.time() - t0
        print(f"[round {round_i}] {datetime.now():%H:%M:%S} "
              f"ok={ok}/{len(cams)} elapsed={elapsed:.1f}s total={total_ok}")

        wait = max(0, INTERVAL - elapsed)
        if time.time() + wait >= end:
            break
        time.sleep(wait)

    print(f"\n[DONE] {round_i} rounds, {total_ok} snapshots saved to {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=60, help="phút chạy")
    ap.add_argument("--max-cameras", type=int, default=30,
                    help="chọn ngẫu nhiên N camera (None = tất cả)")
    args = ap.parse_args()
    main(args.duration, args.max_cameras)
