"""Scrape danh sách camera HCM từ opencctv.org -> hcm_cameras.yaml

Pipeline:
  1. Duyệt trang index 1..35 -> thu thập ~557 URL trang chi tiết
  2. Với mỗi trang chi tiết -> extract imgproxy.windy.com/.../original.jpg
  3. Ghi tiến độ vào data/raw/hcm_cameras.yaml (resume được nếu crash)

Chạy:  python scripts/pseudo/01_scrape_camera_urls.py
"""

import re
import time
import yaml
import requests
from pathlib import Path

BASE = "https://opencctv.org"
INDEX = f"{BASE}/cameras/vietnam/ho-chi-minh-city"
TOTAL_PAGES = 35
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}
OUT_YAML = Path("data/raw/hcm_cameras.yaml")
SLEEP = 2.0

LINK_RE = re.compile(
    r"/cameras/vietnam/ho-chi-minh-city/([a-z0-9-]+)/([a-z0-9-]+-(\d+))"
)
IMG_RE = re.compile(
    r"https?://imgproxy\.windy\.com/[^\"'\s<>]+/original\.jpg"
)


def get(url: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r.text
            print(f"  [{r.status_code}] {url}")
        except requests.RequestException as e:
            print(f"  [err] {url}: {e}")
        time.sleep(2 ** attempt)
    return None


def collect_detail_urls() -> list[dict]:
    """Duyệt page 1..35, trả list [{ward, slug, id, detail_url}]"""
    seen = {}
    for page in range(1, TOTAL_PAGES + 1):
        url = INDEX if page == 1 else f"{INDEX}?page={page}"
        print(f"[index] page {page}/{TOTAL_PAGES}")
        html = get(url)
        if not html:
            continue
        for m in LINK_RE.finditer(html):
            ward, slug, cam_id = m.group(1), m.group(2), m.group(3)
            if cam_id in seen:
                continue
            seen[cam_id] = {
                "id": cam_id,
                "ward": ward,
                "slug": slug,
                "detail_url": f"{BASE}/cameras/vietnam/ho-chi-minh-city/{ward}/{slug}",
            }
        print(f"  cumulative unique: {len(seen)}")
        time.sleep(SLEEP)
    return list(seen.values())


def load_progress() -> dict:
    if OUT_YAML.exists():
        with open(OUT_YAML) as f:
            data = yaml.safe_load(f) or {}
        print(f"[resume] đã có {len(data.get('cameras', {}))} camera trong yaml")
        return data
    return {"meta": {"source": "opencctv.org", "city": "ho-chi-minh-city"},
            "cameras": {}}


def save_progress(data: dict):
    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_YAML, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def extract_image_urls(camera_list: list[dict]):
    data = load_progress()
    cams = data["cameras"]
    done_ids = set(cams.keys())

    pending = [c for c in camera_list if c["id"] not in done_ids]
    print(f"[detail] total={len(camera_list)} done={len(done_ids)} pending={len(pending)}")

    for i, cam in enumerate(pending, 1):
        print(f"[detail {i}/{len(pending)}] {cam['ward']}/{cam['slug']}")
        html = get(cam["detail_url"])
        if not html:
            cams[cam["id"]] = {**cam, "image_url": None, "error": "fetch failed"}
        else:
            m = IMG_RE.search(html)
            cams[cam["id"]] = {
                **cam,
                "image_url": m.group(0) if m else None,
                "error": None if m else "no imgproxy found",
            }

        if i % 10 == 0:
            save_progress(data)
            print(f"  [checkpoint] saved {len(cams)} cameras")
        time.sleep(SLEEP)

    save_progress(data)
    ok = sum(1 for c in cams.values() if c.get("image_url"))
    print(f"\n[DONE] total={len(cams)} with_url={ok} missing={len(cams) - ok}")


if __name__ == "__main__":
    print("=== Step 1: collect detail URLs from index ===")
    detail_list = collect_detail_urls()
    print(f"\nFound {len(detail_list)} unique cameras")

    print("\n=== Step 2: extract image URLs from each detail page ===")
    extract_image_urls(detail_list)
