"""Scrape danh sách camera HCM từ cameragiaothong.net -> hcm_cameras.yaml

Nguồn thật: giaothong.hochiminhcity.gov.vn (Sở GTVT HCM chính thức)
cameragiaothong.net chỉ là wrapper dễ scrape hơn.

ID format: hochiminh-city-000001 .. hochiminh-city-000800 (approximate)
Mỗi trang detail chứa URL dạng:
  giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id=<mongo_object_id>

Chạy:  python scripts/pseudo/01_scrape_camera_urls.py
Resume: tự động nếu yaml đã có camera
"""

import re
import time
import yaml
import requests
import html
from pathlib import Path

BASE = "https://www.cameragiaothong.net"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.cameragiaothong.net/",
}
OUT_YAML = Path("data/raw/hcm_cameras.yaml")
SLEEP = 1.5
MAX_ID = 800   # ID upper bound (750 valid confirmed, 800 safe)

IMG_RE = re.compile(
    r"giaothong\.hochiminhcity\.gov\.vn/render/ImageHandler\.ashx\?id=([a-f0-9]+)"
)
NAME_RE = re.compile(
    r'<title>([^<|]+)', re.IGNORECASE
)
REGION_RE = re.compile(
    r'/region/thanh-pho-ho-chi-minh/([a-z0-9-]+)'
)


def get(url: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 404:
                return "404"
            if r.status_code == 200:
                return r.text
            print(f"  [{r.status_code}] {url}")
        except requests.RequestException as e:
            print(f"  [err] {url}: {e}")
        time.sleep(2 ** attempt)
    return None


def extract_info(html_text: str, camera_id: str) -> dict:
    img_match = IMG_RE.search(html_text)
    name_match = NAME_RE.search(html_text)
    region_match = REGION_RE.search(html_text)

    image_url = None
    handler_id = None
    if img_match:
        handler_id = img_match.group(1)
        image_url = f"https://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id={handler_id}"

    name = None
    if name_match:
        name = html.unescape(name_match.group(1)).strip()
        # Clean up: "Camera giao thông X - Y, Phường Z - Trực tiếp" -> keep as-is

    ward = region_match.group(1) if region_match else "unknown"

    return {
        "id": camera_id,
        "handler_id": handler_id,
        "name": name,
        "ward": ward,
        "detail_url": f"{BASE}/cctv?id=hochiminh-city-{camera_id}",
        "image_url": image_url,
        "error": None if image_url else "no image url found",
    }


def load_progress() -> dict:
    if OUT_YAML.exists():
        with open(OUT_YAML) as f:
            data = yaml.safe_load(f) or {}
        print(f"[resume] đã có {len(data.get('cameras', {}))} camera trong yaml")
        return data
    return {
        "meta": {
            "source": "cameragiaothong.net",
            "upstream": "giaothong.hochiminhcity.gov.vn (Sở GTVT HCM)",
            "city": "ho-chi-minh-city",
        },
        "cameras": {},
    }


def save_progress(data: dict):
    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_YAML, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def main():
    data = load_progress()
    cams = data["cameras"]
    done_ids = set(cams.keys())

    pending_ids = [f"{i:06d}" for i in range(1, MAX_ID + 1) if f"{i:06d}" not in done_ids]
    print(f"[plan] MAX_ID={MAX_ID} done={len(done_ids)} pending={len(pending_ids)}")

    n_404 = 0
    for i, cam_id in enumerate(pending_ids, 1):
        url = f"{BASE}/cctv?id=hochiminh-city-{cam_id}"
        text = get(url)
        if text == "404":
            n_404 += 1
            if i % 20 == 0:
                print(f"[{i}/{len(pending_ids)}] id={cam_id} 404 (total 404: {n_404})")
            time.sleep(SLEEP)
            continue
        if text is None:
            cams[cam_id] = {"id": cam_id, "image_url": None, "error": "fetch failed"}
        else:
            cams[cam_id] = extract_info(text, cam_id)
            if i % 20 == 0:
                info = cams[cam_id]
                name_preview = (info.get("name") or "")[:50]
                print(f"[{i}/{len(pending_ids)}] id={cam_id} ward={info['ward']} "
                      f"handler={info['handler_id']} name={name_preview!r}")

        if i % 25 == 0:
            save_progress(data)
            ok = sum(1 for c in cams.values() if c.get("image_url"))
            print(f"  [checkpoint] saved — total {len(cams)} ({ok} with URL)")

        time.sleep(SLEEP)

    save_progress(data)
    ok = sum(1 for c in cams.values() if c.get("image_url"))
    wards = len({c["ward"] for c in cams.values() if c.get("image_url") and c.get("ward")})
    print(f"\n[DONE] total={len(cams)} with_url={ok} 404s={n_404} wards={wards}")


if __name__ == "__main__":
    main()
