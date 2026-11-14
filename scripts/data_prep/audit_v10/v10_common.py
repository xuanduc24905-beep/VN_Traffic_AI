"""Tiện ích dùng chung cho bộ audit v10 (Roboflow visionvn/vietnam_vehicles v10).

Chỉ ĐỌC dữ liệu — không ghi/sửa gì trong data/.

Cấu trúc v10 (Roboflow export YOLOv8):
    data/raw_vietnam_vehicles_v10/{train,valid,test}/{images,labels}/

Tên file gốc có nhiều variant (Roboflow thêm hậu tố `_640_jpg.rf.<hash>.jpg`):

    1. general_<cam>_ID<vid>_<cls>_model-<m>_color-<c>_<plate>_bien_<màu biển>_DD-MM-YYYY_HH-MM-SS
    2. general_<cam>_<plate>_DD-MM-YYYY_HH-MM-SS                  (không có ID xe)
    3. gen_<cam>_id<vid>_<plate>_<cls>_<Color>_DD-MM-YYYY_HH-MM-SS
    4. veh_<cam>_ID<vid>_<cls>_..._<plate>_bien_..._DD-MM-YYYY_HH-MM-SS
    5. general-<cam>-id<vid>-vuot-den-do-<cls>-...-<plate>-bien-...-DD-MM-YYYY-HH-MM-SS
    + hậu tố tùy chọn `-1-` sau giờ (bản copy khi export trùng tên).

Lưu ý: camera_id có dạng <địa điểm><số kênh>, vd `nga3phuocthe554` và
`nga3phuocthe556` là 2 camera khác nhau ở CÙNG một ngã ba → ta tách thêm
`site_id` (bỏ đuôi số) để kiểm tra leak cấp địa điểm.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
V10_ROOT = REPO_ROOT / "data" / "raw_vietnam_vehicles_v10"
REPORTS_DIR = REPO_ROOT / "reports"
SPLITS = ("train", "valid", "test")
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Bỏ hậu tố Roboflow: "_640_jpg.rf.<hash>" (hash hex 32 ký tự)
_RF_SUFFIX = re.compile(r"_\d+_jpg\.rf\.[0-9a-fA-F]+$")
# Tiền tố + camera: gen_/general_/veh_/general-
_PREFIX_CAM = re.compile(r"^(?P<prefix>general|gen|veh)[_-](?P<cam>[A-Za-z0-9]+?)[_-]")
# Vehicle ID: "ID123" / "id123" đứng riêng giữa dấu phân cách
_VID = re.compile(r"[_-][iI][dD](?P<vid>\d+)(?=[_-])")
# Ngày giờ cuối tên: DD-MM-YYYY_HH-MM-SS (hoặc toàn '-'), hậu tố tùy chọn -1-
_DT = re.compile(
    r"[_-](?P<d>\d{2})-(?P<m>\d{2})-(?P<y>\d{4})[_-](?P<H>\d{2})-(?P<M>\d{2})-(?P<S>\d{2})"
    r"(?P<copy>-\d+-)?$"
)
# Biển số VN: 2 số tỉnh + 1-2 chữ/số series + 4-6 số; đứng trước "_bien" hoặc trước cls/ngày
# Biển số là token NGAY TRƯỚC "_bien_" (có thể là kết quả OCR nhiễu, vd "6A902955")
_PLATE_BEFORE_BIEN = re.compile(r"[_-](?P<plate>[A-Za-z0-9]{5,})[_-]bien[_-]")
_PLATE_ANY = re.compile(r"(?<=[_-])(?P<plate>[0-9]{2}[A-Za-z][A-Za-z0-9]?[0-9]{3,6})(?=[_-])")
_PLATE_GEN = re.compile(r"^gen_[A-Za-z0-9]+_id\d+_(?P<plate>[A-Za-z0-9]+)_")
_PLATE_NOID = re.compile(r"^general_[A-Za-z0-9]+_(?P<plate>[A-Za-z0-9]+)_\d{2}-\d{2}-\d{4}_")
_SITE = re.compile(r"^(?P<site>.*?[A-Za-z])\d+$")


@dataclass
class V10Record:
    split: str
    path: str            # đường dẫn tương đối từ repo root
    stem: str            # tên file không có .jpg
    source_name: str     # tên gốc (đã bỏ hậu tố Roboflow)
    prefix: str | None
    camera_id: str | None
    site_id: str | None
    vehicle_id: str | None
    plate: str | None
    datetime_str: str | None  # ISO
    is_copy_suffix: bool       # có hậu tố "-1-" (bản sao export)

    @property
    def dt(self) -> datetime | None:
        return datetime.fromisoformat(self.datetime_str) if self.datetime_str else None


def source_name_of(stem: str) -> str:
    return _RF_SUFFIX.sub("", stem)


def parse_name(stem: str) -> dict:
    """Parse tên file v10 → dict các trường (None nếu không parse được)."""
    src = source_name_of(stem)
    out = {"source_name": src, "prefix": None, "camera_id": None, "site_id": None,
           "vehicle_id": None, "plate": None, "datetime_str": None, "is_copy_suffix": False}

    m = _PREFIX_CAM.match(src)
    if m:
        out["prefix"] = m["prefix"]
        cam = m["cam"].lower()
        out["camera_id"] = cam
        sm = _SITE.match(cam)
        out["site_id"] = sm["site"] if sm else cam

    m = _VID.search(src)
    if m:
        out["vehicle_id"] = m["vid"]

    m = _DT.search(src)
    if m:
        try:
            dt = datetime(int(m["y"]), int(m["m"]), int(m["d"]),
                          int(m["H"]), int(m["M"]), int(m["S"]))
            out["datetime_str"] = dt.isoformat()
        except ValueError:
            pass
        out["is_copy_suffix"] = bool(m["copy"])

    plate = None
    for rx in (_PLATE_BEFORE_BIEN, _PLATE_GEN, _PLATE_NOID, _PLATE_ANY):
        pm = rx.search(src)
        if pm:
            plate = pm["plate"]
            break
    out["plate"] = plate.upper() if plate else None
    return out


def iter_images(root: Path = V10_ROOT):
    for split in SPLITS:
        d = root / split / "images"
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                yield split, p


def load_records(root: Path = V10_ROOT) -> list[V10Record]:
    recs = []
    for split, p in iter_images(root):
        f = parse_name(p.stem)
        try:
            rel = str(p.relative_to(REPO_ROOT))
        except ValueError:
            rel = str(p)
        recs.append(V10Record(split=split, path=rel, stem=p.stem, **f))
    return recs


def label_path_of(img_path: Path) -> Path:
    return img_path.parent.parent / "labels" / (img_path.stem + ".txt")


def write_json(obj: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    print(f"[OK] Ghi {out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out}")


def survey(recs: list[V10Record]) -> dict:
    """Thống kê tổng quan: số camera, site, vehicle_id, plate, date range, tỉ lệ parse."""
    from collections import Counter
    n = len(recs)
    dts = [r.dt for r in recs if r.dt]
    return {
        "n_images": n,
        "per_split": dict(Counter(r.split for r in recs)),
        "prefix_variants": dict(Counter(r.prefix or "UNPARSED" for r in recs)),
        "parse_rate": {
            "camera_id": sum(r.camera_id is not None for r in recs) / n if n else 0,
            "vehicle_id": sum(r.vehicle_id is not None for r in recs) / n if n else 0,
            "plate": sum(r.plate is not None for r in recs) / n if n else 0,
            "datetime": len(dts) / n if n else 0,
        },
        "n_unique_camera": len({r.camera_id for r in recs if r.camera_id}),
        "n_unique_site": len({r.site_id for r in recs if r.site_id}),
        "n_unique_vehicle_id_global": len({r.vehicle_id for r in recs if r.vehicle_id}),
        "n_unique_camera_vehicle_id": len({(r.camera_id, r.vehicle_id) for r in recs if r.vehicle_id}),
        "n_unique_plate": len({r.plate for r in recs if r.plate}),
        "n_copy_suffix": sum(r.is_copy_suffix for r in recs),
        "date_min": min(dts).isoformat() if dts else None,
        "date_max": max(dts).isoformat() if dts else None,
        "n_unique_dates": len({d.date() for d in dts}),
        "unparsed_examples": [r.source_name for r in recs if not (r.camera_id and r.datetime_str)][:20],
    }


if __name__ == "__main__":
    recs = load_records()
    s = survey(recs)
    print(json.dumps(s, indent=2, ensure_ascii=False))
