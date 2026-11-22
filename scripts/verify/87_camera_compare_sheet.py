#!/usr/bin/env python3
"""Ve ban doi chieu camera SOICT (ngay va dem) de kiem tra bang mat co cung khung canh hay khong.
Moi hang la mot camera, 3 khung hinh cach deu nhau; cac hang xep theo cap nghi trung khung canh."""
import glob
from pathlib import Path

import cv2
import numpy as np

R = Path("/home/xuand/vn_traffic_ai/data/merged_v5plus/images")
W, H = 400, 225
ORDER = [("soictnight_cam_03", "DEM cam_03 (train)"), ("soict24_cam_01", "NGAY cam_01 (train)"), ("soict24_cam_02", "NGAY cam_02 (train)"),
         ("soictnight_cam_05", "DEM cam_05 (train)"), ("soict24_cam_04", "NGAY cam_04 (train)"),
         ("soictnight_cam_08", "DEM cam_08 (train)"), ("soict24_cam_07", "NGAY cam_07 (val)"),
         ("soict24_cam_06", "NGAY cam_06 (train)"),
         ("soict24_cam_09", "NGAY cam_09 (TEST)"), ("soict24_cam_10", "NGAY cam_10 (TEST)")]


def frames(prefix):
    fs = sorted(glob.glob(str(R / "*" / f"{prefix}_*.jpg")))
    assert fs, prefix
    return [fs[int(i * (len(fs) - 1))] for i in (0.1, 0.5, 0.9)], len(fs)


rows = []
for pre, label in ORDER:
    fs, n = frames(pre)
    tiles = []
    for f in fs:
        im = cv2.resize(cv2.imread(f), (W, H))
        tiles.append(im)
    strip = np.hstack(tiles)
    bar = np.full((34, strip.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, f"{label} - {n} anh", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    rows.append(np.vstack([bar, strip]))
sep = np.full((10, rows[0].shape[1], 3), 255, np.uint8)
# nhom theo cap: [03,01,02] [05,04] [08,07] [06] [09,10]
groups = [(0, 3), (3, 5), (5, 7), (7, 8), (8, 10)]
for gi, (a, b) in enumerate(groups, 1):
    img = rows[a]
    for r in rows[a + 1:b]:
        img = np.vstack([img, sep, r])
    out = Path(f"/home/xuand/vn_traffic_ai/outputs/verify/cam_compare_{gi}.jpg")
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(out, img.shape)
