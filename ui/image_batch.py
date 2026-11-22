"""Gán nhãn ảnh hàng loạt bằng 1–2 model (dùng trong tab "Ảnh hàng loạt" của streamlit_app).

Tách khỏi streamlit_app.py để test được mà không cần Streamlit.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

V5_NAMES = ["motorcycle", "car", "bus", "truck"]
V5_ID = {n: i for i, n in enumerate(V5_NAMES)}
COLORS = {"motorcycle": (0, 200, 0), "car": (255, 140, 0), "bus": (0, 0, 255),
          "truck": (255, 0, 255), "bicycle": (0, 220, 220)}
COCO_VEHICLE_IDS = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck


def is_coco_weights(path: str | Path) -> bool:
    """yolov8s.pt/yolov8n.pt... gốc COCO 80 lớp (không phải checkpoint fine-tune 4 lớp)."""
    p = Path(path)
    return p.name.startswith("yolo") and p.suffix == ".pt" and "runs" not in p.parts


@dataclass
class ImageResult:
    name: str
    width: int
    height: int
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))  # xyxy pixel
    classes: list[str] = field(default_factory=list)
    confs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_removed_giant: int = 0

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.classes:
            out[c] = out.get(c, 0) + 1
        return out


def decode_image(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def predict_images(model, images: list[tuple[str, np.ndarray]], *, coco: bool, conf: float, iou: float,
                   imgsz: int, half: bool, agnostic_nms: bool, max_box_frac: float | None,
                   batch: int = 8) -> list[ImageResult]:
    """Chạy detect trên danh sách (tên, ảnh BGR). Trả về kết quả đã lọc hộp khổng lồ."""
    results: list[ImageResult] = []
    for k in range(0, len(images), batch):
        chunk = images[k:k + batch]
        preds = model.predict([im for _, im in chunk], conf=conf, iou=iou, imgsz=imgsz, half=half,
                              agnostic_nms=agnostic_nms, verbose=False,
                              classes=COCO_VEHICLE_IDS if coco else None)
        for (name, im), r in zip(chunk, preds):
            h, w = im.shape[:2]
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = [r.names[int(c)] for c in r.boxes.cls.cpu().numpy()]
            cf = r.boxes.conf.cpu().numpy()
            removed = 0
            if max_box_frac is not None and len(xyxy):
                area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) / (w * h)
                keep = area <= max_box_frac
                removed = int((~keep).sum())
                xyxy, cf = xyxy[keep], cf[keep]
                cls = [c for c, k_ in zip(cls, keep) if k_]
            results.append(ImageResult(name, w, h, xyxy, cls, cf, removed))
    return results


def draw(im: np.ndarray, res: ImageResult, show_conf: bool = True) -> np.ndarray:
    """Vẽ hộp lên bản sao ảnh BGR; trả về ảnh RGB để st.image hiển thị."""
    out = im.copy()
    th = max(1, round(max(res.width, res.height) / 800))
    fs = 0.45 * th
    for (x1, y1, x2, y2), c, s in zip(res.boxes, res.classes, res.confs):
        col = COLORS.get(c, (200, 200, 200))
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, col, th + 1)
        label = f"{c} {s:.2f}" if show_conf else c
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        y_txt = max(p1[1], tht + 4)
        cv2.rectangle(out, (p1[0], y_txt - tht - 4), (p1[0] + tw + 2, y_txt), col, -1)
        cv2.putText(out, label, (p1[0] + 1, y_txt - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def to_yolo_txt(res: ImageResult) -> str:
    """Nhãn YOLO theo schema 4 lớp của project (bicycle và lớp lạ bị bỏ)."""
    lines = []
    for (x1, y1, x2, y2), c in zip(res.boxes, res.classes):
        if c not in V5_ID:
            continue
        lines.append(f"{V5_ID[c]} {(x1 + x2) / 2 / res.width:.6f} {(y1 + y2) / 2 / res.height:.6f} "
                     f"{(x2 - x1) / res.width:.6f} {(y2 - y1) / res.height:.6f}")
    return "\n".join(lines)


def build_zip(images: dict[str, np.ndarray], runs: dict[str, list[ImageResult]]) -> bytes:
    """Zip: <model>/images/*.jpg (đã vẽ hộp) + <model>/labels/*.txt (YOLO 4 lớp) + classes.txt."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for tag, results in runs.items():
            z.writestr(f"{tag}/classes.txt", "\n".join(V5_NAMES))
            for res in results:
                stem = Path(res.name).stem
                ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(draw(images[res.name], res), cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    z.writestr(f"{tag}/images/{stem}.jpg", jpg.tobytes())
                z.writestr(f"{tag}/labels/{stem}.txt", to_yolo_txt(res))
    return buf.getvalue()
