"""Hàm dùng chung cho 4 pipeline cải tiến V5+ (xem docs/pham-hung/HUONG_DAN.md).

Không chứa logic huấn luyện. Chỉ có: đọc/ghi nhãn YOLO, mã camera HCM, chép/liên kết
file, ghi file dataset yaml, và dự đoán bằng một hoặc nhiều mô hình giáo viên (ghép
bằng WBF khi có từ hai mô hình trở lên).
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "scripts" / "eval", REPO / "scripts" / "training"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

NAMES = {0: "motorcycle", 1: "car", 2: "bus", 3: "truck"}
NAME_TO_ID = {v: k for k, v in NAMES.items()}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# tên file ảnh pseudo: hcm_<phường>_<mã camera 6 số>_<yyyymmdd>_<hhmmss>.jpg
CAM_RE = re.compile(r"^hcm_.+?_(\d{6})_\d{8}_\d{6}")


def camera_id(filename: str) -> str | None:
    m = CAM_RE.match(Path(filename).name)
    return m.group(1) if m else None


def load_camera_set(arg: str | None) -> set[str]:
    """arg: danh sách mã cách nhau dấu phẩy, hoặc đường dẫn file.
    File .yaml/.yml: lấy các khoá dưới 'blacklist' (định dạng configs/hcm_cameras_blacklist.yaml).
    File khác: mỗi dòng một mã, bỏ dòng trống và dòng bắt đầu bằng #."""
    if not arg:
        return set()
    p = Path(arg)
    if p.exists() and p.is_file():
        if p.suffix.lower() in (".yaml", ".yml"):
            import yaml
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            return {str(k).zfill(6) for k in (d.get("blacklist") or {})}
        out = set()
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.split("#")[0].strip()
            if ln:
                out.add(ln.zfill(6))
        return out
    return {x.strip().zfill(6) for x in arg.split(",") if x.strip()}


def list_images(d: Path) -> list[Path]:
    return sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXTS)


def label_path_for(img: Path) -> Path:
    """Đường dẫn nhãn tương ứng: thay phần tử thư mục 'images' cuối cùng bằng 'labels'."""
    parts = list(img.parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    else:
        raise ValueError(f"đường dẫn ảnh không nằm trong thư mục 'images': {img}")
    return Path(*parts).with_suffix(".txt")


def read_yolo(path: Path) -> np.ndarray:
    """Trả về mảng [n, 5]: lớp, xc, yc, w, h (chuẩn hoá). Dòng có độ tin cậy (cột 6) bị bỏ cột đó."""
    rows = []
    if Path(path).exists():
        for ln in Path(path).read_text().splitlines():
            t = ln.split()
            if len(t) >= 5:
                rows.append([float(x) for x in t[:5]])
    return np.array(rows, dtype=np.float64).reshape(-1, 5)


def write_yolo(path: Path, rows: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n")


def place(src: Path, dst: Path, link: bool = False) -> None:
    """Chép (mặc định) hoặc liên kết symlink. Google Drive không hỗ trợ symlink nên mặc định là chép."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def write_data_yaml(path: Path, root: Path, comment: str = "") -> None:
    lines = []
    if comment:
        lines += [f"# {c}" for c in comment.splitlines()]
    lines += [f"path: {Path(root).resolve().as_posix()}", "train: images/train", "val: images/val",
              "test: images/test", "", "nc: 4", "names:"]
    lines += [f"  {k}: {v}" for k, v in NAMES.items()]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_boxes(labels_dir: Path) -> dict[str, int]:
    c = {v: 0 for v in NAMES.values()}
    for f in Path(labels_dir).glob("*.txt"):
        for r in read_yolo(f):
            n = NAMES.get(int(r[0]), str(int(r[0])))
            c[n] = c.get(n, 0) + 1
    return c


# ----------------------------------------------------------------- giáo viên
def load_models(weight_paths, device=None):
    from ultralytics import YOLO
    return [YOLO(str(p)) for p in weight_paths]


def teacher_predict(models, img_bgr, imgsz=960, conf_floor=0.10, iou=0.6, device=None, tta=False):
    """Dự đoán bằng 1 mô hình, hoặc ghép bằng WBF nếu có nhiều mô hình.

    Trả về (xyxy [n,4] pixel, conf [n], cls [n]). Với nhiều mô hình, độ tin cậy sau ghép là
    trung bình độ tin cậy nhân min(số mô hình thấy, tổng)/tổng: hộp chỉ một mô hình thấy
    bị giảm một nửa (hai mô hình), nên ngưỡng áp lên giá trị đã ghép.
    """
    outs = []
    for m in models:
        r = m.predict(img_bgr, imgsz=imgsz, conf=conf_floor, iou=iou, max_det=300, device=device,
                      augment=tta, verbose=False)[0]
        b = r.boxes
        outs.append((b.xyxy.cpu().numpy().astype(np.float32), b.conf.cpu().numpy().astype(np.float32),
                     b.cls.cpu().numpy().astype(int)))
    if len(outs) == 1:
        return outs[0]
    from no_retrain_experiments import wbf
    return wbf(outs, n_models=len(outs))


def write_multi_yaml(path: Path, train_dirs, val_dir: Path, test_dir: Path | None = None,
                     comment: str = "") -> None:
    """File dataset với nhiều thư mục train (đường dẫn tuyệt đối). Ultralytics tìm nhãn bằng
    cách thay '/images/' bằng '/labels/' trong đường dẫn ảnh, nên mỗi thư mục train cần có
    thư mục labels song song. Cách này không phải chép lại ảnh của tập gốc."""
    import json
    q = lambda p: json.dumps(Path(p).resolve().as_posix(), ensure_ascii=False)  # noqa: E731
    lines = [f"# {c}" for c in comment.splitlines()] if comment else []
    lines += [f"path: {q(Path(path).parent)}", "train:"]
    lines += [f"  - {q(d)}" for d in train_dirs]
    lines += [f"val: {q(val_dir)}"]
    if test_dir is not None:
        lines += [f"test: {q(test_dir)}"]
    lines += ["", "nc: 4", "names:"] + [f"  {k}: {v}" for k, v in NAMES.items()]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def box_stats(labels_dir: Path, small_frac: float = 0.0025) -> dict:
    """Số ảnh, số hộp theo lớp, hộp/ảnh, tỉ lệ hộp nhỏ (diện tích chuẩn hoá < small_frac)."""
    files = list(Path(labels_dir).glob("*.txt"))
    per = {v: 0 for v in NAMES.values()}
    n = small = 0
    for f in files:
        for r in read_yolo(f):
            n += 1
            k = NAMES.get(int(r[0]), str(int(r[0])))
            per[k] = per.get(k, 0) + 1
            small += (r[3] * r[4]) < small_frac
    return {"images": len(files), "boxes": n, "per_class": per,
            "boxes_per_image": round(n / max(len(files), 1), 2),
            "small_box_share": round(small / max(n, 1), 4)}
