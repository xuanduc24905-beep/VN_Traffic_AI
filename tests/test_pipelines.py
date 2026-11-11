"""Kiểm thử các script trong scripts/pipelines bằng dữ liệu và mô hình giả (không cần GPU).

Chạy:  python -m pytest tests/test_pipelines.py -q      (hoặc: python tests/test_pipelines.py)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "pipelines"))

import common  # noqa: E402
import p1_pseudo_small as p1  # noqa: E402
import p2_build_datasets as p2  # noqa: E402
import p4_tiles as p4  # noqa: E402


# ------------------------------------------------------------------ tiện ích giả lập
class _T:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _T(xyxy), _T(conf), _T(cls)


class _Result:
    def __init__(self, xyxy, conf, cls):
        self.boxes = _Boxes(xyxy, conf, cls)


class FakeModel:
    """Trả về cùng một tập dự đoán cho mọi ảnh (giả sử ảnh 1000x1000)."""

    def __init__(self, xyxy, conf, cls):
        self.args = (np.array(xyxy, np.float32).reshape(-1, 4), np.array(conf, np.float32),
                     np.array(cls, int))

    def predict(self, img, **kw):
        return [_Result(*self.args)]


def _img(path: Path, w=1000, h=1000, color=200):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((h, w, 3), color, np.uint8))


# ------------------------------------------------------------------ common
def test_camera_id_and_sets(tmp_path):
    assert common.camera_id("hcm_phuong-linh-xuan_000224_20261105_114211.jpg") == "000224"
    assert common.camera_id("hcm_unknown_000114_20261106_051412.txt") == "000114"
    assert common.camera_id("soict24_cam_07_00999.jpg") is None
    f = tmp_path / "cams.txt"
    f.write_text("# test\n000026\n96  # sau\n\n")
    assert common.load_camera_set(str(f)) == {"000026", "000096"}
    assert common.load_camera_set("1, 000002") == {"000001", "000002"}
    y = tmp_path / "bl.yaml"
    y.write_text("blacklist:\n  '000026':\n    reason: x\n  '000096':\n    reason: y\n")
    assert common.load_camera_set(str(y)) == {"000026", "000096"}
    real = REPO / "configs" / "hcm_cameras_blacklist.yaml"
    if real.exists():
        assert len(common.load_camera_set(str(real))) >= 1


def test_label_path_for():
    assert common.label_path_for(Path("/a/images/train/x.jpg")) == Path("/a/labels/train/x.txt")
    assert common.label_path_for(Path("/images/v10/train/images/x.png")) == Path("/images/v10/train/labels/x.txt")
    with pytest.raises(ValueError):
        common.label_path_for(Path("/a/b/x.jpg"))


# ------------------------------------------------------------------ p1
W = H = 1000
THR = {0: 0.60, 1: 0.45, 2: 0.50, 3: 0.50}
#            x1   y1   x2   y2     conf  cls  ghi chú
DETS = [(100, 100, 300, 300, 0.90, 0),   # xe máy lớn, chắc chắn
        (500, 500, 520, 550, 0.90, 0),   # xe máy nhỏ (1000 px2 = 0,1%), chắc chắn
        (600, 600, 620, 650, 0.30, 0),   # xe máy nhỏ, nghi ngờ
        (700, 700, 900, 900, 0.30, 1),   # ô tô lớn (4% ảnh) lớn hơn ngưỡng mask, conf thấp: bỏ, không tô
        (110, 110, 130, 150, 0.30, 0),   # nghi ngờ nhưng nằm trong xe máy lớn: không tô
        (900, 100, 910, 120, 0.05, 0)]   # conf quá thấp: bỏ


def _dets():
    a = np.array(DETS, np.float32)
    return a[:, :4], a[:, 4], a[:, 5].astype(int)


def test_select_boxes_modes():
    xyxy, conf, cls = _dets()
    r, mb, st = p1.select_boxes(xyxy, conf, cls, W, H, "drop", THR)
    assert len(r) == 1 and st["small_kept"] == 0 and len(mb) == 0
    r, mb, st = p1.select_boxes(xyxy, conf, cls, W, H, "keep", THR)
    assert len(r) == 2 and st["small_kept"] == 1 and len(mb) == 0
    r, mb, st = p1.select_boxes(xyxy, conf, cls, W, H, "mask", THR)
    assert len(r) == 2 and len(mb) == 1 and np.allclose(mb[0], [600, 600, 620, 650])
    # nhãn YOLO chuẩn hoá đúng
    big = r[np.argmax(r[:, 3])]
    assert np.allclose(big, [0, 0.2, 0.2, 0.2, 0.2])
    # --small-thr hạ ngưỡng riêng cho hộp nhỏ: hộp nhỏ 0,30 được giữ khi small_thr=0,25
    r2, _, _ = p1.select_boxes(xyxy, conf, cls, W, H, "keep", THR, small_thr=0.25)
    assert len(r2) == 4


def test_apply_masks():
    img = np.full((100, 100, 3), 200, np.uint8)
    p1.apply_masks(img, np.array([[10, 10, 30, 30]], np.float32), pad=0)
    assert (img[15, 15] == p1.GRAY).all() and (img[50, 50] == 200).all()


def test_generate_end_to_end(tmp_path):
    src = tmp_path / "frames"
    for n in ["hcm_a_000001_20261105_100000.jpg", "hcm_b_000002_20261105_100001.jpg"]:
        _img(src / n)
    xyxy, conf, cls = _dets()
    model = FakeModel(xyxy, conf, cls)
    imgs = common.list_images(src)
    out = tmp_path / "out"
    st = p1.generate(imgs, [model], out, "mask", THR, progress_every=0)
    assert st["images_kept"] == 2 and st["masked_boxes"] == 2 and st["small_kept"] == 2
    assert len(list((out / "images" / "train").glob("*.jpg"))) == 2
    lab = common.read_yolo(out / "labels" / "train" / "hcm_a_000001_20261105_100000.txt")
    assert lab.shape == (2, 5)
    im = cv2.imread(str(out / "images" / "train" / "hcm_a_000001_20261105_100000.jpg"))
    assert (im[625, 610] == p1.GRAY).all()                    # vùng nghi ngờ đã tô xám
    assert (im[200, 200] != p1.GRAY).any()                    # xe máy lớn còn nguyên
    # ảnh không còn hộp nào bị loại
    st0 = p1.generate(imgs, [FakeModel([], [], [])], tmp_path / "out0", "mask", THR, progress_every=0)
    assert st0["images_kept"] == 0


def test_p1_cli_dry_run_excludes_cameras(tmp_path):
    src = tmp_path / "frames"
    for n in ["hcm_a_000001_20261105_100000.jpg", "hcm_b_000002_20261105_100001.jpg"]:
        _img(src / n)
    r = subprocess.run([sys.executable, str(REPO / "scripts/pipelines/p1_pseudo_small.py"), "--images", str(src),
                        "--out", str(tmp_path / "o"), "--exclude-cameras", "000002", "--dry-run"],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    assert "sau loại camera (1 mã) và giới hạn: 1" in r.stdout


# ------------------------------------------------------------------ p2
def _yolo_dir(root: Path, split: str, n: int, prefix: str, label_rows: str):
    for i in range(n):
        _img(root / split / "images" / f"{prefix}{i}.jpg", 64, 64)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        (root / split / "labels" / f"{prefix}{i}.txt").write_text(label_rows)


def _base(root: Path):
    for sp, n in (("train", 4), ("val", 2), ("test", 2)):
        for i in range(n):
            _img(root / "images" / sp / f"b{sp}{i}.jpg", 64, 64)
            (root / "labels" / sp).mkdir(parents=True, exist_ok=True)
            (root / "labels" / sp / f"b{sp}{i}.txt").write_text("1 0.5 0.5 0.2 0.2\n")


def test_collect_pairs_remap_v10(tmp_path):
    v10 = tmp_path / "v10"
    _yolo_dir(v10, "train", 3, "ok", "4 0.5 0.5 0.1 0.1\n5 0.3 0.3 0.1 0.1\n0 0.7 0.7 0.1 0.1\n")
    _yolo_dir(v10, "train", 2, "bg", "2 0.5 0.5 0.1 0.1\n4 0.2 0.2 0.1 0.1\n")   # có xe ba gác (lớp 2)
    pairs, dropped = p2.collect_pairs(v10, remap_v10=True)
    assert len(pairs) == 3 and dropped == 2
    classes = sorted(pairs[0][1][:, 0].astype(int).tolist())
    assert classes == [0, 3, 3]                      # xe_may -> 0, xe_tai và container -> 3
    pairs2, dropped2 = p2.collect_pairs(v10, remap_v10=True, keep_dropped=True)
    assert len(pairs2) == 5 and dropped2 == 0
    assert all(set(r[:, 0].astype(int)) <= {0, 1, 2, 3} for _, r in pairs2)


def test_sample_budget_boxes():
    pairs = [(Path(f"{i}.jpg"), np.zeros((i + 1, 5))) for i in range(10)]
    ch = p2.sample_budget(pairs, 12, "boxes", 0)
    assert sum(len(r) for _, r in ch) >= 12
    assert len(p2.sample_budget(pairs, 3, "images", 0)) == 3
    assert p2.sample_budget(pairs, 3, "images", 0) == p2.sample_budget(pairs, 3, "images", 0)   # cố định theo seed


def test_p2_cli_builds_yaml_that_ultralytics_reads(tmp_path):
    base, v10, pseudo, out = tmp_path / "base", tmp_path / "v10", tmp_path / "pseudo", tmp_path / "out"
    _base(base)
    _yolo_dir(v10, "train", 6, "h", "4 0.5 0.5 0.1 0.1\n")
    pseudo_dir = pseudo
    for i in range(6):
        _img(pseudo_dir / "images" / "train" / f"p{i}.jpg", 64, 64)
        (pseudo_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (pseudo_dir / "labels" / "train" / f"p{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    r = subprocess.run([sys.executable, str(REPO / "scripts/pipelines/p2_build_datasets.py"), "--base", str(base),
                        "--out", str(out), "--hand", str(v10), "--remap-v10", "--pseudo", str(pseudo),
                        "-n", "3", "--arms", "base,hand,pseudo"], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(list((out / "hand" / "images" / "train").glob("hand_*"))) == 3
    assert len(list((out / "pseudo" / "images" / "train").glob("pseudo_*"))) == 3
    lab = common.read_yolo(next((out / "hand" / "labels" / "train").glob("*.txt")))
    assert int(lab[0, 0]) == 0                      # xe_may (4) -> motorcycle (0)
    try:
        from ultralytics.data.utils import check_det_dataset
    except Exception:
        pytest.skip("không có ultralytics")
    d = check_det_dataset(str(out / "hand.yaml"), autodownload=False)
    assert isinstance(d["train"], list) and len(d["train"]) == 2 and Path(d["val"]).exists()
    d0 = check_det_dataset(str(out / "base.yaml"), autodownload=False)
    assert len(d0["train"]) == 1 if isinstance(d0["train"], list) else True


# ------------------------------------------------------------------ p4
def test_tile_labels_geometry():
    # ảnh 1280x720; hộp A nằm trọn trong ô (0,0,640,640); hộp B cắt ngang biên x=640
    rows = np.array([[0, 100 / 1280, 100 / 720, 40 / 1280, 40 / 720],
                     [0, 640 / 1280, 300 / 720, 100 / 1280, 60 / 720]])
    t = p4.tile_labels(rows, 1280, 720, (0, 0, 640, 640))
    assert len(t) == 2                              # B còn 50% trong ô, đúng ngưỡng 0,5
    assert np.allclose(t[0], [0, 100 / 640, 100 / 640, 40 / 640, 40 / 640])
    assert np.allclose(t[1][1:3], [(640 - 25) / 640, 300 / 640])   # phần B nằm trong ô: x từ 590 đến 640
    assert np.isclose(t[1][3], 50 / 640)
    t2 = p4.tile_labels(rows, 1280, 720, (480, 0, 1120, 640), min_visible=0.9)
    assert len(t2) == 1                             # A ngoài ô; B nằm trọn
    assert p4.tile_labels(rows, 1280, 720, (700, 0, 1280, 640)).shape == (0, 5)


def test_p4_cli(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _base(src)
    _img(src / "images" / "train" / "big.jpg", 1280, 720)
    (src / "labels" / "train" / "big.txt").write_text("0 0.1 0.15 0.03 0.05\n1 0.8 0.6 0.05 0.08\n")
    r = subprocess.run([sys.executable, str(REPO / "scripts/pipelines/p4_tiles.py"), "--src", str(src),
                        "--out", str(out), "--keep-empty", "0"], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    m = json.loads((out / "tiles_manifest.json").read_text(encoding="utf-8"))
    assert m["images_tiled"] == 1 and m["images_not_tiled"] == 4 and m["tiles_written"] >= 2
    tiles = list((out / "images" / "train").glob("big_t*.jpg"))
    assert tiles and cv2.imread(str(tiles[0])).shape[:2] == (640, 640)
    try:
        from ultralytics.data.utils import check_det_dataset
    except Exception:
        pytest.skip("không có ultralytics")
    d = check_det_dataset(str(out / "tiles.yaml"), autodownload=False)
    assert len(d["train"]) == 2


# ------------------------------------------------------------------ train.py
@pytest.mark.parametrize("recipe", ["v5plus", "finetune", "imgsz1280", "multiscale", "clspw", "scale", "p2",
                                    "progressive"])
def test_train_dry_run(recipe, tmp_path):
    ds = tmp_path / "ds"
    _base(ds)
    import common as c
    c.write_data_yaml(tmp_path / "d.yaml", ds)
    r = subprocess.run([sys.executable, str(REPO / "scripts/pipelines/train.py"), "--data", str(tmp_path / "d.yaml"),
                        "--name", "t", "--recipe", recipe, "--dry-run", "--epochs", "1"],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    plan = json.loads(r.stdout[r.stdout.index("{"):r.stdout.rindex("}") + 1])
    assert plan["recipe"] == recipe


def test_train_dry_run_distill_and_bad_key(tmp_path):
    ds = tmp_path / "ds"
    _base(ds)
    common.write_data_yaml(tmp_path / "d.yaml", ds)
    base = [sys.executable, str(REPO / "scripts/pipelines/train.py"), "--data", str(tmp_path / "d.yaml"),
            "--name", "t", "--dry-run"]
    r = subprocess.run(base + ["--distill-model", "x.pt", "--dis", "3"], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0 and '"distill_model"' in r.stdout and '"dis": 3.0' in r.stdout
    r = subprocess.run(base + ["--set", "khong_ton_tai=1"], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode != 0 and "không có tham số" in (r.stdout + r.stderr)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
