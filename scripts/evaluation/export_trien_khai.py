"""Chuẩn bị triển khai: xuất model sang ONNX / TensorRT (FP16, INT8) rồi đo tốc độ và độ chính xác sau xuất.

Chạy bằng venv riêng có TensorRT (không đụng env train yolov8_ft):
  python -m venv --system-site-packages ~/venvs/yolov8_export   (dựa trên python của yolov8_ft)
  ~/venvs/yolov8_export/bin/pip install tensorrt-cu12==10.7.0 onnxruntime-gpu==1.20.2
  ~/venvs/yolov8_export/bin/python scripts/evaluation/export_trien_khai.py --weights <best.pt> --name <ten>

Định dạng (mỗi định dạng ở imgsz 640 và 960, ảnh vuông tĩnh, batch 1):
  pt_fp16    : PyTorch .pt, half=True (mốc)
  onnx       : ONNX (onnxslim), chạy bằng onnxruntime-gpu (CUDA EP)
  trt_fp16   : TensorRT FP16
  trt_int8   : TensorRT INT8, hiệu chỉnh (calibration) bằng N_CALIB ảnh TRAIN của V6split rải đều
               (không dùng ảnh val/test)
Tốc độ: ms/ảnh (tiền xử lý / suy luận / hậu xử lý) trung bình trên 300 ảnh SOICT test, sau 30 ảnh khởi động,
  GPU phải rảnh (không train song song) — script dừng nếu GPU đang có tiến trình khác.
Độ chính xác: mAP50 SOICT test (cam_09/10, nhãn người gán) và tỉ lệ bắt trúng xe ở khung ngã tư (mốc yolov8x,
  thiên về COCO) — cùng hàm với eval_chong_quen.py. LƯU Ý: định dạng xuất dùng ảnh vuông (rect=False), còn
  eval_chong_quen dùng rect=True → mốc .pt ở đây cũng chấm rect=False để so cùng điều kiện.
Kết quả: reports/trien_khai/<ten>.json, reports/trien_khai/<ten>.md; file xuất nằm cạnh best.pt.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
OUT = ROOT / "reports" / "trien_khai"
CALIB_DIR = ROOT / "data" / "calib_int8"  # gitignored (data/*), dựng lại được
N_CALIB, N_SPEED, N_WARM = 512, 300, 30
NAMES = ["motorcycle", "car", "bus", "truck"]


def gpu_busy() -> bool:
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def build_calib() -> Path:
    """N_CALIB ảnh train V6split (rải đều theo tên, gồm cả nguồn VN lẫn MIO/DETRAC) → yaml chỉ dùng cho hiệu chỉnh."""
    yaml = CALIB_DIR / "calib.yaml"
    if yaml.exists():
        return yaml
    src = ROOT / "data" / "merged_v6split"
    ims = sorted((src / "images" / "train").glob("*.jpg"))
    pick = [ims[i] for i in np.linspace(0, len(ims) - 1, N_CALIB).astype(int)]
    for sub in ("images", "labels"):
        (CALIB_DIR / sub / "train").mkdir(parents=True, exist_ok=True)
    for p in pick:
        shutil.copyfile(p, CALIB_DIR / "images" / "train" / p.name)
        shutil.copyfile(src / "labels" / "train" / f"{p.stem}.txt", CALIB_DIR / "labels" / "train" / f"{p.stem}.txt")
    yaml.write_text(f"# {N_CALIB} ảnh train V6split cho hiệu chỉnh INT8 — export_trien_khai.py\npath: {CALIB_DIR}\n"
                    "train: images/train\nval: images/train\n\nnc: 4\nnames:\n  0: motorcycle\n  1: car\n  2: bus\n  3: truck\n", encoding="utf-8")
    return yaml


def export(pt: Path, fmt: str, imgsz: int) -> Path:
    """Xuất (hoặc dùng lại) file định dạng fmt ở imgsz; trả đường dẫn. Tên có hậu tố _<imgsz>[_int8]."""
    from ultralytics import YOLO

    if fmt == "pt_fp16":
        return pt
    ext = {"onnx": ".onnx", "trt_fp16": ".engine", "trt_int8": ".engine"}[fmt]
    dst = pt.with_name(f"{pt.stem}_{imgsz}{'_int8' if fmt == 'trt_int8' else ''}{ext}")
    if dst.exists():
        return dst
    kw = dict(imgsz=imgsz, batch=1, dynamic=False)
    if fmt == "onnx":
        f = YOLO(str(pt)).export(format="onnx", simplify=True, **kw)
    elif fmt == "trt_fp16":
        f = YOLO(str(pt)).export(format="engine", half=True, workspace=4, **kw)
    else:
        f = YOLO(str(pt)).export(format="engine", int8=True, data=str(build_calib()), fraction=1.0, workspace=4, **kw)
    Path(f).rename(dst)
    return dst


def speed(model_path: Path, imgsz: int, half: bool) -> dict:
    from ultralytics import YOLO

    ims = sorted((ROOT / "data" / "merged_v6split" / "images" / "test").glob("*.jpg"))
    ims = random.Random(0).sample(ims, N_WARM + N_SPEED)
    m = YOLO(str(model_path), task="detect")
    sp = []
    for i, p in enumerate(ims):
        r = m.predict(str(p), imgsz=imgsz, half=half, conf=0.25, verbose=False, device=0)[0]
        if i >= N_WARM:
            sp.append([r.speed["preprocess"], r.speed["inference"], r.speed["postprocess"]])
    a = np.array(sp)
    return {"pre_ms": round(float(a[:, 0].mean()), 2), "infer_ms": round(float(a[:, 1].mean()), 2),
            "infer_ms_p95": round(float(np.percentile(a[:, 1], 95)), 2), "post_ms": round(float(a[:, 2].mean()), 2),
            "total_ms": round(float(a.sum(1).mean()), 2), "fps": round(1000 / float(a.sum(1).mean()), 1)}


def accuracy(model_path: Path, imgsz: int, half: bool) -> dict:
    from ultralytics import YOLO
    from eval_chong_quen import prepare_frames
    from ultralytics.utils.metrics import box_iou
    import torch

    m = YOLO(str(model_path), task="detect")
    r = m.val(data=str(ROOT / "data" / "data_v6split.yaml"), split="test", imgsz=imgsz, batch=1, rect=False, half=half,
              conf=0.001, iou=0.7, plots=False, verbose=False, project="/tmp", name="trien_khai_eval", exist_ok=True)
    out = {"soict_mAP50": round(float(r.box.map50), 4), "soict_mAP50_95": round(float(r.box.map), 4),
           "soict_AP50": {NAMES[c]: round(float(r.box.ap50[i]), 4) for i, c in enumerate(r.box.ap_class_index)}}
    # ngã tư: cùng cách tính với eval_chong_quen.deploy_recall (conf 0.25, IoU >= 0.3, không xét lớp)
    ims, ref = prepare_frames()
    tp, ng = np.zeros(2), np.zeros(2)
    for p in ims:
        res = m.predict(str(p), imgsz=imgsz, conf=0.25, half=half, verbose=False, device=0)[0]
        g = np.array(ref[p.name]).reshape(-1, 5)
        d = res.boxes.xyxy.cpu().float()
        for c in range(2):
            ng[c] += (g[:, 0] == c).sum()
        if len(g) and len(d):
            hit = (box_iou(torch.tensor(g[:, 1:]).float(), d).max(1).values >= 0.3).numpy()
            for c in range(2):
                tp[c] += hit[g[:, 0] == c].sum()
    out["ngatu_recall_moto"] = round(float(tp[0] / ng[0]), 4)
    out["ngatu_recall_car"] = round(float(tp[1] / ng[1]), 4)
    return out


def write_md(name: str, res: dict):
    rows = [f"# Triển khai — {name}\n", f"Model: `{res['weights']}`. GPU: {res.get('gpu', '?')}. "
            f"TensorRT {res.get('tensorrt', '?')}, onnxruntime {res.get('onnxruntime', '?')}.\n",
            "Tốc độ: trung bình trên 300 ảnh SOICT test (batch 1, sau 30 ảnh khởi động). SOICT mAP50 chấm ảnh vuông "
            "(rect=False) nên hơi khác số trong reports/chong_quen. Ngã tư: mốc yolov8x (thiên về COCO).\n",
            "| Định dạng | imgsz | Kích thước file (MB) | Suy luận (ms) | Tổng (ms) | FPS | SOICT mAP50 | Xe máy / Xe tải SOICT "
            "| Ngã tư xe máy / ô tô |", "|---|---|---|---|---|---|---|---|---|"]
    for k, e in res["formats"].items():
        if "error" in e:
            rows.append(f"| {e['fmt']} | {e['imgsz']} | – | lỗi: {e['error'][:80]} | | | | | |")
            continue
        s, a = e["speed"], e["acc"]
        rows.append(f"| {e['fmt']} | {e['imgsz']} | {e['size_mb']} | {s['infer_ms']} | {s['total_ms']} | {s['fps']} | "
                    f"{a['soict_mAP50']:.3f} | {a['soict_AP50'].get('motorcycle', 0):.3f} / "
                    f"{a['soict_AP50'].get('truck', 0):.3f} | {a['ngatu_recall_moto']:.3f} / {a['ngatu_recall_car']:.3f} |")
    (OUT / f"{name}.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--formats", nargs="*", default=["pt_fp16", "onnx", "trt_fp16", "trt_int8"])
    ap.add_argument("--imgsz", nargs="*", type=int, default=[640, 960])
    ap.add_argument("--allow-busy-gpu", action="store_true", help="chỉ để thử script; số đo tốc độ không dùng được")
    a = ap.parse_args()
    if gpu_busy() and not a.allow_busy_gpu:
        sys.exit("GPU đang có tiến trình khác — chờ train xong rồi chạy lại (số đo tốc độ sẽ sai)")
    import onnxruntime
    import torch
    try:
        import tensorrt
        trt_v = tensorrt.__version__
    except ImportError:
        trt_v = None
    OUT.mkdir(parents=True, exist_ok=True)
    pt = (ROOT / a.weights).resolve()
    rj = OUT / f"{a.name}.json"
    res = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else {"formats": {}}
    res.update(weights=str(pt.relative_to(ROOT)), gpu=torch.cuda.get_device_name(0), tensorrt=trt_v,
               onnxruntime=onnxruntime.__version__, busy_gpu=a.allow_busy_gpu)
    for fmt in a.formats:
        for sz in a.imgsz:
            key = f"{fmt}_{sz}"
            t0 = time.time()
            try:
                f = export(pt, fmt, sz)
                half = fmt in ("pt_fp16", "trt_fp16")
                e = {"fmt": fmt, "imgsz": sz, "file": str(f.relative_to(ROOT)),
                     "size_mb": round(f.stat().st_size / 2 ** 20, 1), "speed": speed(f, sz, half), "acc": accuracy(f, sz, half)}
            except Exception as ex:  # ghi lỗi, làm tiếp định dạng khác
                e = {"fmt": fmt, "imgsz": sz, "error": f"{type(ex).__name__}: {ex}"}
            e["seconds"] = round(time.time() - t0)
            res["formats"][key] = e
            rj.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
            write_md(a.name, res)
            print(key, json.dumps({k: e.get(k) for k in ("size_mb", "speed", "acc", "error")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
