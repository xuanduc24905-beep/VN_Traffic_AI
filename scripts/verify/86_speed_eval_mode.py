#!/usr/bin/env python3
"""Đo độ trễ YOLOv8s-CW và Faster R-CNN ở ĐÚNG chế độ dùng để tính độ chính xác trong bài báo.

Bài báo so độ chính xác bằng:
  - YOLO: model.val() -> NMS đa nhãn, ngưỡng điểm 0,001, NMS IoU 0,7, tối đa 300 hộp, imgsz 960 (rect)
  - Faster R-CNN: ngưỡng điểm 0,001, NMS IoU 0,5, tối đa 300 hộp, cạnh ngắn 800 / dài 1.333
Lần đo cũ (85_speed_cw.py) dùng predict() mặc định (NMS một nhãn, ngưỡng 0,25) cho YOLO và ngưỡng mặc định
của TorchVision (0,05; 100 hộp) cho Faster R-CNN, nên KHÔNG cùng chế độ với số độ chính xác.

Ở đây, trong CÙNG một phiên, cùng 1.500 ảnh test V5+, batch 1, cùng phạm vi (không tính đọc ảnh từ đĩa):
  - YOLO: thời gian tiền xử lý + suy luận + hậu xử lý (NMS) do model.val() tự đo, FP32 và FP16,
          CỘNG thời gian letterbox (đổi kích thước + đệm) vốn chạy trong dataloader nên val() không tính;
          nhờ vậy cùng phạm vi với Faster R-CNN (có tính bước đổi kích thước bên trong mô hình)
  - Faster R-CNN: thời gian model([tensor]) gồm chuẩn hóa/đổi kích thước bên trong, suy luận và NMS, FP32
Chạy trong môi trường đánh giá của bài (vehicle_env: PyTorch 2.7.1, Ultralytics 8.4.126):
    /home/xuand/miniconda3/envs/vehicle_env/bin/python scripts/verify/86_speed_eval_mode.py
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/xuand/vn_traffic_ai")
CW = ROOT / "runs/detect/r_clspw_v2/weights/best.pt"
FRCNN = ROOT / "weights/trained/frcnn_v5plus/best.pt"
DATA = ROOT / "data/data_v5plus.yaml"
TEST_DIR = ROOT / "data/merged_v5plus/images/test"
OUT = ROOT / "outputs/verify/speed_eval_mode.json"


def yolo_val_speed(half: bool) -> dict:
    from ultralytics import YOLO

    m = YOLO(str(CW))
    r = m.val(data=str(DATA), split="test", imgsz=960, batch=1, conf=0.001, iou=0.7, max_det=300,
              half=half, device=0, plots=False, verbose=False, project="/tmp", name="speed_eval_mode",
              exist_ok=True)
    sp = r.speed  # ms / ảnh
    total = sp["preprocess"] + sp["inference"] + sp["postprocess"]
    return {"precision": "FP16" if half else "FP32", "mode": "model.val(): NMS đa nhãn, conf 0,001, IoU 0,7, max_det 300, rect",
            "images": 1500, "ms_preprocess": round(sp["preprocess"], 3), "ms_inference": round(sp["inference"], 3),
            "ms_postprocess": round(sp["postprocess"], 3), "ms_total": round(total, 3),
            "mAP50_ultralytics": round(float(r.box.map50), 4)}


def letterbox_ms() -> float:
    """Thời gian letterbox của val() (rect, batch 1): ảnh 1280x720 -> 960x540, đệm lên bội số 32 (576)."""
    import cv2
    from ultralytics.data.augment import LetterBox

    lb = LetterBox(new_shape=(576, 960), auto=False, scaleup=False, center=True)
    imgs = [cv2.imread(str(p)) for p in sorted(TEST_DIR.glob("*.jpg"))]
    for im in imgs[:20]:
        lb(image=im)
    t0 = time.perf_counter()
    for im in imgs:
        lb(image=im)
    return (time.perf_counter() - t0) * 1000 / len(imgs)


def frcnn_speed() -> dict:
    import cv2
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.transforms import functional as F

    st = torch.load(FRCNN, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    nc = st.get("num_classes", 5) if isinstance(st, dict) else 5
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=91,
                                    min_size=800, max_size=1333,
                                    box_score_thresh=0.001, box_nms_thresh=0.5, box_detections_per_img=300)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, nc)
    model.load_state_dict(sd)
    dev = torch.device("cuda:0")
    model.eval().to(dev)
    paths = sorted(TEST_DIR.glob("*.jpg"))
    imgs = [F.to_tensor(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)) for p in paths]  # đọc trước, không tính
    lat = []
    with torch.inference_mode():
        for t in imgs[:20]:
            model([t.to(dev)])
        torch.cuda.synchronize()
        for t in imgs:
            x = t.to(dev)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model([x])
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)
    return {"precision": "FP32", "mode": "score_thresh 0,001, NMS IoU 0,5, 300 hộp, cạnh 800/1.333",
            "images": len(lat), "ms_mean": round(float(lat.mean()), 3), "ms_median": round(float(np.median(lat)), 3),
            "ms_p95": round(float(np.percentile(lat, 95)), 3)}


def main():
    import torch

    res = {"gpu_processes_before": subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip(),
        "env": {"torch": torch.__version__, "device": torch.cuda.get_device_name(0)}}
    lb = letterbox_ms()
    res["yolo_letterbox_ms"] = round(lb, 3)
    for key, half in (("yolo_fp32", False), ("yolo_fp16", True)):
        r = yolo_val_speed(half)
        r["ms_total_incl_letterbox"] = round(r["ms_total"] + lb, 3)
        res[key] = r
    res["frcnn_fp32"] = frcnn_speed()
    res["ratio_frcnn_over_yolo_fp32"] = round(res["frcnn_fp32"]["ms_mean"] / res["yolo_fp32"]["ms_total_incl_letterbox"], 2)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
