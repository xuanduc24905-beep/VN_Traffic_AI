#!/usr/bin/env python3
"""Danh gia thong nhat YOLOv8s va Faster R-CNN bang cung pycocotools (gop y phan bien bai bao SSRC).

- Cung tep nhan (chuyen tu nhan YOLO sang COCO), cung toi da 300 hop moi anh, cung nguong diem 0,001.
- Faster R-CNN chay hai che do: cu (100 hop, diem 0,05) de doi chieu so 0,6408, va moi (300 hop, diem 0,001).
- Luu du doan tung anh (.npz) de bootstrap o 81_bootstrap_ci.py.

Chay:  python scripts/verify/80_unified_coco_eval.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("YOLO_VERBOSE", "False")
ROOT = Path("/home/xuand/vn_traffic_ai")
OUT = ROOT / "outputs/verify/unified_eval"
OUT.mkdir(parents=True, exist_ok=True)
NAMES = ["motorcycle", "car", "bus", "truck"]
MAXDET = 300
FRCNN_V5PLUS = "weights/trained/frcnn_v5plus/best.pt"

SETS = {
    "v5plus_test": ROOT / "data/merged_v5plus/images/test",
    "aerial_val": ROOT / "data/vn_vehicles_aerial/images/val",
    "aerial_test": ROOT / "data/vn_vehicles_aerial/images/test",
}
YOLO_MODELS = {
    "exp_baseline": "runs/detect/exp_baseline/weights/best.pt",
    "r_clspw_v2": "runs/detect/r_clspw_v2/weights/best.pt",
    "ctrl_cw0_s0": "runs/detect/ctrl_cw0_s0_partial/weights/best.pt",  # luot doi chung cu, dung o epoch 3/10
}
# doi chung day du (62_clspw_control.py): cls_pw 0 / 0,5 x hat giong 0,1,2, moi luot 10 epoch
for _c in ("cw0", "cw05"):
    for _s in range(3):
        YOLO_MODELS[f"{_c}_s{_s}"] = f"runs/detect/ctrl_{_c}_s{_s}/weights/best.pt"


def list_images(d: Path):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def label_path(img: Path) -> Path:
    return Path(str(img.parent).replace("/images/", "/labels/")) / (img.stem + ".txt")


def build_gt(imgs):
    from PIL import Image
    images, anns, aid = [], [], 1
    for i, p in enumerate(imgs):
        w, h = Image.open(p).size
        images.append({"id": i, "file_name": p.name, "width": w, "height": h})
        lp = label_path(p)
        if lp.exists():
            for ln in lp.read_text().splitlines():
                s = ln.split()
                if len(s) < 5:
                    continue
                c, xc, yc, bw, bh = int(s[0]), *map(float, s[1:5])
                bw, bh = bw * w, bh * h
                x0, y0 = xc * w - bw / 2, yc * h - bh / 2
                anns.append({"id": aid, "image_id": i, "category_id": c + 1, "bbox": [x0, y0, bw, bh],
                             "area": bw * bh, "iscrowd": 0})
                aid += 1
    cats = [{"id": k + 1, "name": n} for k, n in enumerate(NAMES)]
    return {"images": images, "annotations": anns, "categories": cats}


def coco_eval(gt: dict, dets: list):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = gt
        cg.createIndex()
        cd = cg.loadRes(dets) if dets else None
        E = COCOeval(cg, cd, "bbox")
        E.params.maxDets = [1, 10, MAXDET]
        E.evaluate(); E.accumulate(); E.summarize()
    prec = E.eval["precision"]  # [T, R, K, A, M]
    def ap(t, k):
        p = prec[t, :, k, 0, -1]
        p = p[p > -1]
        return float(p.mean()) if p.size else float("nan")
    per50 = {NAMES[k]: ap(0, k) for k in range(4)}
    per5095 = {NAMES[k]: float(np.mean([ap(t, k) for t in range(prec.shape[0])])) for k in range(4)}
    allp = prec[:, :, :, 0, -1]
    # stats[0] cua pycocotools mac dinh maxDets=100 (khong co trong params) nen tinh truc tiep tu mang precision
    return {"mAP50": float(E.stats[1]), "mAP50_95": float(allp[allp > -1].mean()),
            "AP50_per_class": per50, "AP50_95_per_class": per5095}


def yolo_predict(ckpt: str, imgs):
    from ultralytics import YOLO
    m = YOLO(str(ROOT / ckpt))
    out = []
    for p in imgs:
        r = m.predict(source=str(p), imgsz=960, conf=0.001, iou=0.7, max_det=MAXDET, half=True,
                      device=0, verbose=False)[0]
        b = r.boxes
        out.append((b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)))
    return out


def frcnn_predict(imgs, maxdet: int, score: float, ckpt: str = "weights/trained/frcnn/best.pt"):
    import cv2
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.transforms import functional as F
    st = torch.load(ROOT / ckpt, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, min_size=800, max_size=1333,
                                box_score_thresh=score, box_detections_per_img=maxdet, box_nms_thresh=0.5)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 5)
    m.load_state_dict(sd)
    m.eval().cuda()
    out = []
    with torch.inference_mode():
        for p in imgs:
            im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            o = m([F.to_tensor(im).cuda()])[0]
            out.append((o["boxes"].cpu().numpy(), o["scores"].cpu().numpy(), o["labels"].cpu().numpy().astype(int) - 1))
    del m
    torch.cuda.empty_cache()
    return out


def to_coco(preds):
    dets = []
    for i, (xyxy, sc, cl) in enumerate(preds):
        for b, s, c in zip(xyxy, sc, cl):
            dets.append({"image_id": i, "category_id": int(c) + 1,
                         "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                         "score": float(s)})
    return dets


def save_npz(path: Path, preds):
    np.savez_compressed(path, **{f"b{i}": np.concatenate([x, s[:, None], c[:, None]], 1) if len(s) else np.zeros((0, 6))
                                 for i, (x, s, c) in enumerate(preds)})


def rescore():
    """Cham lai tu .npz da luu (khong chay mo hinh)."""
    res_path = OUT / "unified_eval.json"
    res = json.loads(res_path.read_text())
    for key in list(res):
        sname, name = key.split("/")
        gt = json.loads((OUT / f"gt_{sname}.json").read_text())
        z = np.load(OUT / f"pred_{sname}_{name}.npz")
        preds = [(z[f"b{i}"][:, :4], z[f"b{i}"][:, 4], z[f"b{i}"][:, 5].astype(int)) for i in range(len(gt["images"]))]
        res[key].update(coco_eval(gt, to_coco(preds)))
    res_path.write_text(json.dumps(res, indent=1))


def main():
    if "--rescore" in sys.argv:
        return rescore()
    res_path = OUT / "unified_eval.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    for sname, d in SETS.items():
        imgs = list_images(d)
        gt = build_gt(imgs)
        (OUT / f"gt_{sname}.json").write_text(json.dumps(gt))
        info = {"n_images": len(imgs), "n_gt": len(gt["annotations"])}
        runs = [(k, lambda k=k: yolo_predict(YOLO_MODELS[k], imgs)) for k in YOLO_MODELS]
        runs += [("frcnn_300", lambda: frcnn_predict(imgs, 300, 0.001)),
                 ("frcnn_old100", lambda: frcnn_predict(imgs, 100, 0.05))]
        if (ROOT / FRCNN_V5PLUS).exists():  # Faster R-CNN train lai tren V5+ (Colab)
            runs += [("frcnn_v5plus_300", lambda: frcnn_predict(imgs, 300, 0.001, FRCNN_V5PLUS))]
        for name, fn in runs:
            key = f"{sname}/{name}"
            if key in res:
                continue
            t0 = time.time()
            preds = fn()
            save_npz(OUT / f"pred_{sname}_{name}.npz", preds)
            r = coco_eval(gt, to_coco(preds))
            r.update(info, sec=round(time.time() - t0, 1),
                     max_dets_per_img=int(max(len(p[1]) for p in preds)))
            res[key] = r
            print(key, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                                   if not isinstance(v, dict)}), flush=True)
            res_path.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
