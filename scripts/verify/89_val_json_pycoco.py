#!/usr/bin/env python3
"""Kiem tra nguyen nhan chenh ~4,5 diem giua model.val() va pycocotools tren du doan luu bang predict():
chay model.val(save_json=True) (NMS nhieu nhan, nhu luc cham Bang III) roi cham file du doan do bang pycocotools,
cung tep nhan voi 80_unified_coco_eval.py."""
import contextlib
import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "False")
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

ROOT = Path("/home/xuand/vn_traffic_ai")
D = ROOT / "outputs/verify/unified_eval"
gt = json.loads((D / "gt_v5plus_test.json").read_text())
by_stem = {Path(i["file_name"]).stem: i["id"] for i in gt["images"]}
MODELS = {"exp_baseline": "runs/detect/exp_baseline/weights/best.pt",
          "r_clspw_v2": "runs/detect/r_clspw_v2/weights/best.pt"}
for k in sys.argv[1:]:
    MODELS[k] = f"runs/detect/ctrl_{k}/weights/best.pt"
out = {}
for name, ck in MODELS.items():
    proj = f"/tmp/claude-1000/valjson/{name}"
    m = YOLO(str(ROOT / ck))
    r = m.val(data=str(ROOT / "data/data_v5plus.yaml"), split="test", imgsz=960, batch=16, conf=0.001, iou=0.7,
              max_det=300, half=True, device=0, plots=False, verbose=False, save_json=True, project=proj,
              name="v", exist_ok=True)
    preds = json.loads((Path(proj) / "v/predictions.json").read_text())
    dets = [{"image_id": by_stem[str(p["image_id"])], "category_id": int(p["category_id"]), "bbox": p["bbox"],
             "score": p["score"]} for p in preds]
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO(); cg.dataset = gt; cg.createIndex(); cd = cg.loadRes(dets)
        E = COCOeval(cg, cd, "bbox"); E.params.maxDets = [1, 10, 300]; E.evaluate(); E.accumulate(); E.summarize()
    out[name] = {"val_mAP50": float(r.box.map50), "pycoco_on_val_preds_mAP50": float(E.stats[1]),
                 "n_preds": len(dets), "max_per_img": max(sum(1 for d in dets if d["image_id"] == i) for i in range(5))}
    print(name, out[name], flush=True)
(D / "val_json_pycoco.json").write_text(json.dumps(out, indent=1))
