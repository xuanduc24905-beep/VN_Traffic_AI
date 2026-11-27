"""Đo các cách chống quên kiến thức COCO (WiSE-FT, LR thấp, L2-SP, EWC) theo hướng triển khai.

Chỉ số cho mỗi model:
  - deploy_ngatu: khung hình từ 2 video ngã tư đông xe máy, quay chéo từ trên (Release test-videos-v1).
      Mốc = xe yolov8x COCO thấy (imgsz 1280, conf 0.4). Báo tỉ lệ bắt trúng (IoU >= 0.3, không xét lớp)
      cho xe máy và ô tô, ở conf 0.25 và 0.1.
      LƯU Ý: mốc là yolov8x → thiên về các model giống COCO. Phải đọc cùng SOICT/v10 (nhãn người gán).
  - ext (Ấn Độ, camera VN; nhãn NHÁP yolov8x), soict_test, vn_v10, ext_map: như eval_v5fix.py
Kết quả: reports/chong_quen/results.json và results.md

Chạy: python scripts/evaluation/eval_chong_quen.py [--models ten ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
from eval_v5fix import ext_recall, map_val  # noqa: E402

OUT = ROOT / "reports" / "chong_quen"
VIDEOS = ROOT / "data" / "deploy_videos"  # gitignored: video iStock có bản quyền
FRAMES = ROOT / "data" / "deploy_frames"
N_FRAMES = {"nga_tu_dong_xe_may_1.mp4": 24, "nga_tu_dong_xe_may_2.mp4": 8}
REF_WEIGHTS, REF_CONF = "yolov8x.pt", 0.4
COCO_TO_V = {3: 0, 2: 1, 5: 2, 7: 3}
R = "runs/detect"
MODELS = {
    "coco_yolov8s": "yolov8s.pt",
    "r_clspw_v2": f"{R}/r_clspw_v2/weights/best.pt",
    "S1_freeze10": f"{R}/v6split_f1_freeze10/weights/best.pt",
    "S2_nofreeze": f"{R}/v6split_f2_nofreeze/weights/best.pt",
}
for _p in sorted((ROOT / "weights" / "wise").glob("wise_*.pt")):  # mọi bản trộn có trong thư mục
    MODELS.setdefault(_p.stem, f"weights/wise/{_p.name}")
MODELS.update({
    "lowlr_S2": f"{R}/cq_lowlr_s2/weights/best.pt",
    "l2sp_lam_lo": f"{R}/cq_l2sp_lo/weights/best.pt",
    "l2sp_lam_mid": f"{R}/cq_l2sp_mid/weights/best.pt",
    "l2sp_lam_hi": f"{R}/cq_l2sp_hi/weights/best.pt",
    "ewc": f"{R}/cq_ewc/weights/best.pt",
    "l2sp_mid_coco5k": f"{R}/cq_l2sp_mid_coco5k/weights/best.pt",
    "t1_bigtruck": f"{R}/cq_l2sp_coco5k_bigtruck/weights/best.pt",
    "t2_truckx2": f"{R}/cq_l2sp_coco5k_truckx2/weights/best.pt",
    "t2_truckx2_s0": f"{R}/cq_l2sp_coco5k_truckx2_s0/weights/best.pt",
    "t2_truckx2_s1": f"{R}/cq_l2sp_coco5k_truckx2_s1/weights/best.pt",
    "t2_pseudo": f"{R}/ck_t2_pseudo60/weights/best.pt",
    "t2_pseudo50": f"{R}/ck_t2_pseudo50/weights/best.pt",
    "t2_pseudo70": f"{R}/ck_t2_pseudo70/weights/best.pt",
    "t2_ghep": f"{R}/ck_t2_ghep60/weights/best.pt",
    "t2_ghep50": f"{R}/ck_t2_ghep50/weights/best.pt",
    "t2_ghep70": f"{R}/ck_t2_ghep70/weights/best.pt",
})


def prepare_frames():
    """Cắt khung đều từ video và tính mốc yolov8x một lần (lưu cache)."""
    ref_json = FRAMES / "ref_yolov8x.json"
    if ref_json.exists():
        return sorted(FRAMES.glob("*.jpg")), json.loads(ref_json.read_text())
    from ultralytics import YOLO

    FRAMES.mkdir(parents=True, exist_ok=True)
    for name, n in N_FRAMES.items():
        cap = cv2.VideoCapture(str(VIDEOS / name))
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for k in range(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k * (tot - 1) / max(n - 1, 1)))
            ok, f = cap.read()
            if ok:
                cv2.imwrite(str(FRAMES / f"{Path(name).stem}_{k:02d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
    ims = sorted(FRAMES.glob("*.jpg"))
    ref = {}
    for p, r in zip(ims, YOLO(str(ROOT / REF_WEIGHTS)).predict([str(p) for p in ims], imgsz=1280, conf=REF_CONF,
                                                               classes=list(COCO_TO_V), verbose=False)):
        ref[p.name] = [[COCO_TO_V[int(c)], *b] for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())]
    ref_json.write_text(json.dumps(ref))
    return ims, ref


def deploy_recall(weights: str, is_coco: bool, ims, ref) -> dict:
    from ultralytics import YOLO
    from ultralytics.utils.metrics import box_iou

    m = YOLO(str(ROOT / weights))
    out = {}
    for conf in (0.25, 0.1):
        tp, ng, npred = np.zeros(4), np.zeros(4), 0
        res = m.predict([str(p) for p in ims], imgsz=960, conf=conf, verbose=False, half=True,
                        classes=list(COCO_TO_V) if is_coco else None)
        for p, r in zip(ims, res):
            g = np.array(ref[p.name]).reshape(-1, 5)
            d = r.boxes.xyxy.cpu().float()
            npred += len(d)
            for c in range(4):
                ng[c] += (g[:, 0] == c).sum()
            if len(g) and len(d):
                io = box_iou(torch.tensor(g[:, 1:]).float(), d)
                hit = (io.max(1).values >= 0.3).numpy()
                for c in range(4):
                    tp[c] += hit[g[:, 0] == c].sum()
        out[str(conf)] = {"recall_moto": round(float(tp[0] / ng[0]), 4), "recall_car": round(float(tp[1] / ng[1]), 4),
                          "n_pred": int(npred), "n_ref": {"moto": int(ng[0]), "car": int(ng[1])}}
    return out


def write_md(res: dict):
    rows = ["| Model | Ngã tư xe máy (0,25 / 0,1) | Ngã tư ô tô (0,25) | SOICT mAP50 | Xe máy / Xe tải SOICT | "
            "Test v10 | Ấn Độ ô tô | Camera VN | Hộp khổng lồ | mAP tập ngoài |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "–" if v is None else f"{v:.3f}"  # noqa: E731
    for k, e in res.items():
        d, x = e.get("deploy", {}), e.get("ext", {})
        s, v = e.get("soict_test", {}), e.get("vn_v10", {})
        rows.append(
            f"| {k} | {f(d.get('0.25', {}).get('recall_moto'))} / {f(d.get('0.1', {}).get('recall_moto'))} | "
            f"{f(d.get('0.25', {}).get('recall_car'))} | {f(s.get('mAP50'))} | "
            f"{f(s.get('AP50', {}).get('motorcycle'))} / {f(s.get('AP50', {}).get('truck'))} | {f(v.get('mAP50'))} | "
            f"{f(x.get('india', {}).get('recall', {}).get('car'))} | {f(x.get('vncctv', {}).get('recall_all'))} | "
            f"{sum(g['giant_boxes'] for g in x.values()) if x else '–'} | {f(e.get('ext_map', {}).get('mAP50'))} |")
    (OUT / "results.md").write_text(
        "# So sánh các cách chống quên\n\nNgã tư: tỉ lệ bắt trúng xe yolov8x thấy (mốc thiên về COCO). "
        "SOICT, v10: nhãn người gán. Ấn Độ, camera VN: nhãn nháp yolov8x.\n\n" + "\n".join(rows) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rj = OUT / "results.json"
    res = json.loads(rj.read_text()) if rj.exists() else {}
    ims, ref = prepare_frames()
    for name in args.models:
        w = MODELS[name]
        if not (ROOT / w).exists():
            print(f"[SKIP] {name}: chưa có {w}", flush=True)
            continue
        is_coco = name.startswith("coco")
        e = {"weights": w, "deploy": deploy_recall(w, is_coco, ims, ref), "ext": ext_recall(w, is_coco)}
        if not is_coco:
            e["soict_test"] = map_val(w, "data/data_v6split.yaml", "test")
            e["vn_v10"] = map_val(w, "data/data_v6split_vn_v10.yaml", "test")
            e["ext_map"] = map_val(w, "data/data_ext_benchmark.yaml", "test")
        res[name] = e
        rj.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        write_md(res)
        d = e["deploy"]
        print(f"{name}: ngã tư xe máy {d['0.25']['recall_moto']}/{d['0.1']['recall_moto']} ô tô {d['0.25']['recall_car']}"
              f" | SOICT {e.get('soict_test', {}).get('mAP50')} | v10 {e.get('vn_v10', {}).get('mAP50')}", flush=True)


if __name__ == "__main__":
    main()
