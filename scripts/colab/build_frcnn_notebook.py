#!/usr/bin/env python3
"""Sinh notebook Colab train lai Faster R-CNN tren V5+ (cung tap huan luyen voi YOLOv8s).

Cong thuc giong notebooks/ffrcnn.ipynb (ban cu train tren V6), chi doi:
  - DATA = V5+ (7.237 anh train) thay vi V6
  - EPOCHS = 8 (ban cu: diem luu tot nhat o epoch thu 2 / 30)
  - danh gia TOAN BO tap xac thuc moi epoch (ban cu chi 50 batch dau = 200 anh Can Tho)
  - co dinh hat giong 42, luu thoi gian moi epoch, tu resume tu Google Drive khi Colab ngat
Chay:  python scripts/colab/build_frcnn_notebook.py   -> notebooks/colab_frcnn_v5plus.ipynb
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks/colab_frcnn_v5plus.ipynb"

md0 = """# Train lại Faster R-CNN ResNet50-FPN trên V5+ (Colab)

Mục đích: Faster R-CNN dùng **đúng tập huấn luyện V5+** như YOLOv8s (bản cũ train trên V6 có nhãn giả).

**Chuẩn bị (trên máy):** `bash scripts/colab/pack_v5plus_for_colab.sh`, rồi upload 2 file
`data/colab/v5plus_frcnn.tar` và `data/colab/v5plus_frcnn.manifest.json` vào Google Drive, thư mục `MyDrive/vn_traffic_frcnn/`.

**Chạy (trên Colab):** Runtime → Change runtime type → GPU. Chạy ô 1 (cài môi trường, Colab sẽ tự khởi động lại),
sau đó chạy lần lượt từ ô 2 đến hết. Nếu Colab ngắt giữa chừng: chạy lại ô 1 → ô 2 → ... notebook tự train tiếp từ `last.pt` trên Drive.

**Kết quả** nằm ở `MyDrive/vn_traffic_frcnn/frcnn_v5plus/`: `best.pt`, `last.pt`, `train_log.csv`, `env.json`, `config.json`.
Tải cả thư mục về máy, đặt vào `weights/trained/frcnn_v5plus/`, rồi chạy `python scripts/colab/verify_frcnn_v5plus.py`."""

c1 = """# Ô 1 — Đồng nhất môi trường với máy local (vehicle_env): torch 2.7.1+cu118, torchvision 0.22.1+cu118
import subprocess, sys, os
need = {"torch": "2.7.1+cu118", "torchvision": "0.22.1+cu118"}
try:
    import torch, torchvision
    ok = torch.__version__ == need["torch"] and torchvision.__version__ == need["torchvision"]
except Exception:
    ok = False
if not ok:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch==2.7.1", "torchvision==0.22.1",
                           "--index-url", "https://download.pytorch.org/whl/cu118"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==2.2.6", "torchmetrics==1.9.0",
                           "pycocotools", "opencv-python-headless"])
    print("Đã cài xong, khởi động lại runtime... sau đó chạy tiếp từ Ô 2.")
    os.kill(os.getpid(), 9)
else:
    print("Môi trường đã đúng:", torch.__version__, torchvision.__version__)"""

c2 = """# Ô 2 — Kiểm tra phiên bản + GPU, ghi env.json
import os, sys, json, platform
import torch, torchvision, numpy, cv2, torchmetrics
SMOKE = os.environ.get("FRCNN_SMOKE") == "1"   # chỉ dùng khi chạy thử trên máy local
LOCAL = os.environ.get("FRCNN_LOCAL") == "1"   # train tiếp trên máy local (scripts/colab/resume_frcnn_local.sh)
if not SMOKE:
    assert torch.__version__ == "2.7.1+cu118", torch.__version__
    assert torchvision.__version__ == "0.22.1+cu118", torchvision.__version__
    assert torch.cuda.is_available(), "Chưa bật GPU: Runtime -> Change runtime type -> GPU"
ENV = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
       "numpy": numpy.__version__, "opencv": cv2.__version__, "torchmetrics": torchmetrics.__version__,
       "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
print(json.dumps(ENV, indent=1))"""

c3 = """# Ô 3 — Lấy dữ liệu từ Google Drive, kiểm tra sha256, giải nén vào /content
import hashlib, tarfile, shutil
from pathlib import Path
if SMOKE:
    DATA = Path(os.environ["FRCNN_SMOKE_DATA"]); OUT_DIR = Path(os.environ["FRCNN_SMOKE_OUT"])
elif LOCAL:
    DATA = Path(os.environ["FRCNN_LOCAL_DATA"]); OUT_DIR = Path(os.environ["FRCNN_LOCAL_OUT"])
    assert (OUT_DIR / "last.pt").exists(), f"chưa có {OUT_DIR / 'last.pt'} tải từ Colab"
else:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE = Path("/content/drive/MyDrive/vn_traffic_frcnn")
    OUT_DIR = DRIVE / "frcnn_v5plus"
    DATA = Path("/content/v5plus")
    if not (DATA / "images/train").exists():
        man = json.loads((DRIVE / "v5plus_frcnn.manifest.json").read_text())
        local_tar = Path("/content/v5plus_frcnn.tar")
        shutil.copy(DRIVE / "v5plus_frcnn.tar", local_tar)
        h = hashlib.sha256()
        with open(local_tar, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        assert h.hexdigest() == man["tar_sha256"], "sha256 không khớp, upload lại file tar"
        with tarfile.open(local_tar) as t:
            t.extractall("/content")
        for s in ("train", "val", "test"):
            n = len(list((DATA / "images" / s).glob("*.jpg")))
            assert n == man[s]["images"], (s, n, man[s])
        print("Dữ liệu OK:", {s: man[s] for s in ("train", "val", "test")})
OUT_DIR.mkdir(parents=True, exist_ok=True)
# env.json luôn là môi trường Colab; khi train tiếp trên máy thì ghi env_local.json
(OUT_DIR / ("env_local.json" if LOCAL else "env.json")).write_text(json.dumps(ENV, indent=1))"""

c4 = """# Ô 4 — Cấu hình (giữ công thức của notebooks/ffrcnn.ipynb, chỉ đổi dữ liệu, số epoch, cách đánh giá)
import time, csv, random
import numpy as np
from torch.utils.data import Dataset, DataLoader

CFG = dict(
    data="V5+ (merged_v5plus)", epochs=8, batch=4, accum_steps=2, imgsz_min=800, imgsz_max=1333,
    lr=0.005, momentum=0.9, nesterov=True, weight_decay=5e-4, warmup_epochs=2, schedule="linear warmup + cosine",
    grad_clip=10.0, amp=True, seed=42, eval="toàn bộ tập xác thực mỗi epoch, chọn best theo mAP@0,5 (torchmetrics)",
    box_score_thresh_train=0.05, box_detections_per_img=100, box_nms_thresh=0.5, init="COCO (weights=DEFAULT)",
    augment="lật ngang 0,5; độ sáng/tương phản 0,5; hue 0,3")
if SMOKE:
    CFG.update(epochs=1, imgsz_min=320, imgsz_max=544, batch=2, accum_steps=1)
CLASSES = ["motorcycle", "car", "bus", "truck"]
NC = len(CLASSES)
dev = "cuda" if torch.cuda.is_available() and not SMOKE else "cpu"

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(CFG["seed"])
if not (LOCAL and (OUT_DIR / "config.json").exists()):
    (OUT_DIR / "config.json").write_text(json.dumps(CFG, indent=1, ensure_ascii=False))
else:  # train tiếp: cấu hình phải trùng bản đã chạy trên Colab
    _old = json.loads((OUT_DIR / "config.json").read_text())
    assert _old == CFG, ("config khác bản Colab", _old, CFG)
print(CFG)"""

c5 = """# Ô 5 — Dataset (giống notebook cũ: bỏ ảnh không có nhãn, nhãn YOLO c -> c+1)
class YoloDet(Dataset):
    def __init__(self, img_dir, lbl_dir, train=False, limit=None):
        self.lbl_dir, self.train = Path(lbl_dir), train
        self.imgs = []
        for p in sorted(Path(img_dir).glob("*.jpg")) + sorted(Path(img_dir).glob("*.png")):
            lbl = self.lbl_dir / f"{p.stem}.txt"
            if lbl.exists() and lbl.stat().st_size > 0:
                self.imgs.append(p)
        if limit:
            self.imgs = self.imgs[:limit]
        print(f"  {Path(img_dir).name}: {len(self.imgs)} ảnh có nhãn")

    def __len__(self):
        return len(self.imgs)

    def _augment(self, img):
        if torch.rand(1) < 0.5:
            alpha = 0.8 + 0.4 * torch.rand(1).item()
            beta = int(-20 + 40 * torch.rand(1).item())
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if torch.rand(1) < 0.3:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int32)
            hsv[..., 0] = (hsv[..., 0] + int(-10 + 20 * torch.rand(1).item())) % 180
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return img

    def __getitem__(self, i):
        p = self.imgs[i]
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        boxes, labels = [], []
        for line in (self.lbl_dir / f"{p.stem}.txt").read_text().splitlines():
            if not line.strip():
                continue
            c, cx, cy, bw, bh = map(float, line.split()[:5])
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            labels.append(int(c) + 1)
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.int64)
        if self.train:
            img = self._augment(img)
            if torch.rand(1) < 0.5:
                img = img[:, ::-1].copy()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h - 1)
        keep = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255
        return x, {"boxes": boxes[keep], "labels": labels[keep]}

def collate(b):
    valid = [(x, t) for x, t in b if len(t["boxes"]) > 0]
    return tuple(zip(*valid)) if valid else collate(b)

lim = 8 if SMOKE else None
ds_tr = YoloDet(DATA / "images/train", DATA / "labels/train", train=True, limit=lim)
ds_va = YoloDet(DATA / "images/val", DATA / "labels/val", limit=lim)
g = torch.Generator(); g.manual_seed(CFG["seed"])
nw = 0 if SMOKE else 2
dl_tr = DataLoader(ds_tr, CFG["batch"], shuffle=True, num_workers=nw, collate_fn=collate, pin_memory=True,
                   drop_last=True, generator=g)
dl_va = DataLoader(ds_va, CFG["batch"], shuffle=False, num_workers=nw, collate_fn=collate, pin_memory=True)
print(f"Batches: train={len(dl_tr)} val={len(dl_va)}")"""

c6 = """# Ô 6 — Mô hình + optimizer + lịch học (giống notebook cũ)
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection import MeanAveragePrecision

model = fasterrcnn_resnet50_fpn(weights="DEFAULT", min_size=CFG["imgsz_min"], max_size=CFG["imgsz_max"],
                                box_score_thresh=CFG["box_score_thresh_train"],
                                box_detections_per_img=CFG["box_detections_per_img"],
                                box_nms_thresh=CFG["box_nms_thresh"],
                                rpn_pre_nms_top_n_train=2000, rpn_post_nms_top_n_train=1000)
model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, NC + 1)
model.to(dev)

decay, no_decay = [], []
for n, p in model.named_parameters():
    if p.requires_grad:
        (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
opt = torch.optim.SGD([{"params": decay, "weight_decay": CFG["weight_decay"]},
                       {"params": no_decay, "weight_decay": 0.0}],
                      lr=CFG["lr"], momentum=CFG["momentum"], nesterov=CFG["nesterov"])
steps_per_ep = max(1, len(dl_tr) // CFG["accum_steps"])
total_steps, warmup_steps = steps_per_ep * CFG["epochs"], steps_per_ep * CFG["warmup_epochs"]
def lr_lambda(step):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + np.cos(np.pi * prog))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
scaler = torch.amp.GradScaler(enabled=(dev == "cuda"))

last_pt, best_pt, log_csv = OUT_DIR / "last.pt", OUT_DIR / "best.pt", OUT_DIR / "train_log.csv"
start, best_map = 0, -1.0
if last_pt.exists():
    ck = torch.load(last_pt, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    scaler.load_state_dict(ck["scaler"]); torch.set_rng_state(ck["rng_cpu"])
    start, best_map = ck["epoch"] + 1, ck["best_map"]
    print(f"[resume] train tiếp từ epoch {start + 1}, best mAP50 = {best_map:.4f}")
    with open(OUT_DIR / "resume_history.txt", "a") as f:  # ghi lại epoch nào train trên GPU nào
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} resume từ epoch {start + 1} trên {ENV['gpu']} "
                f"(torch {ENV['torch']}, python {ENV['python']})\n")"""

c7 = """# Ô 7 — Train + đánh giá toàn bộ tập xác thực mỗi epoch
@torch.no_grad()
def eval_map(loader):
    model.eval()
    m = MeanAveragePrecision(iou_type="bbox")
    for xs, ts in loader:
        out = model([x.to(dev) for x in xs])
        m.update([{k: o[k].cpu() for k in ("boxes", "scores", "labels")} for o in out],
                 [{"boxes": t["boxes"], "labels": t["labels"]} for t in ts])
    r = m.compute()
    return float(r["map_50"]), float(r["map"])

if not log_csv.exists():
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_mAP50", "val_mAP", "lr", "train_sec", "eval_sec"])

for ep in range(start, CFG["epochs"]):
    model.train(); t0 = time.time(); tot, nb = 0.0, 0
    opt.zero_grad(set_to_none=True)
    for i, (xs, ts) in enumerate(dl_tr):
        xs = [x.to(dev, non_blocking=True) for x in xs]
        ts = [{k: v.to(dev, non_blocking=True) for k, v in t.items()} for t in ts]
        with torch.autocast("cuda", enabled=(dev == "cuda")):
            loss = sum(model(xs, ts).values()) / CFG["accum_steps"]
        scaler.scale(loss).backward()
        if (i + 1) % CFG["accum_steps"] == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        tot += loss.item() * CFG["accum_steps"]; nb += 1
        if i % 100 == 0:
            print(f"  ep{ep + 1} [{i + 1}/{len(dl_tr)}] loss={loss.item() * CFG['accum_steps']:.3f} "
                  f"lr={opt.param_groups[0]['lr']:.5f} {time.time() - t0:.0f}s")
    t_train = time.time() - t0
    t1 = time.time(); m50, m = eval_map(dl_va); t_eval = time.time() - t1
    if m50 > best_map:
        best_map = m50
        torch.save({"model": model.state_dict(), "epoch": ep, "val_map50": m50, "val_map": m,
                    "cfg": CFG, "env": ENV}, best_pt)
        print(f"  * best mới: val mAP50={m50:.4f}")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "rng_cpu": torch.get_rng_state(), "epoch": ep,
                "best_map": best_map, "cfg": CFG, "env": ENV}, last_pt)
    with open(log_csv, "a", newline="") as f:
        csv.writer(f).writerow([ep + 1, tot / max(nb, 1), m50, m, opt.param_groups[0]["lr"],
                                round(t_train, 1), round(t_eval, 1)])
    print(f"EP {ep + 1}/{CFG['epochs']} loss={tot / max(nb, 1):.4f} val mAP50={m50:.4f} mAP={m:.4f} "
          f"train {t_train / 60:.1f} phút, eval {t_eval / 60:.1f} phút")
print("Xong. best val mAP50 =", round(best_map, 4), "| file ở", OUT_DIR)"""

cells = [("markdown", md0), ("code", c1), ("code", c2), ("code", c3), ("code", c4), ("code", c5), ("code", c6),
         ("code", c7)]
nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [{"cell_type": t, "metadata": {}, "source": s, **({"outputs": [], "execution_count": None} if t == "code" else {})}
                for t, s in cells]}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("saved", OUT)
