"""Xuat video so sanh 3 mo hinh canh nhau (COCO yolov8s | D1 freeze | C khong freeze) tren clip that.
Chay: python scripts/evaluation/render_compare_videos.py
Khong tracking — chi ve hop (conf 0.25, imgsz 960, agnostic NMS) de xem bat xe, khong de dem.
"""
import subprocess
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
REC = Path("/mnt/c/Users/xuand/Videos/Screen Recordings")
CLIPS = {"india_0920": (REC / "Screen Recording 2026-09-20 194658.mp4", 0, 400),
         "vn_truongchinh_1105": (REC / "Screen Recording 2026-11-05 033122.mp4", 0, 450),
         "vn_1108": (REC / "Screen Recording 2026-11-08 042445.mp4", 0, 400)}
MODELS = [("COCO yolov8s", "yolov8s.pt"), ("D1 freeze=10", "runs/detect/v5fix_d1_freeze10/weights/best.pt"),
          ("C no-freeze", "runs/detect/v5fix_c_nofreeze/weights/best.pt")]
COCO = {1: 0, 2: 1, 3: 0, 5: 2, 7: 3}  # bicycle->(bo), car, motorcycle, bus, truck
COL = [(0, 255, 0), (255, 160, 0), (0, 0, 255), (255, 0, 255)]
PW = 640

def main():
    models = [(n, YOLO(str(ROOT / w))) for n, w in MODELS]
    for name, (path, start, n) in CLIPS.items():
        cap = cv2.VideoCapture(str(path)); fps = cap.get(5) or 30
        cap.set(1, start)
        W, H = int(cap.get(3)), int(cap.get(4)); ph = int(PW * H / W)
        raw = ROOT / "reports/v5fix/videos" / f"{name}_raw.mp4"
        out = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (PW * 3, ph))
        for _ in range(n):
            ok, f = cap.read()
            if not ok: break
            panels = []
            for (label, m) in models:
                coco = label.startswith("COCO")
                r = m.predict(f, imgsz=960, conf=0.25, iou=0.5, agnostic_nms=True, half=True, verbose=False,
                              classes=[2, 3, 5, 7] if coco else None)[0]
                im = f.copy(); cnt = 0
                for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
                    c = COCO[c] if coco else c
                    cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), COL[c], max(2, W // 640)); cnt += 1
                p = cv2.resize(im, (PW, ph))
                cv2.putText(p, f"{label}: {cnt}", (8, 30), 0, 0.9, (0, 255, 255), 2)
                panels.append(p)
            out.write(np.hstack(panels))
        out.release()
        h264 = raw.with_name(f"{name}.mp4")
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(h264)])
        if r.returncode == 0: raw.unlink()
        print(name, "->", h264 if r.returncode == 0 else raw, flush=True)

if __name__ == "__main__":
    main()
