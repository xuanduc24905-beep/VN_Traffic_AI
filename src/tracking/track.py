"""Người 2 — Wrapper tracking: YOLO + ByteTrack/BoT-SORT (built-in Ultralytics).

Ultralytics đã có sẵn 2 tracker: 'bytetrack.yaml' và 'botsort.yaml'.
Wrapper này trả về stream các frame kèm tracking ID để module counter dùng.
"""
from ultralytics import YOLO


def resolve_device(device: str | int | None = None) -> str | int:
    """Resolve ``None`` to CUDA, MPS, or CPU in that order."""
    if device is not None:
        return device
    import torch
    if torch.cuda.is_available():
        return 0
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def track_stream(weights: str | YOLO, source: str,
                 tracker: str = "bytetrack.yaml",
                 conf: float = 0.25, iou: float = 0.5,
                 imgsz: int = 640, device: str | int | None = None,
                 half: bool = False,
                 classes: list[int] | None = None):
    """Yield từng frame result có `.boxes.id` (tracking ID).

    result.boxes có các thuộc tính:
      - xyxy: (N, 4) — box toạ độ pixel
      - conf: (N,) — confidence
      - cls: (N,) — class id
      - id: (N,) — tracking id (có thể None với box vừa xuất hiện)

    half=True bật FP16 inference (~1.3-1.5× nhanh hơn trên GPU Ada/Ampere).
    """
    model = weights if isinstance(weights, YOLO) else YOLO(weights)
    kwargs = {
        "source": source,
        "tracker": tracker,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "half": half,
        "stream": True,
        "persist": True,
        "verbose": False,
    }
    selected_device = resolve_device(device)
    kwargs["device"] = selected_device
    if str(selected_device).lower() == "cpu":
        kwargs["half"] = False
    if classes is not None:
        kwargs["classes"] = classes
    return model.track(
        **kwargs,
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    choices=["bytetrack.yaml", "botsort.yaml"])
    ap.add_argument("--device", default="auto",
                    help="auto, cpu, mps, 0, 0,1, ...")
    args = ap.parse_args()

    device = None if args.device == "auto" else args.device
    for i, res in enumerate(track_stream(
            args.weights, args.source, args.tracker, device=device)):
        if res.boxes.id is None:
            continue
        n = len(res.boxes)
        print(f"frame {i:5d} n_track = {n} ids = {res.boxes.id.tolist()[:5]}...")
