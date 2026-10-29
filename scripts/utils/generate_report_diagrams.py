"""Sinh 2 ảnh minh họa cho báo cáo tiểu luận:
  1. docs/report/assets/diagrams/pipeline_vn.png
     - Pipeline tổng quan: Video → YOLOv8 → ByteTrack → Counter → Output
  2. docs/report/assets/yolo_format/yolo_coord_system_vn.png
     - Hệ toạ độ YOLO overlay trên ảnh V5 thực
"""
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path("/home/xuand/vn_traffic_ai")
OUT_DIAGRAMS = ROOT / "docs/report/assets/diagrams"
OUT_YOLO = ROOT / "docs/report/assets/yolo_format"
OUT_DIAGRAMS.mkdir(parents=True, exist_ok=True)
OUT_YOLO.mkdir(parents=True, exist_ok=True)

NAMES_V5 = {0: "motorcycle", 1: "car", 2: "bus", 3: "truck"}
# vn_cctv raw dùng schema khác: 0=bus, 1=car, 2=motorbike, 3=truck
NAMES_VNCCTV = {0: "bus", 1: "car", 2: "motorbike", 3: "truck"}
CLASS_COLORS_V5 = {0: "#06d6a0", 1: "#118ab2", 2: "#ffd166", 3: "#e63946"}
CLASS_COLORS_VNCCTV = {0: "#ffd166", 1: "#118ab2", 2: "#06d6a0", 3: "#e63946"}


# ============================================================
# 1. Pipeline diagram
# ============================================================
def make_pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    boxes = [
        (5.0, 13.0, "Video đầu vào", "", "#ffffff", "#000000"),
        (5.0, 10.5, "YOLOv8s Detect", "Bounding box, class, confidence", "#eeeeee", "#000000"),
        (5.0, 8.0, "ByteTrack", "Gán track ID bền vững qua các frame", "#eeeeee", "#000000"),
        (5.0, 5.5, "Counter", "Phát hiện giao cắt vạch ảo + hướng di chuyển", "#eeeeee", "#000000"),
        (2.5, 2.0, "CSV sự kiện", "", "#ffffff", "#000000"),
        (7.5, 2.0, "Video chú thích", "", "#ffffff", "#000000"),
    ]

    def draw_box(cx, cy, title, subtitle, fill, edge, w=4.2, h=1.5):
        bb = FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w, h,
            boxstyle="round,pad=0.1,rounding_size=0.15",
            linewidth=2.5, facecolor=fill, edgecolor=edge,
        )
        ax.add_patch(bb)
        if subtitle:
            ax.text(cx, cy + 0.25, title, ha="center", va="center",
                    fontsize=14, fontweight="bold", family="DejaVu Sans")
            ax.text(cx, cy - 0.3, subtitle, ha="center", va="center",
                    fontsize=10, style="italic", family="DejaVu Sans")
        else:
            ax.text(cx, cy, title, ha="center", va="center",
                    fontsize=14, fontweight="bold", family="DejaVu Sans")

    for cx, cy, t, s, f, e in boxes:
        if "CSV" in t or "chú thích" in t:
            draw_box(cx, cy, t, s, f, e, w=3.5, h=1.2)
        else:
            draw_box(cx, cy, t, s, f, e)

    arrows = [
        ((5, 12.25), (5, 11.25)),
        ((5, 9.75), (5, 8.75)),
        ((5, 7.25), (5, 6.25)),
        ((5, 4.75), (2.5, 2.6)),
        ((5, 4.75), (7.5, 2.6)),
    ]
    for start, end in arrows:
        arr = FancyArrowPatch(
            start, end,
            arrowstyle="-|>",
            mutation_scale=25,
            linewidth=2.5,
            color="#000000",
        )
        ax.add_patch(arr)

    plt.tight_layout()
    out = OUT_DIAGRAMS / "pipeline_vn.png"
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[OK] {out}")


# ============================================================
# 2. YOLO coord system diagram với ảnh VN thực
# ============================================================
def make_coord_system_diagram() -> None:
    # Dùng ảnh SOICT ngày (cam_06): container truck đỏ to rõ, scene đường quốc lộ
    img_name = "soict24_cam_06_00631_jpg.rf.b55fc146d340ae871c080228a16bb906"
    img_path = ROOT / f"data/merged_v5/images/train/{img_name}.jpg"
    lbl_path = ROOT / f"data/merged_v5/labels/train/{img_name}.txt"
    # merged_v5 dùng V5 schema
    NAMES = NAMES_V5
    CLASS_COLORS = CLASS_COLORS_V5

    assert img_path.exists(), f"Thiếu {img_path}"
    assert lbl_path.exists(), f"Thiếu {lbl_path}"

    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    boxes = []
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append((cls_id, cx, cy, w, h))

    # Sort by size descending — pick largest to annotate in detail
    boxes.sort(key=lambda b: -b[3] * b[4])

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(img)
    ax.set_xlim(-W * 0.08, W * 1.08)
    ax.set_ylim(H * 1.08, -H * 0.08)  # inverted for YOLO coord (y down)

    # Axes labels (origin, +X, +Y)
    ax.annotate("Origin (0,0)",
                xy=(0, 0), xytext=(-W * 0.03, -H * 0.03),
                fontsize=14, fontweight="bold", color="#333")
    # X axis arrow
    ax.annotate("", xy=(W * 1.05, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#333"))
    ax.text(W * 1.06, -H * 0.02, "+X (cx)", fontsize=14, fontweight="bold", color="#333")
    # Y axis arrow
    ax.annotate("", xy=(0, H * 1.05), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#333"))
    ax.text(-W * 0.03, H * 1.07, "+Y (cy)", fontsize=14, fontweight="bold", color="#333")

    # Corners labels
    for (nx, ny), text in [
        ((1, 0), "(1, 0)"),
        ((0, 1), "(0, 1)"),
        ((1, 1), "(1, 1)"),
    ]:
        ax.text(nx * W + (10 if nx < 1 else -10),
                ny * H + (10 if ny < 1 else -10),
                text, ha="right" if nx == 1 else "left",
                va="bottom" if ny == 1 else "top",
                fontsize=11, color="#333",
                bbox=dict(facecolor="white", edgecolor="none", pad=1))

    # Draw bboxes
    for i, (cls_id, cx, cy, w, h) in enumerate(boxes):
        px = cx * W
        py = cy * H
        pw = w * W
        ph = h * H
        color = CLASS_COLORS.get(cls_id, "#000000")

        # Highlight first (largest) box with full annotation; others just outline
        is_main = (i == 0)
        alpha = 0.35 if is_main else 0.2

        rect = plt.Rectangle(
            (px - pw / 2, py - ph / 2), pw, ph,
            linewidth=3 if is_main else 2,
            edgecolor=color,
            facecolor=color,
            alpha=alpha,
        )
        ax.add_patch(rect)

        if is_main:
            # Center dot
            ax.plot(px, py, "o", color="red", markersize=10, zorder=5)
            ax.annotate(
                f"({cx:.2f}, {cy:.2f})",
                xy=(px, py), xytext=(px + 25, py + 25),
                fontsize=12, color="red", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="red", pad=3),
                arrowprops=dict(arrowstyle="->", color="red"),
            )
            # Width arrow (horizontal, below center)
            arr_y = py + ph / 2 + 20
            ax.annotate("", xy=(px + pw / 2, arr_y), xytext=(px - pw / 2, arr_y),
                        arrowprops=dict(arrowstyle="<->", lw=2, color="orange"))
            ax.text(px, arr_y + 15, f"w = {w:.2f}", ha="center",
                    fontsize=12, fontweight="bold", color="orange",
                    bbox=dict(facecolor="white", edgecolor="orange", pad=2))
            # Height arrow (vertical, right of center)
            arr_x = px + pw / 2 + 20
            ax.annotate("", xy=(arr_x, py + ph / 2), xytext=(arr_x, py - ph / 2),
                        arrowprops=dict(arrowstyle="<->", lw=2, color="orange"))
            ax.text(arr_x + 15, py, f"h = {h:.2f}", va="center",
                    fontsize=12, fontweight="bold", color="orange",
                    bbox=dict(facecolor="white", edgecolor="orange", pad=2),
                    rotation=90)
            # Class label
            ax.text(px - pw / 2, py - ph / 2 - 10,
                    f"class={cls_id} ({NAMES[cls_id]})",
                    fontsize=11, color="white", fontweight="bold",
                    bbox=dict(facecolor=color, edgecolor="none", pad=3))

    ax.set_title(
        "Hệ toạ độ YOLO — ảnh thực từ bộ SOICT 2024\n"
        f"Origin góc trên trái, +X sang phải, +Y xuống dưới; "
        f"(cx, cy, w, h) chuẩn hoá theo kích thước ảnh gốc {W}×{H} px",
        fontsize=13, pad=15,
    )
    ax.axis("off")

    plt.tight_layout()
    out = OUT_YOLO / "yolo_coord_system_vn.png"
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[OK] {out}")


if __name__ == "__main__":
    print("[1/2] Generate pipeline diagram...")
    make_pipeline_diagram()
    print("[2/2] Generate YOLO coord system diagram with VN image...")
    make_coord_system_diagram()
    print("\nDone.")
