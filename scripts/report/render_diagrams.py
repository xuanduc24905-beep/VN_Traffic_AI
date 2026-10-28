#!/usr/bin/env python3
"""Render pipeline diagrams + YOLO format visuals as PNG assets.

All outputs land in docs/report/assets/{diagrams,yolo_format}/ so the MD→DOCX
converter can embed them directly.
"""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import matplotlib.image as mpimg

ROOT = Path("/home/xuand/vn_traffic_ai")
OUT_DIAG = ROOT / "docs/report/assets/diagrams"
OUT_YOLO = ROOT / "docs/report/assets/yolo_format"
OUT_DIAG.mkdir(parents=True, exist_ok=True)
OUT_YOLO.mkdir(parents=True, exist_ok=True)

# Palette
C_PRIMARY = "#1F3A68"
C_SECONDARY = "#2E5B9C"
C_ACCENT = "#F39C12"
C_SUCCESS = "#27AE60"
C_DANGER = "#C0392B"
C_LIGHT = "#ECF0F1"
C_GREY = "#7F8C8D"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"


def box(ax, x, y, w, h, text, color=C_PRIMARY, fontcolor="white", fontsize=10, bold=True):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4, edgecolor=color, facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            color=fontcolor, fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            wrap=True)


def arrow(ax, x1, y1, x2, y2, color="#333", lw=1.8, label=None):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=15,
                        color=color, lw=lw)
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.1, label,
                ha="center", va="bottom", fontsize=8.5, color=color,
                style="italic")


def setup_ax(ax, xlim=(0, 10), ylim=(0, 7), title=None):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", color=C_PRIMARY, pad=12)


# ---------------- Diagram 1: Workflow tổng quan ----------------

def diag_workflow():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    setup_ax(ax, xlim=(0, 14), ylim=(0, 5),
             title="Workflow tổng quan: từ COCO pretrained đến model phục vụ VN traffic")

    # Row 1: data sources
    box(ax, 0.3, 3.2, 2.3, 1.2, "COCO\n80 class\n~118k images", color=C_GREY, fontsize=9)
    box(ax, 0.3, 1.3, 2.3, 1.2, "VN Traffic data\n5 nguồn, ~52k ảnh\nSOICT • Cần Thơ • Đà Nẵng", color=C_SECONDARY, fontsize=9)

    # Middle: processing
    box(ax, 3.4, 3.2, 2.8, 1.2, "YOLOv8s\npretrained COCO\n(backbone + head)", color=C_PRIMARY, fontsize=9)
    box(ax, 3.4, 1.3, 2.8, 1.2, "Audit + rebuild\nvideo-aware split\n(V5 honest)", color=C_ACCENT, fontsize=9)

    box(ax, 7.1, 2.2, 2.8, 1.3, "Finetune\nyolov8s @ 640/960\nSGD 40 epochs", color=C_PRIMARY, fontsize=9)

    box(ax, 10.6, 2.2, 3.0, 1.3, "Model VN\nmAP50 = 0.782\n(V5+ WINNER)", color=C_SUCCESS, fontsize=9)

    # Arrows
    arrow(ax, 2.65, 3.8, 3.35, 3.8)
    arrow(ax, 2.65, 1.9, 3.35, 1.9)
    arrow(ax, 6.25, 3.8, 7.1, 3.1, label="weights")
    arrow(ax, 6.25, 1.9, 7.1, 2.6, label="data")
    arrow(ax, 9.95, 2.85, 10.6, 2.85)

    # Legend / caption
    ax.text(7, 0.3, "Trọng tâm dự án: finetune + fix data leakage",
            ha="center", fontsize=10, style="italic", color=C_GREY)

    plt.tight_layout()
    out = OUT_DIAG / "pipeline_workflow.png"
    plt.savefig(out, dpi=170, bbox_inches="tight")
    plt.close()
    print("  ✓", out.name)


# ---------------- Diagram 2: Leakage mechanism ----------------

def diag_leak_mechanism():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax in (ax1, ax2):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.set_aspect("equal")
        ax.axis("off")

    ax1.set_title("[SAI] Random split theo frame\n(gây video-level leakage)",
                  fontsize=12, fontweight="bold", color=C_DANGER, pad=10)
    ax2.set_title("[DUNG] Video-aware split\n(mỗi video chỉ 1 split)",
                  fontsize=12, fontweight="bold", color=C_SUCCESS, pad=10)

    # Video strip helper
    def draw_video(ax, y, name, frames_colors, label_right=None):
        ax.text(0.2, y + 0.4, name, fontsize=9.5, fontweight="bold", color=C_PRIMARY)
        for i, c in enumerate(frames_colors):
            ax.add_patch(Rectangle((1.6 + i * 0.6, y), 0.55, 0.65, facecolor=c, edgecolor="#333", lw=0.5))
        if label_right:
            ax.text(1.6 + len(frames_colors) * 0.6 + 0.2, y + 0.3, label_right,
                    fontsize=9, color=C_GREY, va="center")

    # LEFT panel — leaky
    # Legend: train=blue, val=orange, test=red
    train_col = "#3498DB"
    val_col = "#F39C12"
    test_col = "#E74C3C"
    for ax in (ax1, ax2):
        ax.text(5, 6.6, "Video A (30 frames liên tiếp)", ha="center", fontsize=10, color=C_GREY)

    # Leaky: train/val/test randomly from same video
    import random as _r
    _r.seed(1)
    cols_leaky = [_r.choice([train_col, train_col, train_col, val_col, test_col]) for _ in range(12)]
    draw_video(ax1, 5.3, "", cols_leaky, label_right="")
    # Repeat for 3 videos
    cols_leaky2 = [_r.choice([train_col, train_col, train_col, val_col, test_col]) for _ in range(12)]
    draw_video(ax1, 4.3, "", cols_leaky2)
    cols_leaky3 = [_r.choice([train_col, train_col, train_col, val_col, test_col]) for _ in range(12)]
    draw_video(ax1, 3.3, "", cols_leaky3)

    ax1.text(5, 2.5, "➜ Frame 100 (train) và frame 101 (val)\ngần như giống hệt — model memorize scene",
             ha="center", fontsize=10, color=C_DANGER, style="italic", fontweight="bold")
    ax1.text(5, 1.3, "mAP ẢO = 0.954", ha="center", fontsize=16, color=C_DANGER, fontweight="bold")

    # Right: Honest - each video assigned to 1 split
    draw_video(ax2, 5.3, "", [train_col] * 12, label_right="→ TRAIN")
    draw_video(ax2, 4.3, "", [val_col] * 12, label_right="→ VAL")
    draw_video(ax2, 3.3, "", [test_col] * 12, label_right="→ TEST")

    ax2.text(5, 2.5, "➜ Model train trên video A, test\ntrên video khác → đo generalization thật",
             ha="center", fontsize=10, color=C_SUCCESS, style="italic", fontweight="bold")
    ax2.text(5, 1.3, "mAP THẬT = 0.665", ha="center", fontsize=16, color=C_SUCCESS, fontweight="bold")

    # Legend
    legend_handles = [
        mpatches.Patch(color=train_col, label="train frame"),
        mpatches.Patch(color=val_col, label="val frame"),
        mpatches.Patch(color=test_col, label="test frame"),
    ]
    ax1.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    out = OUT_DIAG / "pipeline_leak_mechanism.png"
    plt.savefig(out, dpi=170, bbox_inches="tight")
    plt.close()
    print("  ✓", out.name)


# ---------------- Diagram 3: Rebuild V5 pipeline ----------------

def diag_rebuild_pipeline():
    fig, ax = plt.subplots(figsize=(13, 6))
    setup_ax(ax, xlim=(0, 14), ylim=(0, 8),
             title="Pipeline rebuild V5 — video-aware split với audit tự động")

    # Step 1
    box(ax, 0.3, 5.5, 2.6, 1.4, "① Pool tất cả\nframes từ merged_v3\n(ignore split cũ)", color=C_SECONDARY, fontsize=9)

    # Step 2
    box(ax, 3.5, 5.5, 2.6, 1.4, "② Parse video_id\nbằng regex từ\nfilename", color=C_SECONDARY, fontsize=9)

    # Step 3
    box(ax, 6.7, 5.5, 2.6, 1.4, "③ Group frames\ntheo video_id\n(~18 video)", color=C_SECONDARY, fontsize=9)

    # Step 4
    box(ax, 9.9, 5.5, 3.6, 1.4, "④ Apply SPLIT_PLAN\n(10 train + 3 val + 2 test video)\n+ stride subsample", color=C_PRIMARY, fontsize=9)

    # Arrows row 1
    for x1, x2 in [(2.9, 3.5), (6.1, 6.7), (9.3, 9.9)]:
        arrow(ax, x1, 6.2, x2, 6.2)

    # Vertical arrow
    arrow(ax, 11.7, 5.5, 11.7, 4.3)

    # Step 5
    box(ax, 9.9, 2.9, 3.6, 1.4, "⑤ Symlink sang\ndata/merged_v5/\n(tiết kiệm disk)", color=C_PRIMARY, fontsize=9)

    arrow(ax, 11.7, 2.9, 11.7, 1.7)

    # Step 6
    box(ax, 9.9, 0.3, 3.6, 1.4, "⑥ Save\nsplit_manifest.json\n(reproducibility)", color=C_ACCENT, fontsize=9)

    # Audit check loop
    box(ax, 5, 2.9, 4.0, 1.4, "⑦ audit_leak.py:\nverify 0% leak\n→ fail hard nếu detect", color=C_DANGER, fontsize=9)

    arrow(ax, 9.9, 3.6, 9.0, 3.6, color=C_DANGER)
    arrow(ax, 5, 3.0, 5, 0.9, color=C_DANGER)
    box(ax, 2.3, 0.3, 2.5, 1.4, "Nếu leak > 0\n→ dừng pipeline", color=C_DANGER, fontsize=9)
    arrow(ax, 4.8, 1.0, 5, 1.0, color=C_DANGER)

    # Output label
    box(ax, 0.3, 2.9, 4.0, 1.4,
        "OUTPUT V5\ntrain=2,833 • val=1,466 • test=1,500\nleak=0% ✓",
        color=C_SUCCESS, fontsize=9)

    plt.tight_layout()
    out = OUT_DIAG / "pipeline_rebuild_v5.png"
    plt.savefig(out, dpi=170, bbox_inches="tight")
    plt.close()
    print("  ✓", out.name)


# ---------------- Diagram 4: Training pipeline ----------------

def diag_training_pipeline():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    setup_ax(ax, xlim=(0, 14), ylim=(0, 6.5),
             title="Pipeline training V5+ (yolov8s @ 960)")

    # Inputs
    box(ax, 0.3, 4.2, 2.5, 1.4, "merged_v5/\ntrain + val\n(video-aware)", color=C_SECONDARY, fontsize=9)
    box(ax, 0.3, 2.0, 2.5, 1.4, "yolov8s.pt\n(COCO pretrained)\n80 class", color=C_GREY, fontsize=9)
    box(ax, 0.3, -0.1, 2.5, 1.4, "data_v5.yaml\nnc=4\n{moto,car,bus,truck}", color=C_SECONDARY, fontsize=9)

    # Trainer core
    box(ax, 3.6, 2.0, 3.6, 3.5,
        "Ultralytics YOLO.train()\n\n• imgsz=960\n• batch=16  epochs=40\n• SGD + cos_lr\n• mixup=0.1  copy_paste=0.3\n• patience=15",
        color=C_PRIMARY, fontsize=9.5)

    # Arrows into trainer
    for y1 in (4.9, 2.7, 0.6):
        arrow(ax, 2.8, y1, 3.6, 3.75)

    # Outputs
    box(ax, 8.0, 4.2, 2.7, 1.4, "best.pt\n(checkpoint)", color=C_SUCCESS, fontsize=9)
    box(ax, 8.0, 2.3, 2.7, 1.4, "results.csv\n(train curves)", color=C_SUCCESS, fontsize=9)
    box(ax, 8.0, 0.4, 2.7, 1.4, "confusion_matrix\n+ PR/F1 curves", color=C_SUCCESS, fontsize=9)

    for y in (4.9, 3.0, 1.1):
        arrow(ax, 7.2, 3.75, 8.0, y)

    # Eval downstream
    box(ax, 11.1, 2.3, 2.7, 1.4,
        "Cross-eval\nV5 test + OOD\n(vn_cctv_test)", color=C_ACCENT, fontsize=9)
    arrow(ax, 10.7, 3.0, 11.1, 3.0)

    plt.tight_layout()
    out = OUT_DIAG / "pipeline_training_v5plus.png"
    plt.savefig(out, dpi=170, bbox_inches="tight")
    plt.close()
    print("  ✓", out.name)


# ---------------- Diagram 5: V7 pipeline ----------------

def diag_v7_pipeline():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    setup_ax(ax, xlim=(0, 14), ylim=(0, 7),
             title="Pipeline V7 — DETRAC bus/truck subset + imgsz 1280")

    y_top = 5.3
    y_mid = 3.3
    y_bot = 1.3

    # Row 1 — ingestion
    box(ax, 0.2, y_top, 2.4, 1.2, "Phase 1\nIngest DETRAC\n60k frames", color=C_SECONDARY, fontsize=9)
    box(ax, 3.0, y_top, 2.4, 1.2, "Phase 2\nFilter bus/van\ntop 6+6 videos", color=C_SECONDARY, fontsize=9)
    box(ax, 5.8, y_top, 2.4, 1.2, "Phase 3\nStride 20\n~550 frames", color=C_SECONDARY, fontsize=9)
    box(ax, 8.6, y_top, 2.4, 1.2, "Phase 4a\nMerge với V5\ntrain", color=C_PRIMARY, fontsize=9)
    box(ax, 11.4, y_top, 2.4, 1.2, "Phase 4b\nValidate\nmerged dataset", color=C_PRIMARY, fontsize=9)

    # Row arrows
    for x in [2.6, 5.4, 8.2, 11.0]:
        arrow(ax, x, y_top + 0.6, x + 0.4, y_top + 0.6)

    # Phase 4b.5 safety gate (highlight)
    arrow(ax, 12.6, y_top, 12.6, y_mid + 1.2)
    box(ax, 10.5, y_mid, 3.3, 1.2,
        "⚠ Phase 4b.5\nLEAK AUDIT (fail hard)\n0% leak required",
        color=C_DANGER, fontsize=9)

    arrow(ax, 10.5, y_mid + 0.6, 8.5, y_mid + 0.6)

    # Phase 5 train
    box(ax, 4.5, y_mid, 4.0, 1.2,
        "Phase 5 — Train V7\nyolov8s @ 1280, batch=8\nwarm start từ V5+ best.pt",
        color=C_PRIMARY, fontsize=9)
    arrow(ax, 4.5, y_mid + 0.6, 3.0, y_mid + 0.6)

    # Phase 6 eval
    box(ax, 0.2, y_mid, 2.8, 1.2, "Phase 6\nCross-eval\nV5 test + OOD", color=C_ACCENT, fontsize=9)
    arrow(ax, 1.5, y_mid, 1.5, y_bot + 1.2)

    # Output
    box(ax, 0.2, y_bot, 13.6, 1.2,
        "Expected: mAP50 ~0.80-0.83 • bus/truck recall ↑ • verified honest (0% leak)",
        color=C_SUCCESS, fontsize=10)

    plt.tight_layout()
    out = OUT_DIAG / "pipeline_v7.png"
    plt.savefig(out, dpi=170, bbox_inches="tight")
    plt.close()
    print("  ✓", out.name)


# ---------------- YOLO Format: directory tree ----------------

def yolo_dir_tree():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")
    ax.set_title("Cấu trúc thư mục YOLO (data/merged_v5)",
                 fontsize=13, fontweight="bold", color=C_PRIMARY, pad=10)

    tree_lines = [
        ("data/merged_v5/", C_PRIMARY, True, 0),
        ("├── data_v5.yaml", C_ACCENT, True, 1),
        ("├── split_manifest.json", C_SECONDARY, False, 1),
        ("├── images/", C_PRIMARY, True, 1),
        ("│   ├── train/   (2,833 .jpg)", "#333", False, 2),
        ("│   │   ├── cantho_CoopMart-8h-8h30_mp4-0_jpg.rf.ada....jpg", "#555", False, 3),
        ("│   │   ├── soictnight_cam_03_00010_jpg.rf.3b4....jpg", "#555", False, 3),
        ("│   │   └── ... (2,831 more)", "#999", False, 3),
        ("│   ├── val/     (1,466 .jpg)", "#333", False, 2),
        ("│   │   └── cantho_CoopMart-8h30-9h00_mp4-0_jpg.rf....jpg", "#555", False, 3),
        ("│   └── test/    (1,500 .jpg)", "#333", False, 2),
        ("│       └── soict24_cam_09_00001_jpg.rf.175....jpg", "#555", False, 3),
        ("└── labels/", C_PRIMARY, True, 1),
        ("    ├── train/   (2,833 .txt - match each image)", "#333", False, 2),
        ("    │   ├── cantho_CoopMart-8h-8h30_mp4-0_jpg.rf.ada....txt", "#555", False, 3),
        ("    │   └── ...", "#999", False, 3),
        ("    ├── val/     (1,466 .txt)", "#333", False, 2),
        ("    └── test/    (1,500 .txt)", "#333", False, 2),
    ]

    y0 = 0.92
    dy = 0.048
    for i, (txt, color, bold, lvl) in enumerate(tree_lines):
        ax.text(0.03, y0 - i * dy, txt,
                fontfamily="monospace", fontsize=10.5,
                color=color, fontweight="bold" if bold else "normal",
                transform=ax.transAxes)

    # Side note
    ax.text(0.03, 0.03,
            "Nguyên tắc YOLO: với mỗi ảnh X.jpg trong images/<split>/,\n"
            "phải có file X.txt cùng tên trong labels/<split>/.\n"
            "Nếu ảnh không có bbox nào thì file .txt rỗng (vẫn phải tồn tại).",
            family="DejaVu Sans", fontsize=10,
            color=C_GREY, style="italic",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=C_LIGHT, edgecolor="none"))

    out = OUT_YOLO / "yolo_directory_tree.png"
    plt.savefig(out, dpi=170, bbox_inches="tight")
    plt.close()
    print("  ✓", out.name)


# ---------------- YOLO Format: YAML explained ----------------

def yolo_yaml_explained():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axis("off")
    ax.set_title("data_v5.yaml — config dataset cho Ultralytics",
                 fontsize=13, fontweight="bold", color=C_PRIMARY, pad=10)

    # Left: raw YAML
    yaml_text = """# V5 - honest split, video-aware, no leak
# Auto-generated by scripts/data_prep/rebuild_v5_by_video.py
# Schema: {motorcycle, car, bus, truck}

path: /home/xuand/vn_traffic_ai/data/merged_v5
train: images/train
val:   images/val
test:  images/test

nc: 4
names:
  0: motorcycle
  1: car
  2: bus
  3: truck"""

    ax.text(0.02, 0.95, yaml_text,
            fontfamily="monospace", fontsize=10.5,
            color="#222", va="top",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#F7F7F7", edgecolor="#CCC"))

    # Right: annotations
    annotations = [
        ("path", "Thư mục gốc dataset. Các đường dẫn train/val/test\n"
                 "được resolve tương đối từ đây."),
        ("train/val/test", "Đường dẫn tới folder ẢNH. Ultralytics tự suy ra folder\n"
                            "nhãn bằng cách thay 'images' → 'labels' trong path."),
        ("nc", "Number of classes — phải khớp với max class_id\n"
               "trong file .txt."),
        ("names", "Ánh xạ class_id (0,1,2,...) → tên class.\n"
                  "Thứ tự này được dùng xuyên suốt training + inference."),
    ]
    y = 0.92
    for key, desc in annotations:
        ax.text(0.56, y, f"● {key}", fontsize=11, fontweight="bold",
                color=C_PRIMARY, transform=ax.transAxes)
        ax.text(0.58, y - 0.035, desc, fontsize=9.5, color="#333",
                va="top", transform=ax.transAxes)
        y -= 0.19

    out = OUT_YOLO / "yolo_yaml_explained.png"
    plt.savefig(out, dpi=170, bbox_inches="tight")
    plt.close()
    print("  ✓", out.name)


# ---------------- YOLO Format: real label example (3-panel) ----------------

def yolo_label_example():
    # Pick a real train image with labels
    img_dir = ROOT / "data/merged_v5/images/train"
    lbl_dir = ROOT / "data/merged_v5/labels/train"
    # choose one with many classes if possible
    candidates = sorted(img_dir.glob("*.jpg"))
    # try to find an image whose label has 3+ classes
    chosen = None
    for p in candidates[:200]:
        lbl = lbl_dir / (p.stem + ".txt")
        if not lbl.exists():
            continue
        classes = set()
        for line in lbl.read_text().strip().split("\n"):
            if line:
                classes.add(line.split()[0])
        if len(classes) >= 2:
            chosen = p
            if len(classes) >= 3:
                break
    chosen = chosen or candidates[0]
    lbl_path = lbl_dir / (chosen.stem + ".txt")

    img = mpimg.imread(chosen)
    H, W = img.shape[:2]
    lines = [l for l in lbl_path.read_text().strip().split("\n") if l]

    class_names = ["motorcycle", "car", "bus", "truck"]
    class_colors = ["#E74C3C", "#3498DB", "#F39C12", "#27AE60"]

    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.0, 1.4], wspace=0.15)

    # Panel 1: raw image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img)
    ax1.set_title(f"① Ảnh gốc\n{chosen.name[:40]}...\n({W}×{H} px)",
                  fontsize=10.5, fontweight="bold", color=C_PRIMARY)
    ax1.axis("off")

    # Panel 2: label txt content
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    ax2.set_title(f"② File nhãn tương ứng\n{lbl_path.name[:40]}...\n({len(lines)} dòng = {len(lines)} box)",
                  fontsize=10.5, fontweight="bold", color=C_PRIMARY)

    header = "class  cx       cy       w        h"
    header_desc = "─" * 42
    display_lines = [header, header_desc]
    for ln in lines[:14]:
        parts = ln.split()
        if len(parts) == 5:
            c, cx, cy, w, h = parts
            display_lines.append(f"  {c}    {float(cx):.4f}   {float(cy):.4f}   {float(w):.4f}   {float(h):.4f}")
    if len(lines) > 14:
        display_lines.append(f"  ... ({len(lines) - 14} dòng khác)")

    ax2.text(0.02, 0.96, "\n".join(display_lines),
             fontfamily="monospace", fontsize=9.5, va="top",
             transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F7F7F7", edgecolor="#CCC"))

    # Legend of classes
    ax2.text(0.02, 0.30,
             "Giải thích format YOLO:\n"
             "   class  = class_id (0 - nc-1)\n"
             "   cx, cy = tâm box, chuẩn hóa trong [0,1]\n"
             "   w, h   = width/height, chuẩn hóa trong [0,1]\n\n"
             "Tọa độ pixel = giá trị nhân với (W, H) của ảnh.",
             family="DejaVu Sans", fontsize=10, va="top", color="#333",
             transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF8E1", edgecolor="#E6C860"))

    # Panel 3: image with boxes drawn
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img)
    for ln in lines:
        parts = ln.split()
        if len(parts) != 5:
            continue
        c = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        x1 = (cx - w / 2) * W
        y1 = (cy - h / 2) * H
        bw, bh = w * W, h * H
        color = class_colors[c] if c < len(class_colors) else "#999"
        rect = Rectangle((x1, y1), bw, bh, linewidth=1.6,
                         edgecolor=color, facecolor="none")
        ax3.add_patch(rect)
        ax3.text(x1, y1 - 3, class_names[c] if c < len(class_names) else f"c{c}",
                 fontsize=8, color="white",
                 bbox=dict(boxstyle="round,pad=0.15", facecolor=color, edgecolor="none"))
    ax3.set_title(f"③ Visualize bbox từ nhãn\n(ground truth)",
                  fontsize=10.5, fontweight="bold", color=C_PRIMARY)
    ax3.axis("off")

    # Legend
    legend_handles = [mpatches.Patch(color=c, label=n) for c, n in zip(class_colors, class_names)]
    ax3.legend(handles=legend_handles, loc="lower center", ncol=4,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.08))

    plt.suptitle("Format nhãn YOLOv8 — ví dụ thực từ data/merged_v5/train/",
                 fontsize=13, fontweight="bold", color=C_PRIMARY, y=1.00)

    out = OUT_YOLO / "yolo_label_example.png"
    plt.savefig(out, dpi=170, bbox_inches="tight")
    plt.close()
    print("  ✓", out.name, f"[dùng {chosen.name}]")


# ---------------- YOLO Format: coordinate conversion diagram ----------------

def yolo_coord_diagram():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title("Cách chuẩn hóa tọa độ YOLO (normalized 0-1)",
                 fontsize=13, fontweight="bold", color=C_PRIMARY, pad=12)

    # Draw image frame
    W, H = 10, 6
    ax.add_patch(Rectangle((0, 0), W, H, linewidth=2, edgecolor="#333", facecolor="#F0F4F8"))
    ax.text(W / 2, H + 0.25, f"Ảnh gốc: W × H pixel (ví dụ 1280 × 720)",
            ha="center", fontsize=10, color=C_GREY, style="italic")

    # Draw a bbox
    bx, by, bw, bh = 3.5, 2.2, 2.8, 1.8
    cx, cy = bx + bw / 2, by + bh / 2
    ax.add_patch(Rectangle((bx, by), bw, bh, linewidth=2.5, edgecolor=C_DANGER, facecolor="none"))
    ax.plot([cx], [cy], marker="o", color=C_DANGER, markersize=8)

    # Annotations
    # cx line
    ax.annotate("", xy=(cx, -0.3), xytext=(cx, 0),
                arrowprops=dict(arrowstyle="<->", color=C_SECONDARY, lw=1.5))
    ax.text(cx + 0.1, -0.55, "cx (pixel)", fontsize=10, color=C_SECONDARY)
    # cy line
    ax.annotate("", xy=(-0.3, cy), xytext=(0, cy),
                arrowprops=dict(arrowstyle="<->", color=C_SECONDARY, lw=1.5))
    ax.text(-0.9, cy + 0.1, "cy", fontsize=10, color=C_SECONDARY)
    # w/h
    ax.annotate("", xy=(bx, by - 0.3), xytext=(bx + bw, by - 0.3),
                arrowprops=dict(arrowstyle="<->", color=C_PRIMARY, lw=1.5))
    ax.text(bx + bw / 2, by - 0.7, "w (box width)", ha="center", fontsize=9.5, color=C_PRIMARY)
    ax.annotate("", xy=(bx + bw + 0.3, by), xytext=(bx + bw + 0.3, by + bh),
                arrowprops=dict(arrowstyle="<->", color=C_PRIMARY, lw=1.5))
    ax.text(bx + bw + 0.5, by + bh / 2, "h", fontsize=10, color=C_PRIMARY)

    # Formulas (header + monospace block + footer in sans for Vietnamese)
    ax.text(10.3, H / 2 + 1.5, "Công thức chuẩn hóa:",
            family="DejaVu Sans", fontsize=11, fontweight="bold", va="center", color=C_PRIMARY)
    formula = (
        "cx_norm = cx_pixel / W\n"
        "cy_norm = cy_pixel / H\n"
        "w_norm  = w_pixel  / W\n"
        "h_norm  = h_pixel  / H"
    )
    ax.text(10.3, H / 2 + 0.3, formula,
            family="DejaVu Sans Mono", fontsize=11, va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF8E1", edgecolor="#E6C860"))
    ax.text(10.3, H / 2 - 1.5,
            "=> Mọi giá trị trong [0, 1],\n"
            "    không phụ thuộc kích thước ảnh gốc.",
            family="DejaVu Sans", fontsize=10, va="center", color="#333", style="italic")

    # Label on box
    ax.text(cx, by + bh + 0.25, "class = 1 (car)",
            ha="center", fontsize=9.5, color=C_DANGER, fontweight="bold")

    ax.set_xlim(-1.5, 17)
    ax.set_ylim(-1.5, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    out = OUT_YOLO / "yolo_coord_system.png"
    plt.savefig(out, dpi=170, bbox_inches="tight")
    plt.close()
    print("  ✓", out.name)


# ---------------- Main ----------------

def main():
    print("Rendering pipeline diagrams...")
    diag_workflow()
    diag_leak_mechanism()
    diag_rebuild_pipeline()
    diag_training_pipeline()
    diag_v7_pipeline()

    print("Rendering YOLO format visuals...")
    yolo_dir_tree()
    yolo_yaml_explained()
    yolo_label_example()
    yolo_coord_diagram()

    print("Done.")


if __name__ == "__main__":
    main()
