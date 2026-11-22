from __future__ import annotations

from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "docs/tieuluan_V3/Nhom02_PhanLoaiPhuongTien_UPDATED5 (1).docx"
BACKUP = ROOT / "docs/tieuluan_V3/Nhom02_PhanLoaiPhuongTien_UPDATED5 (1).before_UI_FIGURE.docx"
OUT = ROOT / "outputs/verify/docx_update_assets/fig_5_2_streamlit_ui_layout.png"


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def rounded(draw, box, fill, outline="#c8d0da", radius=16, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def make_ui_mock() -> None:
    w, h = 1800, 1050
    im = Image.new("RGB", (w, h), "#f4f6f8")
    d = ImageDraw.Draw(im)
    # Header
    d.rectangle((0, 0, w, 105), fill="#ffffff")
    d.text((55, 28), "HỆ THỐNG PHÁT HIỆN VÀ ĐẾM PHƯƠNG TIỆN", fill="#17202a", font=font(34, True))
    d.text((1370, 38), "YOLOv8 + ByteTrack", fill="#52616b", font=font(22))

    # Sidebar
    rounded(d, (35, 130, 500, 1015), "#ffffff")
    d.text((65, 160), "CẤU HÌNH", fill="#1f4e79", font=font(28, True))
    labels = [
        ("Mô hình", "r_clspw_v2 (mặc định)"),
        ("Pipeline", "Improved"),
        ("Tracker", "ByteTrack"),
        ("Confidence", "0,30"),
        ("IoU", "0,50"),
        ("Vạch đếm", "Ngang – vị trí 50%"),
        ("Hướng", "Cả hai chiều"),
    ]
    y = 220
    for lab, val in labels:
        d.text((70, y), lab, fill="#34495e", font=font(19, True))
        rounded(d, (65, y + 32, 465, y + 82), "#f7f9fb", radius=9, width=1)
        d.text((82, y + 44), val, fill="#2c3e50", font=font(18))
        y += 108
    rounded(d, (65, 925, 465, 982), "#1f77b4", outline="#1f77b4", radius=10)
    d.text((183, 939), "CHẠY PHÂN TÍCH", fill="white", font=font(20, True))

    # Main content
    rounded(d, (530, 130, 1765, 1015), "#ffffff")
    d.text((570, 165), "Tải video đầu vào", fill="#273746", font=font(25, True))
    rounded(d, (565, 210, 1725, 290), "#f8fafc", outline="#9aa9b8", radius=12)
    d.text((820, 235), "Kéo thả MP4 / MOV / AVI hoặc chọn tệp", fill="#607080", font=font(21))

    # Video preview
    rounded(d, (565, 320, 1325, 790), "#263238", outline="#263238", radius=12)
    d.text((830, 525), "VIDEO PREVIEW", fill="#dfe6e9", font=font(32, True))
    d.line((620, 610, 1265, 610), fill="#ff5252", width=6)
    d.text((620, 625), "Vạch đếm", fill="#ffb3b3", font=font(18, True))
    # sample boxes
    for box, label, color in [((720, 420, 850, 535), "car #17", "#35d07f"), ((1010, 470, 1100, 585), "motor #8", "#ffd54f")]:
        d.rectangle(box, outline=color, width=5)
        d.text((box[0], box[1] - 28), label, fill=color, font=font(17, True))

    # Counts panel
    rounded(d, (1360, 320, 1725, 790), "#f8fafc", radius=12)
    d.text((1410, 350), "KẾT QUẢ ĐẾM", fill="#1f4e79", font=font(22, True))
    counts = [("motorcycle", "0"), ("car", "87"), ("bus", "0"), ("truck", "6"), ("Tổng", "93")]
    y = 410
    for name, value in counts:
        d.line((1400, y + 45, 1685, y + 45), fill="#d8dee4", width=2)
        d.text((1410, y), name, fill="#34495e", font=font(19, name == "Tổng"))
        d.text((1630, y), value, fill="#17202a", font=font(20, True))
        y += 70

    # Output buttons
    d.text((570, 835), "Xuất kết quả", fill="#273746", font=font(24, True))
    for x, label in [(565, "TẢI CSV SỰ KIỆN"), (900, "TẢI VIDEO MP4"), (1235, "BÁO CÁO ĐÁNH GIÁ")]:
        rounded(d, (x, 885, x + 290, 950), "#eaf2f8", outline="#6fa8c7", radius=10)
        d.text((x + 28, 905), label, fill="#1f4e79", font=font(17, True))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    im.save(OUT)


def main() -> None:
    if not BACKUP.exists():
        shutil.copy2(DOCX, BACKUP)
    make_ui_mock()
    doc = Document(DOCX)
    cap_i = next(
        i for i, p in enumerate(doc.paragraphs)
        if p.style.name == "Chu thich hinh" and p.text.startswith("Hình 5.2")
    )
    image_p = doc.paragraphs[cap_i - 1]
    image_p._p.clear_content()
    image_p.style = doc.styles["Hinh anh"]
    image_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    image_p.add_run().add_picture(str(OUT), width=Inches(6.4))
    doc.save(DOCX)
    print(f"Updated {DOCX}")


if __name__ == "__main__":
    main()
