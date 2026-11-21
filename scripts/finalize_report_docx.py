from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches
from docx.text.paragraph import Paragraph


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "docs/tieuluan_V3/Nhom02_PhanLoaiPhuongTien_UPDATED5 (1).docx"
BACKUP = ROOT / "docs/tieuluan_V3/Nhom02_PhanLoaiPhuongTien_UPDATED5 (1).before_FINALIZE_47.docx"
ASSETS = ROOT / "outputs/verify/docx_update_assets"


def set_paragraph(p, text: str) -> None:
    if p.runs:
        p.runs[0].text = text
        for run in p.runs[1:]:
            run.text = ""
    else:
        p.add_run(text)


def replace_start(doc: Document, prefix: str, text: str) -> None:
    matches = [p for p in doc.paragraphs if p.text.strip().startswith(prefix)]
    if not matches:
        if any(p.text.strip() == text for p in doc.paragraphs):
            return
        raise RuntimeError(f"paragraph not found: {prefix}")
    for p in matches:
        set_paragraph(p, text)


def paragraph_after(p: Paragraph, text: str = "", style: str | None = None) -> Paragraph:
    new_p = OxmlElement("w:p")
    p._p.addnext(new_p)
    out = Paragraph(new_p, p._parent)
    if style:
        out.style = style
    if text:
        out.add_run(text)
    return out


def create_confusion_matrix() -> Path:
    data = json.loads((ROOT / "outputs/verify/error_analysis_r_clspw_v2.json").read_text())
    cm = np.array(data["confusion_matrix"]["matrix"], dtype=int)
    labels = ["motorcycle", "car", "bus", "truck", "background"]
    out = ASSETS / "fig_4_4_confusion_matrix_r_clspw_v2.png"
    fig, ax = plt.subplots(figsize=(8.4, 6.8), dpi=200)
    shown = np.log1p(cm)
    im = ax.imshow(shown, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if shown[i, j] > shown.max() * 0.55 else "black"
            ax.text(j, i, f"{cm[i, j]:,}".replace(",", "."), ha="center", va="center", color=color, fontsize=10)
    ax.set_xticks(range(5), labels, rotation=25, ha="right")
    ax.set_yticks(range(5), labels)
    ax.set_xlabel("Nhãn ground truth")
    ax.set_ylabel("Nhãn dự đoán")
    ax.set_title("Ma trận nhầm lẫn r_clspw_v2 – V5+ test\nconf=0,25; IoU ghép=0,5")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log(1 + số trường hợp)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def create_error_montage() -> Path:
    src = ROOT / "outputs/verify/error_examples_v2"
    rows = [
        ("Bỏ sót (FN)", sorted(src.glob("missed_*.jpg"))[:3]),
        ("Dự đoán dư (FP)", sorted(src.glob("extra_*.jpg"))[:3]),
        ("Nhầm lớp", sorted(src.glob("wrong_cls_*.jpg"))[:3]),
    ]
    if any(len(paths) != 3 for _, paths in rows):
        raise RuntimeError("expected three images for each error category")
    tile_w, tile_h = 640, 360
    label_h = 42
    canvas = Image.new("RGB", (tile_w * 3, (tile_h + label_h) * 3), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    for r, (label, paths) in enumerate(rows):
        for c, path in enumerate(paths):
            im = Image.open(path).convert("RGB").resize((tile_w, tile_h))
            x, y = c * tile_w, r * (tile_h + label_h)
            canvas.paste(im, (x, y + label_h))
            cell_label = f"{label} – ví dụ {c + 1}"
            draw.rectangle((x, y, x + tile_w, y + label_h), fill=(245, 245, 245))
            draw.text((x + 12, y + 7), cell_label, fill=(20, 20, 20), font=font)
    out = ASSETS / "fig_4_5_error_examples_r_clspw_v2.jpg"
    canvas.save(out, quality=92)
    return out


def add_section_47_figures(doc: Document, cm_path: Path, montage_path: Path) -> None:
    if any(p.text.startswith("Hình 4.4 Ma trận nhầm lẫn") for p in doc.paragraphs):
        return
    anchor = next(p for p in doc.paragraphs if p.text.startswith("Ma trận nhầm lẫn cho thấy"))

    image_p = paragraph_after(anchor)
    image_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    image_p.add_run().add_picture(str(cm_path), width=Inches(6.25))
    cap = paragraph_after(image_p, "Hình 4.4 Ma trận nhầm lẫn của r_clspw_v2 trên V5+ test; hàng là lớp dự đoán, cột là lớp ground truth.", "Chu thich hinh")

    image_p2 = paragraph_after(cap)
    image_p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    image_p2.add_run().add_picture(str(montage_path), width=Inches(6.5))
    paragraph_after(image_p2, "Hình 4.5 Ví dụ lỗi của r_clspw_v2: bỏ sót, dự đoán dư và nhầm lớp trên V5+ test.", "Chu thich hinh")


def update_count_table(doc: Document) -> None:
    target = None
    for table in doc.tables:
        header = [c.text.strip() for c in table.rows[0].cells]
        if header == ["Lớp", "Ground truth", "Hệ thống", "Sai số tuyệt đối", "Sai số tương đối"]:
            target = table
            break
    if target is None:
        raise RuntimeError("counting table not found")
    rows = [
        ["Lớp", "Ground truth", "Hệ thống", "Sai số tuyệt đối", "Sai số tương đối"],
        ["motorcycle", "0", "0", "0", "—"],
        ["car", "80", "87", "7", "8,75%"],
        ["bus", "1", "0", "1", "100%"],
        ["truck", "6", "6", "0", "0%"],
        ["Tổng", "87", "93", "6", "6,90%"],
    ]
    for row, vals in zip(target.rows, rows):
        for cell, val in zip(row.cells, vals):
            set_paragraph(cell.paragraphs[0], val)


def request_field_updates(doc: Document) -> None:
    settings = doc.settings._element
    old = settings.find(qn("w:updateFields"))
    if old is None:
        old = OxmlElement("w:updateFields")
        settings.append(old)
    old.set(qn("w:val"), "true")
    for fld in doc.element.body.xpath('.//w:fldChar[@w:fldCharType="begin"]'):
        fld.set(qn("w:dirty"), "true")


def clean_current_table_list_entry(doc: Document) -> None:
    """Keep the visible field result readable until Word regenerates the list."""
    for p in doc.paragraphs[:130]:
        if p.text.startswith("Bảng 4.10 Kết quả đếm phương tiện"):
            p._p.clear_content()
            p.add_run("Bảng 4.10 Kết quả đếm phương tiện của r_clspw_v2 trên video demo_traffic.mp4.\t56")
            break


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    if not BACKUP.exists():
        shutil.copy2(DOCX, BACKUP)
    doc = Document(DOCX)

    replacements = [
        ("r_clspw_v2 is selected for in-domain deployment", "r_clspw_v2 is selected for in-domain deployment, while p4_scale is preferable for the evaluated aerial domain. YOLO and Faster R-CNN are compared on the same 1,500-image V5+ test set. On demo_traffic.mp4, the final r_clspw_v2 pipeline counts 93 vehicles against 87 ground-truth events, corresponding to a 6.90% total-count error and a per-class MAE of 2.00."),
        ("r_clspw_v2 phù hợp mục tiêu in-domain", "r_clspw_v2 phù hợp mục tiêu in-domain, còn p4_scale tốt hơn trên VN Aerial đã khảo sát. YOLO và Faster R-CNN được so sánh trên cùng tập test V5+ gồm 1.500 ảnh. Trên demo_traffic.mp4, pipeline dùng r_clspw_v2 đếm 93 lượt so với 87 lượt ground truth, sai số tổng 6,90% và MAE theo lớp 2,00."),
        ("Bảng 4.10 Kết quả đếm phương tiện", "Bảng 4.10 Kết quả đếm phương tiện của r_clspw_v2 trên video demo_traffic.mp4."),
        ("Chương 5 mô tả ứng dụng Streamlit", "Chương 5 mô tả ứng dụng Streamlit, kết quả tốc độ và phép đếm đã được đo lại bằng checkpoint cuối r_clspw_v2 trên video demo có ground truth."),
        ("Huấn luyện baseline exp_baseline", "Huấn luyện baseline exp_baseline và khảo sát p4_scale, p4_imgsz1280, r_p2, r_clspw_v2 và r_progressive. r_clspw_v2 warm-start từ exp_baseline/best.pt; r_p2 thay đổi kiến trúc và dùng phần trọng số COCO tương thích. Faster R-CNN được xây dựng riêng làm mô hình đối chứng và được đánh giá trên cùng V5+ test."),
        ("Phân rã theo kích thước cho thấy", "Phân rã theo kích thước cho thấy recall motorcycle đạt 49,8% ở nhóm nhỏ, 79,3% ở nhóm vừa và 95,4% ở nhóm lớn; với car lần lượt là 75,5%, 90,1% và 93,7%. Bus đạt 57,9% ở nhóm vừa và 77,1% ở nhóm lớn; truck đạt 48,2% và 86,1%. Trong Bảng 4.9, FP theo kích thước là dự đoán không ghép với ground truth (FP nền); các trường hợp nhầm lớp được thể hiện riêng trong ma trận nhầm lẫn. Tập test chỉ gồm cam_09/cam_10 ban ngày nên không thể kết luận chênh lệch ngày–đêm."),
        ("Pipeline đếm (ByteTrack", "Pipeline đếm (ByteTrack, conf=0,3, IoU=0,5, imgsz=960, FP16) được chạy lại bằng checkpoint cuối r_clspw_v2 trên demo_traffic.mp4 gồm 178 khung hình. Vạch ngang đặt tại y=474 và đếm theo cả hai chiều."),
        ("Trên video thử nghiệm, hệ thống đếm", "Trên video thử nghiệm, r_clspw_v2 đếm 87 car so với GT 80, bỏ sót 1 bus, đếm đúng 6 truck và video không có motorcycle. Tổng hệ thống đếm 93 lượt so với 87 lượt GT; sai số tổng là 6 lượt (6,90%) và MAE theo lớp là 2,00. Vì chỉ có một video GT, kết quả chưa đủ để khái quát cho nhiều camera."),
        ("Trên demo_traffic.mp4 (87 lượt GT)", "Trên demo_traffic.mp4 (87 lượt GT), pipeline với r_clspw_v2 đếm 93 lượt, sai số tổng 6,90% và MAE theo lớp 2,00. Benchmark tốc độ riêng đạt 37,8 FPS cold-start, 80,4 FPS warm khi không ghi video và 45,1 FPS warm khi ghi MP4. Cần thêm nhiều video có GT để đánh giá độ ổn định."),
        ("Ứng dụng hiện có tải video", "Ứng dụng hiện có tải video, lựa chọn checkpoint YOLO, một vạch ngang/dọc, theo vết, thống kê và xuất kết quả. r_clspw_v2 là checkpoint ưu tiên; pipeline đạt 37,8 FPS cold-start và 80,4 FPS warm. Trên video demo có GT, sai số tổng là 6,90%; cần mở rộng đánh giá trên nhiều video."),
        ("Đóng góp 5 — Cài đặt ứng dụng demo", "Đóng góp 5 — Cài đặt ứng dụng demo. Streamlit có pipeline phát hiện, theo vết, đếm qua một vạch và xuất CSV/video. Với r_clspw_v2, pipeline đạt 37,8 FPS cold-start và 80,4 FPS warm; trên video demo có GT, hệ thống đếm 93/87 lượt, sai số tổng 6,90%."),
        ("Hạn chế 3 — Chưa có nhãn track ID", "Hạn chế 3 — Chưa có nhãn track ID để đo IDF1/MOTA/HOTA. Phép đếm có GT mới chỉ gồm một video; validation cũ còn trùng camera và quy trình chọn recipe đã tham khảo test."),
        ("Checkpoint, log huấn luyện", "Checkpoint, log huấn luyện, kết quả đánh giá, phép đếm bằng r_clspw_v2 và benchmark tốc độ của các run chính đã được đối chiếu. Hạn chế còn lại là bộ GT đếm mới có một video và thiếu đánh giá tracking theo ID."),
    ]
    for prefix, text in replacements:
        replace_start(doc, prefix, text)

    update_count_table(doc)
    cm_path = create_confusion_matrix()
    montage_path = create_error_montage()
    add_section_47_figures(doc, cm_path, montage_path)
    clean_current_table_list_entry(doc)
    request_field_updates(doc)
    doc.save(DOCX)
    print(f"Updated {DOCX}")
    print(f"Backup  {BACKUP}")


if __name__ == "__main__":
    main()
