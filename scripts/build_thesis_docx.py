"""Convert all revised markdown chapters to a single .docx with proper formatting.

Format theo yêu cầu GVHD:
- Times New Roman 13, canh đều, line spacing 1.5, indent 1cm
- Heading 1: 14, bold, uppercase, center
- Heading 2: 13, bold
- Heading 3: 13, bold italic
- Caption: 12, center
- Table cell: 12, line spacing 1.0
- Margin: left 3cm, right 2cm, top/bottom 2cm, A4

Embeds images from docs/tieuluan_V3/images/ at labeled positions.
"""
from pathlib import Path
import re
from docx import Document
from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path("/home/xuand/vn_traffic_ai/docs/tieuluan_V3")
IMAGES = ROOT / "images"
OUT = ROOT / "Nhom02_PhanLoaiPhuongTien_FINAL.docx"

CHAPTERS = [
    ROOT / "chapter1_revised.md",
    ROOT / "chapter2_revised.md",
    ROOT / "chapter3_revised.md",
    ROOT / "chapter4_new.md",
    ROOT / "chapter5_new.md",
    ROOT / "conclusion_new.md",
]

# Mapping figure numbers → image filename
FIGURE_IMAGES = {
    "Hình 3.1": "fig_3_pipeline.png",
    "Hình 3.2": "fig_3_sample_soict_night.jpg",
    "Hình 3.3": "fig_3_sample_soict_2024.jpg",
    "Hình 3.4": "fig_3_sample_cantho.jpg",
    "Hình 3.5": "fig_3_sample_vnv3.jpg",
    "Hình 3.6": "fig_3_sample_v10.jpg",
    "Hình 4.1": "fig_4_training_curves_v5plus.png",
    "Hình 4.2": "fig_4_confusion_matrix_p4scale.png",
    # Comparison charts inserted after key tables
    "Hình 4.5": "fig_4_model_comparison.png",
    "Hình 4.6": "fig_4_per_class_ap.png",
    "Hình 4.7": "fig_4_yolo_vs_frcnn.png",
    "Hình 4.8": "fig_4_fps_comparison.png",
    "Hình 4.9": "fig_4_val_test_gap.png",
    "Hình 4.10": "fig_4_counting.png",
    "Hình 5.1": "fig_5_streamlit_mockup.png",  # pipeline flow
    "Hình 5.2": "fig_5_streamlit_mockup.png",  # main UI
}


def set_margins(doc):
    section = doc.sections[0]
    section.page_height = Cm(29.7); section.page_width = Cm(21.0)
    section.left_margin = Cm(3); section.right_margin = Cm(2)
    section.top_margin = Cm(2); section.bottom_margin = Cm(2)


def set_normal_style(doc):
    style = doc.styles["Normal"]
    f = style.font; f.name = "Times New Roman"; f.size = Pt(13)
    # Set East Asian font fallback
    rpr = style.element.get_or_add_rPr()
    rFonts = rpr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts"); rpr.append(rFonts)
    rFonts.set(qn("w:ascii"), "Times New Roman")
    rFonts.set(qn("w:hAnsi"), "Times New Roman")
    rFonts.set(qn("w:cs"), "Times New Roman")
    rFonts.set(qn("w:eastAsia"), "Times New Roman")

    pf = style.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.line_spacing = 1.5
    pf.space_before = Pt(0); pf.space_after = Pt(6)
    pf.first_line_indent = Cm(1)


def set_run_font(run, size=13, bold=False, italic=False, color=None):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size); run.bold = bold; run.italic = italic
    if color: run.font.color.rgb = color
    # East Asian fallback
    rpr = run._element.get_or_add_rPr()
    rFonts = rpr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts"); rpr.append(rFonts)
    rFonts.set(qn("w:ascii"), "Times New Roman"); rFonts.set(qn("w:hAnsi"), "Times New Roman")
    rFonts.set(qn("w:cs"), "Times New Roman"); rFonts.set(qn("w:eastAsia"), "Times New Roman")


def add_heading(doc, text, level=1):
    """Heading theo yêu cầu GVHD:
    - H1: 14, bold, uppercase, center, page break trước
    - H2: 13, bold
    - H3: 13, bold italic
    """
    p = doc.add_paragraph()
    if level == 1:
        # Page break trước mỗi chương (trừ chương đầu tiên — xử lý ngoài)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_before = Pt(0); p.paragraph_format.space_after = Pt(12)
        run = p.add_run(text.upper())
        set_run_font(run, size=14, bold=True)
    elif level == 2:
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_before = Pt(12); p.paragraph_format.space_after = Pt(6)
        run = p.add_run(text)
        set_run_font(run, size=13, bold=True)
    elif level == 3:
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_before = Pt(6); p.paragraph_format.space_after = Pt(6)
        run = p.add_run(text)
        set_run_font(run, size=13, bold=True, italic=True)
    else:
        p.paragraph_format.first_line_indent = Cm(0)
        run = p.add_run(text)
        set_run_font(run, size=13, bold=True, italic=True)
    return p


def add_paragraph(doc, text):
    """Đoạn văn thân bài 13pt, justify, 1.5 line, indent 1cm."""
    p = doc.add_paragraph()
    # Parse inline bold **text** and italic *text*
    pattern = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")
    parts = pattern.split(text)
    for part in parts:
        if not part: continue
        if part.startswith("**") and part.endswith("**"):
            run = p.add_run(part[2:-2]); set_run_font(run, bold=True)
        elif part.startswith("*") and part.endswith("*") and not part.startswith("**"):
            run = p.add_run(part[1:-1]); set_run_font(run, italic=True)
        elif part.startswith("`") and part.endswith("`"):
            run = p.add_run(part[1:-1]); set_run_font(run)
            run.font.name = "Consolas"; run.font.size = Pt(11)
        else:
            run = p.add_run(part); set_run_font(run)
    return p


def add_bullet(doc, text):
    """Bullet list — cùng font size."""
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.first_line_indent = Cm(0); p.paragraph_format.left_indent = Cm(1)
    # Parse inline formatting
    pattern = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")
    parts = pattern.split(text)
    for part in parts:
        if not part: continue
        if part.startswith("**") and part.endswith("**"):
            run = p.add_run(part[2:-2]); set_run_font(run, bold=True)
        elif part.startswith("*") and part.endswith("*") and not part.startswith("**"):
            run = p.add_run(part[1:-1]); set_run_font(run, italic=True)
        else:
            run = p.add_run(part); set_run_font(run)
    return p


def add_table_from_md(doc, lines):
    """Parse markdown table block (list of lines) và tạo Word table.
    Format: | col1 | col2 | ...
            |------|------|
            | val1 | val2 |
    """
    rows = []
    for line in lines:
        line = line.strip()
        if not line or set(line) <= set("|-: "):
            continue  # separator line
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            rows.append(cells)
    if not rows: return
    n_rows = len(rows); n_cols = max(len(r) for r in rows)
    table = doc.add_table(rows=n_rows, cols=n_cols)
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row_data in enumerate(rows):
        for j in range(n_cols):
            cell = table.cell(i, j)
            text = row_data[j] if j < len(row_data) else ""
            # Clear default paragraph
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_before = Pt(0); p.paragraph_format.space_after = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            # Parse inline bold
            pattern = re.compile(r"(\*\*[^*]+\*\*)")
            parts = pattern.split(text)
            for part in parts:
                if not part: continue
                if part.startswith("**") and part.endswith("**"):
                    run = p.add_run(part[2:-2]); set_run_font(run, size=12, bold=True)
                else:
                    run = p.add_run(part); set_run_font(run, size=12, bold=(i == 0))
    doc.add_paragraph("")  # spacing after table


def add_figure_caption(doc, text):
    """Caption Hình X.Y — 12pt center italic."""
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0); p.paragraph_format.space_after = Pt(12)
    run = p.add_run(text); set_run_font(run, size=12, italic=True)
    return p


def add_figure(doc, image_path, caption):
    """Insert image centered, max width 15cm, kèm caption bên dưới."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_before = Pt(12); p.paragraph_format.space_after = Pt(0)
    run = p.add_run()
    try:
        run.add_picture(str(image_path), width=Cm(15))
    except Exception as e:
        run.add_text(f"[Không thể chèn ảnh: {image_path}: {e}]")
    add_figure_caption(doc, caption)


def process_markdown(doc, md_file: Path, is_first_chapter=False):
    """Parse markdown và append vào doc."""
    content = md_file.read_text(encoding="utf-8")
    lines = content.split("\n")
    i = 0
    first_h1 = True
    while i < len(lines):
        line = lines[i].rstrip()

        # Fenced code block ```...``` — render dưới dạng mono
        if line.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].rstrip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            # Render code block as plain mono paragraph
            p = doc.add_paragraph()
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.left_indent = Cm(1)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_before = Pt(6); p.paragraph_format.space_after = Pt(6)
            run = p.add_run("\n".join(code_lines))
            run.font.name = "Consolas"; run.font.size = Pt(10)
            continue

        # Headings
        if line.startswith("# "):
            # Add page break trước H1 (trừ chương đầu tiên)
            if not (is_first_chapter and first_h1):
                doc.add_page_break()
            first_h1 = False
            add_heading(doc, line[2:].strip(), level=1)
            i += 1; continue
        if line.startswith("## "):
            add_heading(doc, line[3:].strip(), level=2)
            i += 1; continue
        if line.startswith("### "):
            add_heading(doc, line[4:].strip(), level=3)
            i += 1; continue
        if line.startswith("#### "):
            add_heading(doc, line[5:].strip(), level=4)
            i += 1; continue

        # Table (starts with |)
        if line.startswith("|"):
            tbl_lines = []
            while i < len(lines) and lines[i].rstrip().startswith("|"):
                tbl_lines.append(lines[i].rstrip()); i += 1
            add_table_from_md(doc, tbl_lines)
            continue

        # Bullet list
        if line.startswith("- ") or line.startswith("• "):
            add_bullet(doc, line[2:].strip())
            i += 1; continue

        # Numbered list → bullet (đơn giản hóa)
        if re.match(r"^\d+\.\s", line):
            m = re.match(r"^\d+\.\s(.+)", line)
            add_bullet(doc, m.group(1))
            i += 1; continue

        # Figure placeholder — e.g. "**Hình 4.1** caption text"
        fig_match = re.match(r"^\*\*(Hình \d+\.\d+)\*\*\s*(.*)$", line)
        if fig_match:
            fig_num = fig_match.group(1); caption_body = fig_match.group(2).strip()
            caption = f"{fig_num}  {caption_body}" if caption_body else fig_num
            img_file = FIGURE_IMAGES.get(fig_num)
            if img_file and (IMAGES / img_file).exists():
                add_figure(doc, IMAGES / img_file, caption)
            else:
                # Chỉ hiển thị caption nếu không có ảnh
                add_figure_caption(doc, f"{caption}  [Ảnh sẽ chèn tại đây]")
            i += 1; continue

        # Blank line
        if not line.strip():
            i += 1; continue

        # Normal paragraph (có thể gộp nhiều dòng liên tiếp)
        para_lines = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "- ", "• ")) and not re.match(r"^\d+\.\s", lines[i]) and not re.match(r"^\*\*(Hình|Bảng) \d+\.\d+\*\*", lines[i]):
            para_lines.append(lines[i].rstrip()); i += 1
        add_paragraph(doc, " ".join(para_lines))


def add_cover_page(doc):
    """Trang bìa theo yêu cầu GVHD."""
    # University
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    run = p.add_run("TRƯỜNG ĐẠI HỌC …\nKHOA CÔNG NGHỆ THÔNG TIN\nNGÀNH KHOA HỌC MÁY TÍNH — MÃ NGÀNH 7480101")
    set_run_font(run, size=13, bold=True)

    doc.add_paragraph("\n\n")

    # Report title
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    run = p.add_run("BÁO CÁO ĐỒ ÁN MÔN HỌC\nTHỊ GIÁC MÁY TÍNH VÀ ỨNG DỤNG")
    set_run_font(run, size=16, bold=True)

    doc.add_paragraph("\n")

    # Project title
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    run = p.add_run("ĐỀ TÀI:\nPHÁT HIỆN, PHÂN LOẠI VÀ ĐẾM PHƯƠNG TIỆN GIAO THÔNG VIỆT NAM TỪ VIDEO BẰNG CNN")
    set_run_font(run, size=15, bold=True)

    doc.add_paragraph("\n\n")

    # Instructor
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    run = p.add_run("Giảng viên hướng dẫn: TS. Lê Thị Vĩnh Thanh")
    set_run_font(run, size=13, bold=True)

    doc.add_paragraph("")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    run = p.add_run("Nhóm thực hiện: Nhóm 02\nLớp học phần: [ghi mã lớp HP]")
    set_run_font(run, size=13)

    doc.add_paragraph("\n\n\n")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    run = p.add_run("TP. Hồ Chí Minh, năm 2026")
    set_run_font(run, size=13, italic=True)

    doc.add_page_break()


def add_front_matter(doc):
    """Lời cảm ơn, Nhận xét GV, Tóm tắt, Mục lục placeholders."""
    for title, content in [
        ("LỜI CẢM ƠN",
         "Nhóm xin trân trọng cảm ơn Giảng viên hướng dẫn TS. Lê Thị Vĩnh Thanh đã tận tình hỗ trợ và góp ý chi tiết trong suốt quá trình thực hiện đồ án. Đặc biệt, các nhận xét về phương pháp luận (đặc biệt là vấn đề rò rỉ dữ liệu khi chia tập theo khung hình) đã giúp nhóm điều chỉnh hướng tiếp cận và đạt được kết quả thực chất hơn. Nhóm cũng xin cảm ơn các tác giả của các bộ dữ liệu giao thông Việt Nam công khai trên Roboflow, đã đóng góp quan trọng vào phần dữ liệu của đề tài."),
        ("NHẬN XÉT CỦA GIẢNG VIÊN HƯỚNG DẪN",
         "[Phần này để trống, giảng viên sẽ ghi nhận xét sau khi đọc bản cuối]"),
        ("TÓM TẮT",
         "Đề tài xây dựng hệ thống phát hiện, phân loại và đếm phương tiện giao thông Việt Nam từ video bằng mô hình CNN. Bộ dữ liệu V5+ gồm 10.203 ảnh được hợp nhất từ năm nguồn Roboflow công khai, chia tập theo video_id với kiểm tra tự động xác nhận 0% rò rỉ. Mô hình chính YOLOv8s được tinh chỉnh qua ba giai đoạn (V5 honest 640px → V5+ 960px → p4_scale finetune), đạt mAP@0.5 = 0,787 trên tập test gồm 1.500 ảnh. So sánh với mô hình đối chứng Faster R-CNN ResNet50-FPN trên cùng tập test, YOLOv8s vượt +14,6 điểm mAP@0.5 và nhanh hơn ~10 lần (55,6 FPS vs 5,4 FPS). Hệ thống đếm phương tiện tích hợp ByteTrack đạt sai số MAPE 5,1% trên các đoạn video kiểm thử có đếm tay. Toàn bộ pipeline được đóng gói thành ứng dụng demo bằng Streamlit."),
        ("MỤC LỤC",
         "[Cập nhật tự động trong Word: References → Table of Contents]"),
        ("DANH MỤC HÌNH",
         "[Cập nhật tự động trong Word: References → Insert Table of Figures]"),
        ("DANH MỤC BẢNG",
         "[Cập nhật tự động trong Word: References → Insert Table of Figures → Caption label: Bảng]"),
    ]:
        # H1 heading
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_before = Pt(0); p.paragraph_format.space_after = Pt(18)
        run = p.add_run(title); set_run_font(run, size=14, bold=True)
        # Content
        add_paragraph(doc, content)
        doc.add_page_break()


def main():
    doc = Document()
    set_margins(doc)
    set_normal_style(doc)

    # Cover
    add_cover_page(doc)
    # Front matter
    add_front_matter(doc)

    # Chapters
    for idx, md_file in enumerate(CHAPTERS):
        print(f"Processing {md_file.name}...")
        process_markdown(doc, md_file, is_first_chapter=(idx == 0))

    # Save
    doc.save(str(OUT))
    import os
    print(f"\n✓ Saved: {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
