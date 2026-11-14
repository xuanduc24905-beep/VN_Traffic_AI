"""Cập nhật trực tiếp file Nhom02_PhanLoaiPhuongTien_NhanXet_GVHD.docx:
1. Xoá các đoạn COMMENT màu đỏ C00000 của cô (các em xoá trước khi nộp bản cuối)
2. Fix "năm lớp" → "4 lớp", "năm loại" → "4 loại", "schema năm lớp" → "schema 4 lớp"
3. Fix mục 1.8/1.9 title: "Sản phẩm dự kiến" → "Sản phẩm đã thực hiện", v.v.
4. Chèn Ch4 Thực nghiệm, Ch5 Hệ thống, Kết luận TRƯỚC phần Tài liệu tham khảo
5. Giữ nguyên format gốc (styles, TOC, captions).

Output: Nhom02_PhanLoaiPhuongTien_UPDATED.docx (cạnh file gốc)
"""
from pathlib import Path
import re
import copy
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path("/home/xuand/vn_traffic_ai/docs/tieuluan_V3")
SRC = ROOT / "Nhom02_PhanLoaiPhuongTien_NhanXet_GVHD.docx"
OUT = ROOT / "Nhom02_PhanLoaiPhuongTien_UPDATED.docx"
IMAGES = ROOT / "images"

# Map figure placeholders → image files
FIGURE_IMAGES = {
    "Hình 4.1": "fig_4_training_curves_v5plus.png",
    "Hình 4.2": "fig_4_confusion_matrix_p4scale.png",
    "Hình 4.5": "fig_4_model_comparison.png",
    "Hình 4.6": "fig_4_per_class_ap.png",
    "Hình 4.7": "fig_4_yolo_vs_frcnn.png",
    "Hình 4.8": "fig_4_fps_comparison.png",
    "Hình 4.9": "fig_4_val_test_gap.png",
    "Hình 4.10": "fig_4_counting.png",
    "Hình 5.1": "fig_5_streamlit_mockup.png",
    "Hình 5.2": "fig_5_streamlit_mockup.png",
}

CHAPTERS_TO_APPEND = [
    ROOT / "chapter4_new.md",
    ROOT / "chapter5_new.md",
    ROOT / "conclusion_new.md",
]

# Replacements inline (regex)
REPLACEMENTS = [
    (r"\bnăm lớp\b", "4 lớp"),
    (r"\bnăm loại\b", "4 loại"),
    (r"\bschema năm lớp\b", "schema 4 lớp"),
    (r"\b5 lớp\b", "4 lớp"),
    (r"nhận dạng năm loại xe", "nhận dạng 4 loại xe"),
    (r"năm loại xe", "4 loại xe"),
    (r"năm nhóm xe", "4 nhóm xe"),
    (r"1\.8 Sản phẩm dự kiến", "1.8 Sản phẩm đã thực hiện"),
    (r"1\.9 Ý nghĩa và đóng góp dự kiến", "1.9 Ý nghĩa và đóng góp đã đạt được"),
    (r"2\.9 Công nghệ và công cụ dự kiến", "2.9 Công nghệ và công cụ đã sử dụng"),
]

RED_HEX = "C00000"


def is_teacher_comment(paragraph):
    """Return True nếu đoạn có bất kỳ run nào có color = C00000 (đỏ)."""
    for run in paragraph.runs:
        rPr = run._element.find(qn("w:rPr"))
        if rPr is None:
            continue
        color = rPr.find(qn("w:color"))
        if color is None:
            continue
        val = color.get(qn("w:val"), "").upper()
        if val == RED_HEX:
            return True
    return False


def set_font(run, size=13, bold=False, italic=False):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    rpr = run._element.get_or_add_rPr()
    rFonts = rpr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts"); rpr.append(rFonts)
    rFonts.set(qn("w:ascii"), "Times New Roman")
    rFonts.set(qn("w:hAnsi"), "Times New Roman")
    rFonts.set(qn("w:cs"), "Times New Roman")
    rFonts.set(qn("w:eastAsia"), "Times New Roman")


def apply_text_replacements(para, patterns):
    """Thay thế text trong paragraph, giữ nguyên format run đầu tiên."""
    if not para.runs:
        return
    full = para.text
    changed = False
    for pat, repl in patterns:
        new = re.sub(pat, repl, full)
        if new != full:
            full = new
            changed = True
    if changed:
        # Clear all runs, set text vào run đầu
        first = para.runs[0]
        # Giữ format của run đầu
        first.text = full
        # Xoá các run còn lại
        for r in para.runs[1:]:
            r._element.getparent().remove(r._element)


def remove_paragraph(paragraph):
    """Xoá một paragraph khỏi doc."""
    el = paragraph._element
    el.getparent().remove(el)


def add_heading(doc, text, level=1, insert_before=None):
    """Thêm heading với level (1=14 bold uppercase, 2=13 bold, 3=13 bold italic)."""
    if insert_before is not None:
        new_el = OxmlElement("w:p")
        insert_before._element.addprevious(new_el)
        from docx.text.paragraph import Paragraph
        para = Paragraph(new_el, insert_before._parent)
    else:
        para = doc.add_paragraph()

    para.paragraph_format.first_line_indent = Cm(0)
    if level == 1:
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_before = Pt(0)
        para.paragraph_format.space_after = Pt(12)
        run = para.add_run(text.upper())
        set_font(run, 14, bold=True)
    elif level == 2:
        para.paragraph_format.space_before = Pt(12)
        para.paragraph_format.space_after = Pt(6)
        run = para.add_run(text)
        set_font(run, 13, bold=True)
    else:
        para.paragraph_format.space_before = Pt(6)
        para.paragraph_format.space_after = Pt(6)
        run = para.add_run(text)
        set_font(run, 13, bold=True, italic=True)
    return para


def add_body(doc, text, insert_before=None):
    """Thêm đoạn văn body 13pt justify."""
    if insert_before is not None:
        new_el = OxmlElement("w:p")
        insert_before._element.addprevious(new_el)
        from docx.text.paragraph import Paragraph
        para = Paragraph(new_el, insert_before._parent)
    else:
        para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    para.paragraph_format.first_line_indent = Cm(1)
    para.paragraph_format.line_spacing = 1.5
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.space_after = Pt(6)

    # Parse inline **bold** / *italic* / `code`
    pattern = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")
    parts = pattern.split(text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = para.add_run(part[2:-2]); set_font(r, 13, bold=True)
        elif part.startswith("*") and part.endswith("*") and not part.startswith("**"):
            r = para.add_run(part[1:-1]); set_font(r, 13, italic=True)
        elif part.startswith("`") and part.endswith("`"):
            r = para.add_run(part[1:-1])
            r.font.name = "Consolas"; r.font.size = Pt(11)
        else:
            r = para.add_run(part); set_font(r, 13)
    return para


def add_bullet(doc, text, insert_before=None):
    if insert_before is not None:
        new_el = OxmlElement("w:p")
        insert_before._element.addprevious(new_el)
        from docx.text.paragraph import Paragraph
        para = Paragraph(new_el, insert_before._parent)
        para.style = doc.styles["List Bullet"]
    else:
        para = doc.add_paragraph(style="List Bullet")
    para.paragraph_format.first_line_indent = Cm(0)
    para.paragraph_format.left_indent = Cm(1)
    pattern = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*)")
    for part in pattern.split(text):
        if not part: continue
        if part.startswith("**") and part.endswith("**"):
            r = para.add_run(part[2:-2]); set_font(r, 13, bold=True)
        elif part.startswith("*") and part.endswith("*"):
            r = para.add_run(part[1:-1]); set_font(r, 13, italic=True)
        else:
            r = para.add_run(part); set_font(r, 13)
    return para


def add_table_from_md(doc, md_lines, insert_before=None):
    rows = []
    for line in md_lines:
        line = line.strip()
        if not line or set(line) <= set("|-: "):
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            rows.append(cells)
    if not rows:
        return
    n_rows = len(rows); n_cols = max(len(r) for r in rows)

    if insert_before is not None:
        # Thêm table trước một paragraph
        # Cách đơn giản: add_table ở cuối rồi di chuyển
        table = doc.add_table(rows=n_rows, cols=n_cols)
        # Di chuyển element lên trên insert_before
        insert_before._element.addprevious(table._element)
    else:
        table = doc.add_table(rows=n_rows, cols=n_cols)

    table.style = "Light Grid Accent 1"
    for i, row in enumerate(rows):
        for j in range(n_cols):
            cell = table.cell(i, j)
            text = row[j] if j < len(row) else ""
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.line_spacing = 1.0
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            parts = re.split(r"(\*\*[^*]+\*\*)", text)
            for part in parts:
                if not part: continue
                if part.startswith("**") and part.endswith("**"):
                    r = p.add_run(part[2:-2]); set_font(r, 12, bold=True)
                else:
                    r = p.add_run(part); set_font(r, 12, bold=(i == 0))


def add_figure(doc, img_path, caption, insert_before=None):
    """Chèn ảnh centered + caption bên dưới."""
    if insert_before is not None:
        new_el = OxmlElement("w:p")
        insert_before._element.addprevious(new_el)
        from docx.text.paragraph import Paragraph
        para = Paragraph(new_el, insert_before._parent)
    else:
        para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.first_line_indent = Cm(0)
    run = para.add_run()
    try:
        run.add_picture(str(img_path), width=Cm(15))
    except Exception as e:
        run.add_text(f"[Lỗi chèn ảnh {img_path.name}: {e}]")

    # Caption
    if insert_before is not None:
        new_el2 = OxmlElement("w:p")
        insert_before._element.addprevious(new_el2)
        from docx.text.paragraph import Paragraph
        cap = Paragraph(new_el2, insert_before._parent)
    else:
        cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.first_line_indent = Cm(0)
    cap.paragraph_format.space_after = Pt(12)
    r = cap.add_run(caption); set_font(r, 12, italic=True)


def process_markdown_append(doc, md_file, insert_before=None):
    """Parse markdown và insert vào doc. insert_before=None → append at end."""
    content = md_file.read_text(encoding="utf-8")
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # Fenced code
        if line.startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].rstrip().startswith("```"):
                code_lines.append(lines[i]); i += 1
            i += 1
            if insert_before is not None:
                p = insert_before._element.addprevious(OxmlElement("w:p"))
                from docx.text.paragraph import Paragraph
                para = Paragraph(p, doc)
            else:
                para = doc.add_paragraph()
            para.paragraph_format.first_line_indent = Cm(0)
            para.paragraph_format.left_indent = Cm(1)
            para.paragraph_format.line_spacing = 1.0
            r = para.add_run("\n".join(code_lines))
            r.font.name = "Consolas"; r.font.size = Pt(10)
            continue

        if line.startswith("# "):
            add_heading(doc, line[2:].strip(), 1, insert_before=insert_before)
            i += 1; continue
        if line.startswith("## "):
            add_heading(doc, line[3:].strip(), 2, insert_before=insert_before)
            i += 1; continue
        if line.startswith("### "):
            add_heading(doc, line[4:].strip(), 3, insert_before=insert_before)
            i += 1; continue

        # Table
        if line.startswith("|"):
            tbl = []
            while i < len(lines) and lines[i].rstrip().startswith("|"):
                tbl.append(lines[i].rstrip()); i += 1
            add_table_from_md(doc, tbl, insert_before=insert_before)
            continue

        # Bullet
        if line.startswith("- ") or line.startswith("• "):
            add_bullet(doc, line[2:].strip(), insert_before=insert_before)
            i += 1; continue

        # Numbered
        if re.match(r"^\d+\.\s", line):
            m = re.match(r"^\d+\.\s(.+)", line)
            add_bullet(doc, m.group(1), insert_before=insert_before)
            i += 1; continue

        # Figure placeholder
        fig_match = re.match(r"^\*\*(Hình \d+\.\d+)\*\*\s*(.*)$", line)
        if fig_match:
            fig_num = fig_match.group(1); body = fig_match.group(2).strip()
            caption = f"{fig_num}  {body}" if body else fig_num
            img_file = FIGURE_IMAGES.get(fig_num)
            if img_file and (IMAGES / img_file).exists():
                add_figure(doc, IMAGES / img_file, caption, insert_before=insert_before)
            i += 1; continue

        # Empty
        if not line.strip():
            i += 1; continue

        # Normal para — gộp nhiều dòng
        para_lines = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "- ", "• ", "```")) \
              and not re.match(r"^\d+\.\s", lines[i]) \
              and not re.match(r"^\*\*(Hình|Bảng) \d+\.\d+\*\*", lines[i]):
            para_lines.append(lines[i].rstrip()); i += 1
        add_body(doc, " ".join(para_lines), insert_before=insert_before)


def main():
    doc = Document(str(SRC))

    print(f"[1/4] Đang quét {len(doc.paragraphs)} paragraph...")

    # Step 1: Xoá các đoạn COMMENT màu đỏ C00000 của cô
    to_remove = []
    for p in doc.paragraphs:
        if is_teacher_comment(p):
            to_remove.append(p)
    print(f"      Phát hiện {len(to_remove)} đoạn comment đỏ của cô → xoá")
    for p in to_remove:
        remove_paragraph(p)

    # Step 2: Apply text replacements
    print(f"[2/4] Apply {len(REPLACEMENTS)} replacement patterns...")
    replace_count = 0
    for p in doc.paragraphs:
        before = p.text
        apply_text_replacements(p, REPLACEMENTS)
        if p.text != before:
            replace_count += 1
    print(f"      Đã sửa {replace_count} đoạn")

    # Step 3: Tìm paragraph "TÀI LIỆU THAM KHẢO" để insert chương mới TRƯỚC nó
    tldk_para = None
    for p in doc.paragraphs:
        if p.text.strip().upper().startswith("TÀI LIỆU THAM KHẢO"):
            tldk_para = p
            break

    if tldk_para is None:
        print("      [WARN] Không tìm thấy 'TÀI LIỆU THAM KHẢO' → append cuối file")
    else:
        print(f"[3/4] Tìm thấy 'TÀI LIỆU THAM KHẢO' → insert chương 4-5-kết luận trước đó")

    # Step 4: Append Ch4, Ch5, Conclusion
    print(f"[4/4] Append {len(CHAPTERS_TO_APPEND)} chương mới...")
    for md in CHAPTERS_TO_APPEND:
        print(f"      + {md.name}")
        process_markdown_append(doc, md, insert_before=tldk_para)

    # Save
    doc.save(str(OUT))
    import os
    print(f"\n✓ Đã lưu: {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")

    # Verify
    doc2 = Document(str(OUT))
    imgs = sum(1 for p in doc2.paragraphs for r in p.runs
               if r._element.findall(".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip"))
    tables = len(doc2.tables)
    print(f"  → {len(doc2.paragraphs)} paragraphs, {tables} tables, {imgs} images")


if __name__ == "__main__":
    main()
