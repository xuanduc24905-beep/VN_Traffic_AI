"""Rebuild file tiểu luận từ bản gốc của cô:

Chiến lược: GIỮ front matter (cover, TOC, DMH, DMB) + TLTK + styles gốc;
THAY toàn bộ Ch1-3 bằng bản revised; CHÈN Ch4-5-Kết luận mới.

Đầu ra: Nhom02_PhanLoaiPhuongTien_UPDATED.docx (overwrite)
"""
from pathlib import Path
import re
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph

ROOT = Path("/home/xuand/vn_traffic_ai/docs/tieuluan_V3")
SRC = ROOT / "Nhom02_PhanLoaiPhuongTien_NhanXet_GVHD.docx"
OUT = ROOT / "Nhom02_PhanLoaiPhuongTien_UPDATED.docx"
IMAGES = ROOT / "images"

CHAPTERS = [
    ROOT / "chapter1_revised.md",
    ROOT / "chapter2_revised.md",
    ROOT / "chapter3_revised.md",
    ROOT / "chapter4_new.md",
    ROOT / "chapter5_new.md",
    ROOT / "conclusion_new.md",
]

FIGURE_IMAGES = {
    # Ch2 lý thuyết — ảnh gốc từ docx của cô
    "Hình 2.1": "fig_2_1_original.png",   # kernel 3x3
    "Hình 2.2": "fig_2_2_original.png",   # ResNet residual
    "Hình 2.3": "fig_2_3_original.png",   # 3 nhiệm vụ classification/detection/segmentation
    "Hình 2.5": "fig_2_5_original.png",   # YOLOv8 architecture
    "Hình 2.6": "fig_2_6_original.png",   # Faster R-CNN architecture
    "Hình 2.7": "fig_2_7_original.png",   # IoU minh họa
    # Ch3 pipeline + samples
    "Hình 3.1": "fig_3_pipeline.png",
    "Hình 3.2": "fig_3_sample_soict_night.jpg",
    "Hình 3.3": "fig_3_sample_soict_2024.jpg",
    "Hình 3.4": "fig_3_sample_cantho.jpg",
    "Hình 3.5": "fig_3_sample_vnv3.jpg",
    "Hình 3.6": "fig_3_sample_v10.jpg",
    "Hình 3.7": "fig_3_7_original.png",   # YOLO coord system
    # Ch4 charts
    "Hình 4.1": "fig_4_training_curves_v5plus.png",
    "Hình 4.2": "fig_4_confusion_matrix_p4scale.png",
    "Hình 4.5": "fig_4_model_comparison.png",
    "Hình 4.6": "fig_4_per_class_ap.png",
    "Hình 4.7": "fig_4_yolo_vs_frcnn.png",
    "Hình 4.8": "fig_4_fps_comparison.png",
    "Hình 4.9": "fig_4_val_test_gap.png",
    "Hình 4.10": "fig_4_counting.png",
    # Ch5 UI
    "Hình 5.1": "fig_5_streamlit_mockup.png",
    "Hình 5.2": "fig_5_streamlit_mockup.png",
}

RED_HEX = "C00000"


def is_teacher_comment(p):
    for r in p.runs:
        rPr = r._element.find(qn("w:rPr"))
        if rPr is None: continue
        c = rPr.find(qn("w:color"))
        if c is not None and c.get(qn("w:val"), "").upper() == RED_HEX:
            return True
    return False


def set_font(run, size=13, bold=False, italic=False):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size); run.bold = bold; run.italic = italic
    rpr = run._element.get_or_add_rPr()
    rFonts = rpr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts"); rpr.append(rFonts)
    for k in ("ascii", "hAnsi", "cs", "eastAsia"):
        rFonts.set(qn(f"w:{k}"), "Times New Roman")


def insert_paragraph_before(before_para, doc):
    """Insert a new empty paragraph trước before_para, trả về Paragraph."""
    new_el = OxmlElement("w:p")
    before_para._element.addprevious(new_el)
    return Paragraph(new_el, before_para._parent)


def add_heading(before_para, doc, text, level=1):
    para = insert_paragraph_before(before_para, doc)
    # ÁP DỤNG Word Heading style thật → TOC auto-detect được
    try:
        style_name = {1: "Heading 1", 2: "Heading 2", 3: "Heading 3"}.get(level, "Heading 4")
        para.style = doc.styles[style_name]
    except KeyError:
        pass
    para.paragraph_format.first_line_indent = Cm(0)
    if level == 1:
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_before = Pt(0)
        para.paragraph_format.space_after = Pt(12)
        r = para.add_run(text.upper()); set_font(r, 14, bold=True)
    elif level == 2:
        para.paragraph_format.space_before = Pt(12); para.paragraph_format.space_after = Pt(6)
        r = para.add_run(text); set_font(r, 13, bold=True)
    elif level == 3:
        para.paragraph_format.space_before = Pt(6); para.paragraph_format.space_after = Pt(6)
        r = para.add_run(text); set_font(r, 13, bold=True, italic=True)
    else:
        r = para.add_run(text); set_font(r, 13, bold=True, italic=True)


def add_body(before_para, doc, text):
    para = insert_paragraph_before(before_para, doc)
    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    para.paragraph_format.first_line_indent = Cm(1)
    para.paragraph_format.line_spacing = 1.5
    para.paragraph_format.space_after = Pt(6)
    pat = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")
    for part in pat.split(text):
        if not part: continue
        if part.startswith("**") and part.endswith("**"):
            r = para.add_run(part[2:-2]); set_font(r, 13, bold=True)
        elif part.startswith("*") and part.endswith("*") and not part.startswith("**"):
            r = para.add_run(part[1:-1]); set_font(r, 13, italic=True)
        elif part.startswith("`") and part.endswith("`"):
            r = para.add_run(part[1:-1])
            r.font.name = "Consolas"; r.font.size = Pt(11)
        else:
            r = para.add_run(part); set_font(r, 13)


def add_bullet(before_para, doc, text):
    para = insert_paragraph_before(before_para, doc)
    try:
        para.style = doc.styles["List Bullet"]
    except KeyError:
        pass
    para.paragraph_format.first_line_indent = Cm(0)
    para.paragraph_format.left_indent = Cm(1)
    pat = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*)")
    for part in pat.split(text):
        if not part: continue
        if part.startswith("**") and part.endswith("**"):
            r = para.add_run(part[2:-2]); set_font(r, 13, bold=True)
        elif part.startswith("*") and part.endswith("*"):
            r = para.add_run(part[1:-1]); set_font(r, 13, italic=True)
        else:
            r = para.add_run(part); set_font(r, 13)


def add_code_block(before_para, doc, code_lines):
    para = insert_paragraph_before(before_para, doc)
    para.paragraph_format.first_line_indent = Cm(0)
    para.paragraph_format.left_indent = Cm(1)
    para.paragraph_format.line_spacing = 1.0
    r = para.add_run("\n".join(code_lines))
    r.font.name = "Consolas"; r.font.size = Pt(10)


def add_table(before_para, doc, md_lines):
    rows = []
    for line in md_lines:
        line = line.strip()
        if not line or set(line) <= set("|-: "): continue
        if line.startswith("|") and line.endswith("|"):
            rows.append([c.strip() for c in line.strip("|").split("|")])
    if not rows: return
    n_rows, n_cols = len(rows), max(len(r) for r in rows)
    # Create table at end of doc, then move trước before_para
    table = doc.add_table(rows=n_rows, cols=n_cols)
    before_para._element.addprevious(table._element)
    try:
        table.style = "Light Grid Accent 1"
    except Exception:
        pass
    for i, row in enumerate(rows):
        for j in range(n_cols):
            cell = table.cell(i, j)
            text = row[j] if j < len(row) else ""
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.line_spacing = 1.0
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for part in re.split(r"(\*\*[^*]+\*\*)", text):
                if not part: continue
                if part.startswith("**") and part.endswith("**"):
                    r = p.add_run(part[2:-2]); set_font(r, 12, bold=True)
                else:
                    r = p.add_run(part); set_font(r, 12, bold=(i == 0))


def add_figure(before_para, doc, img_path, caption):
    para = insert_paragraph_before(before_para, doc)
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.first_line_indent = Cm(0)
    r = para.add_run()
    try:
        r.add_picture(str(img_path), width=Cm(15))
    except Exception as e:
        r.add_text(f"[Lỗi ảnh {img_path.name}: {e}]")
    cap = insert_paragraph_before(before_para, doc)
    # Caption style để danh mục hình/bảng tự detect
    try:
        cap.style = doc.styles["Caption"]
    except KeyError:
        pass
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.first_line_indent = Cm(0)
    cap.paragraph_format.space_after = Pt(12)
    r2 = cap.add_run(caption); set_font(r2, 12, italic=True)


def insert_md(before_para, doc, md_file):
    """Parse md và insert nội dung TRƯỚC before_para."""
    content = md_file.read_text(encoding="utf-8")
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if line.startswith("```"):
            i += 1; code = []
            while i < len(lines) and not lines[i].rstrip().startswith("```"):
                code.append(lines[i]); i += 1
            i += 1
            add_code_block(before_para, doc, code); continue

        if line.startswith("# "):
            add_heading(before_para, doc, line[2:].strip(), 1); i += 1; continue
        if line.startswith("## "):
            add_heading(before_para, doc, line[3:].strip(), 2); i += 1; continue
        if line.startswith("### "):
            add_heading(before_para, doc, line[4:].strip(), 3); i += 1; continue

        if line.startswith("|"):
            tbl = []
            while i < len(lines) and lines[i].rstrip().startswith("|"):
                tbl.append(lines[i].rstrip()); i += 1
            add_table(before_para, doc, tbl); continue

        if line.startswith("- ") or line.startswith("• "):
            add_bullet(before_para, doc, line[2:].strip()); i += 1; continue

        if re.match(r"^\d+\.\s", line):
            m = re.match(r"^\d+\.\s(.+)", line)
            add_bullet(before_para, doc, m.group(1)); i += 1; continue

        fig = re.match(r"^\*\*(Hình \d+\.\d+)\*\*\s*(.*)$", line)
        if fig:
            num, body = fig.group(1), fig.group(2).strip()
            caption = f"{num}  {body}" if body else num
            img = FIGURE_IMAGES.get(num)
            if img and (IMAGES / img).exists():
                add_figure(before_para, doc, IMAGES / img, caption)
            i += 1; continue

        if not line.strip():
            i += 1; continue

        para_lines = [line]; i += 1
        while i < len(lines) and lines[i].strip() \
              and not lines[i].startswith(("#", "|", "- ", "• ", "```")) \
              and not re.match(r"^\d+\.\s", lines[i]) \
              and not re.match(r"^\*\*(Hình|Bảng) \d+\.\d+\*\*", lines[i]):
            para_lines.append(lines[i].rstrip()); i += 1
        add_body(before_para, doc, " ".join(para_lines))


def find_heading_by_text(doc, start_substring, must_be_heading=True):
    """Tìm paragraph là Heading 1 bắt đầu bằng text cho trước. Trả về index hoặc None."""
    for i, p in enumerate(doc.paragraphs):
        text = p.text.strip().upper()
        if text.startswith(start_substring.upper()):
            if not must_be_heading or p.style.name.startswith("Heading"):
                return i
    return None


def delete_range(doc, start_idx, end_idx_exclusive):
    """Xoá các paragraph từ start_idx đến end_idx_exclusive (không bao gồm) + bất kỳ table nào giữa."""
    # Collect XML elements trong khoảng
    # Note: cần xoá cả table (w:tbl) nằm giữa các paragraph
    start_p = doc.paragraphs[start_idx]._element
    # Tìm end element
    if end_idx_exclusive < len(doc.paragraphs):
        end_p = doc.paragraphs[end_idx_exclusive]._element
    else:
        end_p = None
    # Walk từ start, xoá cho tới khi gặp end
    cur = start_p
    while cur is not None and cur is not end_p:
        nxt = cur.getnext()
        cur.getparent().remove(cur)
        cur = nxt


def main():
    doc = Document(str(SRC))
    print(f"Load gốc: {len(doc.paragraphs)} paragraphs, {len(doc.tables)} tables")

    # Step 1a: Xoá comment đỏ
    red = [p for p in doc.paragraphs if is_teacher_comment(p)]
    for p in red:
        p._element.getparent().remove(p._element)
    print(f"[1a] Xoá {len(red)} đoạn comment đỏ của cô")

    # Step 1b: Sửa access date trong TLTK (Nov 15 → Oct 15, 2026)
    import re as _re
    date_fixed = 0
    for p in doc.paragraphs:
        new = _re.sub(r"Accessed:\s*Nov\.\s*15,\s*2026", "Accessed: Oct. 15, 2026", p.text)
        if new != p.text and p.runs:
            p.runs[0].text = new
            for r in p.runs[1:]:
                r.text = ""
            date_fixed += 1
    print(f"[1b] Sửa {date_fixed} ngày truy cập Nov 15 → Oct 15, 2026")

    # Step 2: Tìm index Ch1 và TLTK
    ch1_idx = find_heading_by_text(doc, "CHƯƠNG 1")
    tltk_idx = find_heading_by_text(doc, "TÀI LIỆU THAM KHẢO")
    if ch1_idx is None or tltk_idx is None:
        raise SystemExit(f"Không tìm thấy ranh giới: Ch1={ch1_idx}, TLTK={tltk_idx}")
    print(f"[2] Ranh giới: Ch1 at para {ch1_idx}, TLTK at para {tltk_idx}")

    # Step 3: Xoá [Ch1_start ... TLTK_start) — toàn bộ Ch1-3 cũ
    print(f"[3] Xoá từ para {ch1_idx} đến {tltk_idx-1} (giữ nguyên trước Ch1 và từ TLTK trở đi)")
    # Lấy element của TLTK để làm anchor
    tltk_para = doc.paragraphs[tltk_idx]
    # Lấy element cuối cùng cần giữ (trước Ch1)
    # Xoá từ Ch1 đến ngay trước TLTK
    ch1_para_elem = doc.paragraphs[ch1_idx]._element
    cur = ch1_para_elem
    end_elem = tltk_para._element
    deleted = 0
    while cur is not None and cur is not end_elem:
        nxt = cur.getnext()
        cur.getparent().remove(cur); deleted += 1
        cur = nxt
    print(f"    Đã xoá {deleted} elements (paragraph + table) giữa Ch1 cũ và TLTK")

    # Step 4: Chèn 6 chương mới TRƯỚC TLTK
    print(f"[4] Chèn 6 chương mới trước TLTK:")
    for md in CHAPTERS:
        print(f"    + {md.name}")
        insert_md(tltk_para, doc, md)

    # Step 5: Thêm [31] Streamlit vào cuối danh mục TLTK
    print(f"[5] Thêm [31] Streamlit vào TLTK")
    streamlit_ref = ('[31]\tStreamlit Inc., "Streamlit — A faster way to build and share data apps," '
                     'Streamlit Documentation. Accessed: Nov. 14, 2026. [Online]. '
                     'Available: https://docs.streamlit.io/')
    # Append paragraph at end of doc (after TLTK references)
    last = doc.add_paragraph()
    last.paragraph_format.first_line_indent = Cm(0)
    last.paragraph_format.line_spacing = 1.5
    last.paragraph_format.space_after = Pt(6)
    r = last.add_run(streamlit_ref); set_font(r, 13)

    # Save
    doc.save(str(OUT))
    import os
    print(f"\n✓ Lưu: {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")

    # Verify
    d2 = Document(str(OUT))
    imgs = sum(1 for p in d2.paragraphs for r in p.runs
               if r._element.findall(".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip"))
    print(f"  → {len(d2.paragraphs)} paragraphs, {len(d2.tables)} tables, {imgs} images")

    # Check các câu chữ quan trọng
    text_full = "\n".join(p.text for p in d2.paragraphs)
    checks = [
        ("V5+", "V5+" in text_full),
        ("10.203 ảnh", "10.203" in text_full or "10,203" in text_full),
        ("5 nguồn Roboflow", "5 nguồn" in text_full or "năm nguồn" in text_full),
        ("Vietnam Vehicles v10", "v10" in text_full.lower() or "visionvn" in text_full.lower()),
        ("p4_scale", "p4_scale" in text_full),
        ("Faster R-CNN cấu hình", "Faster R-CNN" in text_full and "ResNet50-FPN" in text_full),
        ("CHƯƠNG 4", "CHƯƠNG 4" in text_full.upper()),
        ("CHƯƠNG 5", "CHƯƠNG 5" in text_full.upper()),
        ("KẾT LUẬN", "KẾT LUẬN" in text_full.upper()),
        ("TÀI LIỆU THAM KHẢO", "TÀI LIỆU THAM KHẢO" in text_full.upper()),
        ("không còn 'năm lớp'", "năm lớp" not in text_full.lower()),
    ]
    print("\nKiểm tra nội dung:")
    for name, ok in checks:
        print(f"  {'✓' if ok else '✗'} {name}")


if __name__ == "__main__":
    main()
