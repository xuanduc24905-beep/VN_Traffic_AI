from __future__ import annotations

from pathlib import Path
import re
import shutil

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "docs/tieuluan_V3/Nhom02_PhanLoaiPhuongTien_UPDATED5 (1).docx"
BACKUP = ROOT / "docs/tieuluan_V3/Nhom02_PhanLoaiPhuongTien_UPDATED5 (1).before_FORMAT_STANDARD.docx"
INVISIBLE = "\u200b\u200c\u200d\u2060\ufeff\u00ad"


def set_rfonts(rpr, name: str) -> None:
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
        rfonts.set(qn(f"w:{attr}"), name)


def set_run_font(run, name: str, size: float) -> None:
    run.font.name = name
    run.font.size = Pt(size)
    set_rfonts(run._r.get_or_add_rPr(), name)


def set_style_font(style, name: str, size: float, bold=None, italic=None) -> None:
    style.font.name = name
    style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if italic is not None:
        style.font.italic = italic
    set_rfonts(style.element.get_or_add_rPr(), name)


def field_run(paragraph, instruction: str, display: str):
    begin = paragraph.add_run()
    fc = OxmlElement("w:fldChar")
    fc.set(qn("w:fldCharType"), "begin")
    fc.set(qn("w:dirty"), "true")
    begin._r.append(fc)

    instr_run = paragraph.add_run()
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = f" {instruction} "
    instr_run._r.append(instr)

    separate = paragraph.add_run()
    fc2 = OxmlElement("w:fldChar")
    fc2.set(qn("w:fldCharType"), "separate")
    separate._r.append(fc2)

    shown = paragraph.add_run(display)

    end = paragraph.add_run()
    fc3 = OxmlElement("w:fldChar")
    fc3.set(qn("w:fldCharType"), "end")
    end._r.append(fc3)
    return shown


def ensure_main_toc(doc: Document) -> None:
    title_index = next(i for i, p in enumerate(doc.paragraphs) if p.text.strip() == "MỤC LỤC")
    title = doc.paragraphs[title_index]
    # Do not duplicate a main TOC if one already exists.
    instrs = doc.element.body.xpath('.//w:instrText[contains(text(), "TOC \\o")]')
    if instrs:
        return
    if title_index + 1 < len(doc.paragraphs):
        p = doc.paragraphs[title_index + 1]
    else:
        new_p = OxmlElement("w:p")
        title._p.addnext(new_p)
        from docx.text.paragraph import Paragraph
        p = Paragraph(new_p, title._parent)
    p._p.clear_content()
    p.style = doc.styles["Normal"]
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.line_spacing = 1.0
    field_run(p, 'TOC \\o "1-3" \\h \\z \\u', "Mục lục sẽ được cập nhật khi mở tài liệu")


def set_update_fields(doc: Document) -> None:
    settings = doc.settings._element
    update = settings.find(qn("w:updateFields"))
    if update is None:
        update = OxmlElement("w:updateFields")
        settings.append(update)
    update.set(qn("w:val"), "true")
    for fld in doc.element.body.xpath('.//w:fldChar[@w:fldCharType="begin"]'):
        fld.set(qn("w:dirty"), "true")


def configure_styles(doc: Document) -> None:
    normal = doc.styles["Normal"]
    set_style_font(normal, "Times New Roman", 13)
    pf = normal.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.line_spacing = 1.5
    pf.space_before = Pt(0)
    pf.space_after = Pt(6)
    pf.first_line_indent = Cm(1)

    h1 = doc.styles["Heading 1"]
    set_style_font(h1, "Times New Roman", 14, bold=True, italic=False)
    pf = h1.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.line_spacing = 1.5
    pf.space_before = Pt(0)
    pf.space_after = Pt(12)
    pf.first_line_indent = Cm(0)
    pf.page_break_before = True
    pf.keep_with_next = True

    h2 = doc.styles["Heading 2"]
    set_style_font(h2, "Times New Roman", 13, bold=True, italic=False)
    pf = h2.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.line_spacing = 1.5
    pf.space_before = Pt(6)
    pf.space_after = Pt(0)
    pf.first_line_indent = Cm(0)
    pf.keep_with_next = True

    h3 = doc.styles["Heading 3"]
    set_style_font(h3, "Times New Roman", 13, bold=True, italic=True)
    pf = h3.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.line_spacing = 1.5
    pf.space_before = Pt(6)
    pf.space_after = Pt(0)
    pf.first_line_indent = Cm(0)
    pf.keep_with_next = True

    for name in ("Chu thich hinh", "Chu thich bang"):
        style = doc.styles[name]
        set_style_font(style, "Times New Roman", 12, bold=False, italic=False)
        pf = style.paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf.line_spacing = 1.0
        pf.first_line_indent = Cm(0)
        pf.space_before = Pt(0)
        pf.space_after = Pt(6)

    if "List Bullet" in [s.name for s in doc.styles]:
        st = doc.styles["List Bullet"]
        set_style_font(st, "Times New Roman", 13)
        st.paragraph_format.line_spacing = 1.5
        st.paragraph_format.space_before = Pt(0)
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    for level in range(1, 4):
        name = f"toc {level}"
        if name not in [s.name for s in doc.styles]:
            continue
        st = doc.styles[name]
        set_style_font(st, "Times New Roman", 13)
        st.paragraph_format.line_spacing = 1.0
        st.paragraph_format.space_before = Pt(0)
        st.paragraph_format.space_after = Pt(0)
        st.paragraph_format.tab_stops.clear_all()
        st.paragraph_format.tab_stops.add_tab_stop(
            Cm(16), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS
        )

    if "Hinh anh" in [s.name for s in doc.styles]:
        st = doc.styles["Hinh anh"]
        st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        st.paragraph_format.first_line_indent = Cm(0)
        st.paragraph_format.space_before = Pt(6)
        st.paragraph_format.space_after = Pt(0)
        st.paragraph_format.keep_with_next = True


def configure_sections(doc: Document) -> None:
    for i, section in enumerate(doc.sections):
        section.page_width = Cm(21)
        section.page_height = Cm(29.7)
        section.left_margin = Cm(3)
        section.right_margin = Cm(2)
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.header_distance = Cm(1)
        section.footer_distance = Cm(1)
        if i == 1:
            section.start_type = WD_SECTION_START.NEW_PAGE

        pg = section._sectPr.find(qn("w:pgNumType"))
        if pg is None:
            pg = OxmlElement("w:pgNumType")
            section._sectPr.append(pg)
        pg.set(qn("w:start"), "1")

        footer = section.footer
        footer.is_linked_to_previous = False
        p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        p._p.clear_content()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        shown = field_run(p, "PAGE \\* roman" if i == 0 else "PAGE", "i" if i == 0 else "1")
        set_run_font(shown, "Times New Roman", 13)


def clean_invisible_characters(doc: Document) -> None:
    translation = {ord(ch): None for ch in INVISIBLE}
    translation[ord("\u00a0")] = ord(" ")
    roots = [doc.element]
    for section in doc.sections:
        roots.extend([section.header._element, section.footer._element])
    for root in roots:
        for node in root.xpath('.//w:t | .//w:instrText'):
            if node.text:
                node.text = node.text.translate(translation)


def configure_paragraphs(doc: Document) -> None:
    front_titles = {"MỤC LỤC", "ABSTRACT", "TÓM TẮT", "LỜI CẢM ƠN", "NHẬN XÉT CỦA GIÁO VIÊN HƯỚNG DẪN", "DANH MỤC HÌNH", "DANH MỤC BẢNG", "DANH MỤC TỪ VIẾT TẮT"}
    reference_started = False
    for p in doc.paragraphs:
        text = p.text.strip()
        style = p.style.name
        has_math = bool(p._p.xpath('.//*[local-name()="oMath"]'))
        has_drawing = bool(p._p.xpath('.//*[local-name()="drawing"]'))

        if text == "TÀI LIỆU THAM KHẢO":
            reference_started = True

        if style == "Heading 1":
            for run in p.runs:
                set_run_font(run, "Times New Roman", 14)
                run.bold = True
                run.italic = False
        elif style == "Heading 2":
            for run in p.runs:
                set_run_font(run, "Times New Roman", 13)
                run.bold = True
                run.italic = False
        elif style == "Heading 3":
            for run in p.runs:
                set_run_font(run, "Times New Roman", 13)
                run.bold = True
                run.italic = True
        elif style in ("Chu thich hinh", "Chu thich bang"):
            for run in p.runs:
                set_run_font(run, "Times New Roman", 12)
                run.italic = False
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.first_line_indent = Cm(0)
        elif style == "toc 1":
            for run in p.runs:
                set_run_font(run, "Times New Roman", 13)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.first_line_indent = Cm(0)
        elif style == "List Bullet":
            p.paragraph_format.line_spacing = 1.5
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            for run in p.runs:
                set_run_font(run, "Times New Roman", 13)
        elif style == "Normal" and not has_drawing:
            if has_math and not text:
                p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(6)
                p.paragraph_format.first_line_indent = Cm(0)
            elif text.startswith("model = fasterrcnn_resnet50_fpn("):
                p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.first_line_indent = Cm(0)
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(6)
                for run in p.runs:
                    set_run_font(run, "Consolas", 11)
            elif reference_started and re.match(r"^\[\d+\]", text):
                # IEEE references use a hanging indent, but left alignment avoids
                # Word stretching short lines and URLs into very large spaces.
                p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(6)
                p.paragraph_format.left_indent = Cm(1)
                p.paragraph_format.first_line_indent = Cm(-1)
                for run in p.runs:
                    if "\t" in run.text:
                        run.text = run.text.replace("\t", " ", 1)
                    set_run_font(run, "Times New Roman", 13)
            else:
                if p.alignment not in (WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.RIGHT):
                    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                    p.paragraph_format.first_line_indent = Cm(1)
                else:
                    p.paragraph_format.first_line_indent = Cm(0)
                p.paragraph_format.line_spacing = 1.5
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(6)
                for run in p.runs:
                    if run.font.name == "Consolas":
                        set_run_font(run, "Consolas", 11)
                    else:
                        set_run_font(run, "Times New Roman", 13)

        if text in front_titles:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Cm(0)
            for run in p.runs:
                set_run_font(run, "Times New Roman", 14)
                run.bold = True

    # Remove the only visually padded signature line and normalize the YOLO schema sample.
    for p in doc.paragraphs:
        if p.text.strip() == "CHỮ KÝ CỦA GIẢNG VIÊN":
            for run in p.runs:
                if run.text:
                    run.text = run.text.strip()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Cm(0)
        if re.sub(r"\s+", " ", p.text.strip()) == "class_id cx cy w h":
            if not p._p.xpath('.//*[local-name()="oMath"]'):
                p.text = "class_id  cx  cy  w  h"
                for run in p.runs:
                    set_run_font(run, "Consolas", 11)
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.first_line_indent = Cm(0)


def configure_tables(doc: Document) -> None:
    # Table 0 is the cover-page layout and retains its designed title sizes.
    for table in doc.tables[1:]:
        for ri, row in enumerate(table.rows):
            trpr = row._tr.get_or_add_trPr()
            if trpr.find(qn("w:cantSplit")) is None:
                trpr.append(OxmlElement("w:cantSplit"))
            if ri == 0 and trpr.find(qn("w:tblHeader")) is None:
                trpr.append(OxmlElement("w:tblHeader"))
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.line_spacing = 1.0
                    p.paragraph_format.space_before = Pt(0)
                    p.paragraph_format.space_after = Pt(0)
                    p.paragraph_format.first_line_indent = Cm(0)
                    for run in p.runs:
                        set_run_font(run, "Times New Roman", 12)


def convert_caption_numbers_to_fields(doc: Document) -> None:
    pattern = re.compile(r"^(Hình|Bảng) (\d+)\.(\d+)\s+(.+)$", re.S)
    for p in doc.paragraphs:
        if p.style.name not in ("Chu thich hinh", "Chu thich bang"):
            continue
        m = pattern.match(p.text.strip())
        if not m:
            continue
        label, chapter, number, description = m.groups()
        p._p.clear_content()
        p.style = doc.styles["Chu thich hinh" if label == "Hình" else "Chu thich bang"]
        prefix = p.add_run(f"{label} {chapter}.")
        set_run_font(prefix, "Times New Roman", 12)
        seq_name = ("Hinh" if label == "Hình" else "Bang") + chapter
        shown = field_run(p, f"SEQ {seq_name} \\* ARABIC", number)
        set_run_font(shown, "Times New Roman", 12)
        suffix = p.add_run(f" {description}")
        set_run_font(suffix, "Times New Roman", 12)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.first_line_indent = Cm(0)


def validate_references(doc: Document) -> None:
    nums = []
    for p in doc.paragraphs:
        m = re.match(r"^\[(\d+)\]", p.text.strip())
        if m:
            nums.append(int(m.group(1)))
    if nums != list(range(1, len(nums) + 1)):
        raise RuntimeError(f"IEEE reference numbering is not sequential: {nums}")


def main() -> None:
    if not BACKUP.exists():
        shutil.copy2(DOCX, BACKUP)
    doc = Document(DOCX)
    clean_invisible_characters(doc)
    configure_styles(doc)
    configure_sections(doc)
    ensure_main_toc(doc)
    configure_paragraphs(doc)
    configure_tables(doc)
    convert_caption_numbers_to_fields(doc)
    validate_references(doc)
    set_update_fields(doc)
    doc.save(DOCX)
    print(f"Formatted {DOCX}")
    print(f"Backup    {BACKUP}")


if __name__ == "__main__":
    main()
