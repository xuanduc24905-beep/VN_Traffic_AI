#!/usr/bin/env python3
"""Convert BAO_CAO_PHAN_TICH_DU_AN.md to DOCX with real images embedded.

Hand-rolled converter (not pandoc) so we keep tight control over styling,
tables, image sizing, and Vietnamese typography.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Cm, Pt, RGBColor
from PIL import Image

REPORT_DIR = Path("/home/xuand/vn_traffic_ai/docs/report")
MD_PATH = REPORT_DIR / "BAO_CAO_PHAN_TICH_DU_AN.md"
OUT_PATH = REPORT_DIR / "BAO_CAO_PHAN_TICH_DU_AN.docx"

MAX_IMG_WIDTH_CM = 15.5  # fits A4 with normal margins

# ---------------- Inline formatting ----------------

INLINE_PATTERNS = [
    (re.compile(r"\*\*(.+?)\*\*"), "bold"),
    (re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)"), "italic"),
    (re.compile(r"`([^`]+)`"), "code"),
]
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def add_inline(paragraph, text):
    """Add text with inline bold/italic/code to a paragraph. Strip links -> keep label."""
    # Strip markdown link syntax, keep label only
    text = LINK_RE.sub(lambda m: m.group(1), text)

    # Tokenize by first matching pattern repeatedly
    pos = 0
    while pos < len(text):
        best = None
        for pat, kind in INLINE_PATTERNS:
            m = pat.search(text, pos)
            if m and (best is None or m.start() < best[0].start()):
                best = (m, kind)
        if best is None:
            run = paragraph.add_run(text[pos:])
            break
        m, kind = best
        if m.start() > pos:
            paragraph.add_run(text[pos:m.start()])
        run = paragraph.add_run(m.group(1))
        if kind == "bold":
            run.bold = True
        elif kind == "italic":
            run.italic = True
        elif kind == "code":
            run.font.name = "Consolas"
            run.font.size = Pt(10)
            # light shading
            run.font.color.rgb = RGBColor(0xB0, 0x30, 0x30)
        pos = m.end()


# ---------------- Block parsing ----------------

IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
TABLE_SEP_RE = re.compile(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$")


def iter_blocks(lines):
    """Yield blocks: ('heading',lvl,txt) ('para',txt) ('image',alt,path)
    ('code',lang,code) ('table',rows) ('bullet',items) ('quote',txt) ('hr',)
    ('num',items)."""
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip("\n")

        if not line.strip():
            i += 1
            continue

        # HR
        if re.match(r"^-{3,}\s*$", line):
            yield ("hr",)
            i += 1
            continue

        # Heading
        m = HEADING_RE.match(line)
        if m:
            yield ("heading", len(m.group(1)), m.group(2).strip())
            i += 1
            continue

        # Code fence
        if line.startswith("```"):
            lang = line[3:].strip()
            buf = []
            i += 1
            while i < n and not lines[i].startswith("```"):
                buf.append(lines[i].rstrip("\n"))
                i += 1
            i += 1  # skip closing
            yield ("code", lang, "\n".join(buf))
            continue

        # Blockquote
        if line.lstrip().startswith(">"):
            buf = []
            while i < n and lines[i].lstrip().startswith(">"):
                buf.append(lines[i].lstrip()[1:].strip())
                i += 1
            yield ("quote", "\n".join(buf))
            continue

        # Image only on a line
        m = IMG_RE.match(line.strip())
        if m and IMG_RE.sub("", line.strip()) == "":
            yield ("image", m.group(1), m.group(2))
            i += 1
            continue

        # Table — header pipe row + separator
        if line.lstrip().startswith("|") and i + 1 < n and TABLE_SEP_RE.match(lines[i + 1].strip()):
            rows = []
            while i < n and lines[i].lstrip().startswith("|"):
                row = lines[i].strip()
                if TABLE_SEP_RE.match(row):
                    i += 1
                    continue
                cells = [c.strip() for c in row.strip("|").split("|")]
                rows.append(cells)
                i += 1
            yield ("table", rows)
            continue

        # Bullet list
        if re.match(r"^\s*[-*]\s+", line):
            items = []
            while i < n and re.match(r"^\s*[-*]\s+", lines[i]):
                items.append(re.sub(r"^\s*[-*]\s+", "", lines[i]).rstrip("\n"))
                i += 1
            yield ("bullet", items)
            continue

        # Numbered list
        if re.match(r"^\s*\d+\.\s+", line):
            items = []
            while i < n and re.match(r"^\s*\d+\.\s+", lines[i]):
                items.append(re.sub(r"^\s*\d+\.\s+", "", lines[i]).rstrip("\n"))
                i += 1
            yield ("num", items)
            continue

        # Paragraph — accumulate until blank line / block boundary
        buf = []
        while i < n and lines[i].strip() and not HEADING_RE.match(lines[i]) \
                and not lines[i].startswith("```") \
                and not re.match(r"^-{3,}\s*$", lines[i]) \
                and not lines[i].lstrip().startswith(">") \
                and not (lines[i].lstrip().startswith("|") and i + 1 < n and TABLE_SEP_RE.match(lines[i + 1].strip())) \
                and not re.match(r"^\s*[-*]\s+", lines[i]) \
                and not re.match(r"^\s*\d+\.\s+", lines[i]) \
                and not (IMG_RE.match(lines[i].strip()) and IMG_RE.sub("", lines[i].strip()) == ""):
            buf.append(lines[i].rstrip("\n"))
            i += 1
        yield ("para", " ".join(buf))


# ---------------- Rendering ----------------

def shade_cell(cell, color_hex):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color_hex)
    tc_pr.append(shd)


def set_cell_border(cell):
    tc_pr = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        b = OxmlElement(f"w:{edge}")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), "4")
        b.set(qn("w:color"), "888888")
        tcBorders.append(b)
    tc_pr.append(tcBorders)


def add_heading(doc, level, text):
    if level == 1:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(20)
        r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x68)
    else:
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.bold = True
        if level == 2:
            r.font.size = Pt(15)
            r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x68)
            # Page break before main chapters (except first)
        elif level == 3:
            r.font.size = Pt(12.5)
            r.font.color.rgb = RGBColor(0x2E, 0x5B, 0x9C)
        else:
            r.font.size = Pt(11.5)
            r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)


def add_image(doc, alt, rel_path):
    abs_path = (REPORT_DIR / rel_path).resolve()
    if not abs_path.exists():
        p = doc.add_paragraph()
        r = p.add_run(f"[ẢNH THIẾU: {rel_path}]")
        r.italic = True
        r.font.color.rgb = RGBColor(0xAA, 0x00, 0x00)
        return

    # Compute width (respect max)
    with Image.open(abs_path) as im:
        w_px, h_px = im.size
    # Assume 96 dpi -> cm
    width_cm = min(MAX_IMG_WIDTH_CM, w_px / 96 * 2.54)
    # For small charts, don't blow up
    if w_px < 400:
        width_cm = min(width_cm, 10)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(abs_path), width=Cm(width_cm))

    if alt:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cap.add_run(f"Hình: {alt}")
        r.italic = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)


def add_code(doc, lang, code):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.3)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    # grey shading via pPr
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), "F2F2F2")
    pPr.append(shd)
    r = p.add_run(code)
    r.font.name = "Consolas"
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)


def add_table(doc, rows):
    if not rows:
        return
    ncols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < ncols:
            r.append("")
    tbl = doc.add_table(rows=len(rows), cols=ncols)
    tbl.style = "Light Grid Accent 1"
    tbl.autofit = True
    for ri, row in enumerate(rows):
        for ci, cell_txt in enumerate(row):
            cell = tbl.rows[ri].cells[ci]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            # clear default paragraph
            para = cell.paragraphs[0]
            para.text = ""
            add_inline(para, cell_txt)
            for run in para.runs:
                run.font.size = Pt(10)
            if ri == 0:
                for run in para.runs:
                    run.bold = True
                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                shade_cell(cell, "1F3A68")
    doc.add_paragraph()  # spacing after table


def add_quote(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.6)
    pPr = p._p.get_or_add_pPr()
    # left border
    pBdr = OxmlElement("w:pBdr")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), "18")
    left.set(qn("w:color"), "1F3A68")
    pBdr.append(left)
    pPr.append(pBdr)
    r = p.add_run(text)
    r.italic = True
    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)


def add_bullets(doc, items, numbered=False):
    style = "List Number" if numbered else "List Bullet"
    for item in items:
        p = doc.add_paragraph(style=style)
        add_inline(p, item)


def add_para(doc, text):
    if not text.strip():
        return
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    add_inline(p, text)


# ---------------- Main ----------------

def main():
    text = MD_PATH.read_text(encoding="utf-8")
    lines = text.split("\n")

    doc = Document()
    # Default font
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    # A4 margins
    for section in doc.sections:
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(2.2)
        section.right_margin = Cm(2.2)

    first_h2 = True
    for block in iter_blocks(lines):
        kind = block[0]
        if kind == "heading":
            lvl, txt = block[1], block[2]
            if lvl == 2 and not first_h2:
                doc.add_page_break()
            if lvl == 2:
                first_h2 = False
            add_heading(doc, lvl, txt)
        elif kind == "para":
            add_para(doc, block[1])
        elif kind == "image":
            add_image(doc, block[1], block[2])
        elif kind == "code":
            add_code(doc, block[1], block[2])
        elif kind == "table":
            add_table(doc, block[1])
        elif kind == "quote":
            add_quote(doc, block[1])
        elif kind == "bullet":
            add_bullets(doc, block[1], numbered=False)
        elif kind == "num":
            add_bullets(doc, block[1], numbered=True)
        elif kind == "hr":
            # subtle separator
            p = doc.add_paragraph()
            pPr = p._p.get_or_add_pPr()
            pBdr = OxmlElement("w:pBdr")
            bottom = OxmlElement("w:bottom")
            bottom.set(qn("w:val"), "single")
            bottom.set(qn("w:sz"), "6")
            bottom.set(qn("w:color"), "AAAAAA")
            pBdr.append(bottom)
            pPr.append(pBdr)

    doc.save(OUT_PATH)
    print(f"OK -> {OUT_PATH}  ({OUT_PATH.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
