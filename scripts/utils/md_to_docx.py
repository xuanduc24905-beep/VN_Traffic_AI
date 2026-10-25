"""Convert handbook markdown → .docx với ảnh embed inline.

Dùng python-docx + markdown-it-py để parse chuẩn CommonMark + GFM table.
Chạy:
    python scripts/utils/md_to_docx.py docs/handbook/vn_traffic_ai_handbook.md
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from markdown_it import MarkdownIt
from markdown_it.token import Token


def add_hyperlink(paragraph, text, url):
    """Add a clickable hyperlink to a paragraph."""
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    part = paragraph.part
    r_id = part.relate_to(url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True)
    hyperlink = OxmlElement('w:hyperlink')
    hyperlink.set(qn('r:id'), r_id)
    new_run = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')
    rStyle = OxmlElement('w:rStyle')
    rStyle.set(qn('w:val'), 'Hyperlink')
    rPr.append(rStyle)
    new_run.append(rPr)
    t = OxmlElement('w:t')
    t.text = text
    new_run.append(t)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)


def render_inline(paragraph, inline_tokens, md_dir: Path):
    """Render inline tokens vào paragraph (text + bold + italic + code + link)."""
    bold = italic = code = False
    link_url = None
    link_text_buf = ""

    for tok in inline_tokens:
        t = tok.type
        if t == 'strong_open':   bold = True
        elif t == 'strong_close': bold = False
        elif t == 'em_open':      italic = True
        elif t == 'em_close':     italic = False
        elif t == 'code_inline':
            run = paragraph.add_run(tok.content)
            run.font.name = 'Consolas'
            run.font.size = Pt(10)
            run.font.color.rgb = RGBColor(0xC7, 0x25, 0x4E)
        elif t == 'link_open':
            link_url = tok.attrs.get('href', '')
            link_text_buf = ""
        elif t == 'link_close':
            # Convert relative repo path to clickable; keep http URL as-is
            if link_url:
                add_hyperlink(paragraph, link_text_buf or link_url, link_url)
            link_url = None
        elif t == 'text':
            if link_url is not None:
                link_text_buf += tok.content
            else:
                run = paragraph.add_run(tok.content)
                run.bold = bold
                run.italic = italic
        elif t == 'softbreak' or t == 'hardbreak':
            paragraph.add_run('\n')
        elif t == 'image':
            src = tok.attrs.get('src', '')
            img_path = (md_dir / src).resolve()
            if img_path.exists():
                try:
                    run = paragraph.add_run()
                    run.add_picture(str(img_path), width=Inches(6))
                except Exception as e:
                    paragraph.add_run(f"[image load error: {e}]")
            else:
                paragraph.add_run(f"[image not found: {src}]")


def convert(md_path: Path, out_path: Path):
    md = MarkdownIt("gfm-like").enable('table').enable('strikethrough')
    text = md_path.read_text()
    tokens = md.parse(text)

    doc = Document()

    # Set default font
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(11)

    md_dir = md_path.parent

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # Heading
        if tok.type == 'heading_open':
            level = int(tok.tag[1])
            inline = tokens[i+1]
            heading = doc.add_heading(level=min(level, 4))
            if inline.type == 'inline':
                render_inline(heading, inline.children or [], md_dir)
            i += 3; continue

        # Paragraph (may contain inline image)
        if tok.type == 'paragraph_open':
            inline = tokens[i+1]
            para = doc.add_paragraph()
            # Center-align if the paragraph contains ONLY an image
            if (inline.type == 'inline' and inline.children
                and len(inline.children) == 1
                and inline.children[0].type == 'image'):
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if inline.type == 'inline':
                render_inline(para, inline.children or [], md_dir)
            i += 3; continue

        # Fenced code block
        if tok.type == 'fence':
            para = doc.add_paragraph()
            run = para.add_run(tok.content.rstrip('\n'))
            run.font.name = 'Consolas'
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
            # Shade bg: skip (needs direct XML)
            i += 1; continue

        # Bullet list
        if tok.type == 'bullet_list_open':
            i += 1
            while tokens[i].type != 'bullet_list_close':
                if tokens[i].type == 'list_item_open':
                    i += 1
                    # Each list item may contain multiple paragraphs
                    while tokens[i].type != 'list_item_close':
                        if tokens[i].type == 'paragraph_open':
                            inline = tokens[i+1]
                            para = doc.add_paragraph(style='List Bullet')
                            if inline.type == 'inline':
                                render_inline(para, inline.children or [], md_dir)
                            i += 3
                        else:
                            i += 1
                    i += 1  # list_item_close
                else:
                    i += 1
            i += 1; continue

        # Ordered list
        if tok.type == 'ordered_list_open':
            i += 1
            while tokens[i].type != 'ordered_list_close':
                if tokens[i].type == 'list_item_open':
                    i += 1
                    while tokens[i].type != 'list_item_close':
                        if tokens[i].type == 'paragraph_open':
                            inline = tokens[i+1]
                            para = doc.add_paragraph(style='List Number')
                            if inline.type == 'inline':
                                render_inline(para, inline.children or [], md_dir)
                            i += 3
                        else:
                            i += 1
                    i += 1
                else:
                    i += 1
            i += 1; continue

        # Table
        if tok.type == 'table_open':
            # Collect rows
            rows = []
            current_row = []
            i += 1
            in_header = False
            while tokens[i].type != 'table_close':
                tt = tokens[i].type
                if tt == 'thead_open':      in_header = True
                elif tt == 'thead_close':   in_header = False
                elif tt == 'tr_open':       current_row = []
                elif tt == 'tr_close':      rows.append((in_header, current_row))
                elif tt in ('th_open', 'td_open'):
                    inline = tokens[i+1]
                    cell_text = ""
                    # Collect text of inline tokens for cells (plain text only)
                    if inline.type == 'inline' and inline.children:
                        for ch in inline.children:
                            if ch.type == 'text':
                                cell_text += ch.content
                            elif ch.type == 'code_inline':
                                cell_text += ch.content
                            elif ch.type == 'softbreak':
                                cell_text += ' '
                    current_row.append(cell_text)
                i += 1
            if rows:
                ncols = max(len(r[1]) for r in rows)
                table = doc.add_table(rows=len(rows), cols=ncols)
                table.style = 'Light Grid Accent 1'
                for r_idx, (hdr, cells) in enumerate(rows):
                    for c_idx, cell_text in enumerate(cells):
                        cell = table.cell(r_idx, c_idx)
                        cell.text = cell_text
                        if hdr:
                            for p in cell.paragraphs:
                                for run in p.runs:
                                    run.bold = True
            i += 1  # table_close
            continue

        # Horizontal rule
        if tok.type == 'hr':
            doc.add_paragraph('_' * 50)
            i += 1; continue

        # Blockquote
        if tok.type == 'blockquote_open':
            i += 1
            while tokens[i].type != 'blockquote_close':
                if tokens[i].type == 'paragraph_open':
                    inline = tokens[i+1]
                    para = doc.add_paragraph(style='Intense Quote')
                    if inline.type == 'inline':
                        render_inline(para, inline.children or [], md_dir)
                    i += 3
                else:
                    i += 1
            i += 1; continue

        i += 1

    doc.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    md_path = Path(sys.argv[1] if len(sys.argv) > 1
                   else "docs/handbook/vn_traffic_ai_handbook.md")
    out_path = md_path.with_suffix(".docx")
    convert(md_path, out_path)
