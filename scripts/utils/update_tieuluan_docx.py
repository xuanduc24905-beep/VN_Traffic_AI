"""Update docx tiểu luận: replace Chương 3 hiện tại + add Chương 4 mới.

Workflow:
1. Load docx gốc (Nhom_02_phanloaiphuongtienbangcnn.docx)
2. Tìm vị trí "CHƯƠNG 3" trong content body (không phải TOC)
3. Xoá toàn bộ paragraphs từ CHƯƠNG 3 đến TÀI LIỆU THAM KHẢO
4. Insert Chương 3 mới + Chương 4 mới từ file markdown
5. Save thành docx mới (giữ docx gốc nguyên vẹn)

Styles được giữ nguyên từ docx gốc (Heading 1, Heading 2, Normal, Table styles).
"""
from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = Path("/home/xuand/vn_traffic_ai")
SRC_DOCX = ROOT / "docs/tieuluan_V3/Nhom_02_phanloaiphuongtienbangcnn.docx"
NEW_DOCX = ROOT / "docs/tieuluan_V3/Nhom_02_phanloaiphuongtienbangcnn_v4.docx"
MD_FILE = ROOT / "docs/tieuluan_V3/chuong3_chuong4_restructured.md"


def find_marker_indices(doc: Document, markers: list[str]) -> dict[str, int]:
    """Find paragraph index for each marker (first occurrence with Heading 1 style)."""
    results = {}
    for i, p in enumerate(doc.paragraphs):
        text = p.text.strip()
        # Only accept if it's a Heading 1 style (actual chapter heading, not TOC entry)
        if p.style.name != "Heading 1":
            continue
        for m in markers:
            if m in results:
                continue
            if text.startswith(m):
                results[m] = i
    return results


def delete_paragraphs_range(doc: Document, start: int, end: int) -> None:
    """Delete paragraphs[start..end-1] by removing their XML elements."""
    paras = doc.paragraphs
    # Collect XML elements to remove
    to_remove = [paras[i]._element for i in range(start, end)]
    for el in to_remove:
        el.getparent().remove(el)


def add_paragraph_before(doc: Document, before_para, text: str,
                         style: str = "Normal") -> None:
    """Insert a new paragraph with text + style right before given paragraph."""
    new_para = before_para.insert_paragraph_before(text, style=style)
    return new_para


def parse_md_to_blocks(md_text: str) -> list[dict]:
    """Parse markdown to list of blocks: heading/paragraph/table/list.

    Each block is dict:
      {"type": "h1"|"h2"|"h3"|"para"|"table"|"list", "content": ...}
    """
    blocks = []
    lines = md_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        # Heading
        if line.startswith("# "):
            blocks.append({"type": "h1", "content": line[2:].strip()})
            i += 1
        elif line.startswith("## "):
            blocks.append({"type": "h2", "content": line[3:].strip()})
            i += 1
        elif line.startswith("### "):
            blocks.append({"type": "h3", "content": line[4:].strip()})
            i += 1
        # Horizontal rule → skip
        elif line.strip() == "---":
            i += 1
        # Code block → treat as plain paragraph
        elif line.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # closing ```
            if code_lines:
                blocks.append({"type": "code", "content": "\n".join(code_lines)})
        # Table (starts with |)
        elif line.startswith("|"):
            table_rows = []
            while i < len(lines) and lines[i].startswith("|"):
                row_line = lines[i].strip()
                # Skip separator row (|---|---|...)
                if not re.match(r"^\|[\s\-:\|]+\|$", row_line):
                    cells = [c.strip() for c in row_line.strip("|").split("|")]
                    table_rows.append(cells)
                i += 1
            if table_rows:
                blocks.append({"type": "table", "content": table_rows})
        # Image block ![FIG:caption](path) — standalone image with caption
        elif re.match(r"^!\[FIG:[^\]]+\]\([^)]+\)\s*$", line):
            m = re.match(r"^!\[FIG:([^\]]+)\]\(([^)]+)\)\s*$", line)
            caption = m.group(1).strip()
            img_path = m.group(2).strip()
            blocks.append({"type": "image", "caption": caption, "path": img_path})
            i += 1
        # List
        elif line.startswith("- ") or line.startswith("* ") or re.match(r"^\d+\.\s", line):
            list_items = []
            while i < len(lines):
                m = re.match(r"^(-|\*|\d+\.)\s+(.+)$", lines[i].rstrip())
                if not m:
                    break
                list_items.append(m.group(2))
                i += 1
            if list_items:
                blocks.append({"type": "list", "content": list_items})
        # Regular paragraph (may span multiple lines)
        else:
            para_lines = []
            while i < len(lines) and lines[i].strip() and not any(
                lines[i].startswith(prefix)
                for prefix in ("# ", "## ", "### ", "|", "- ", "* ", "```", "---")
            ) and not re.match(r"^\d+\.\s", lines[i]):
                para_lines.append(lines[i].rstrip())
                i += 1
            if para_lines:
                blocks.append({"type": "para", "content": " ".join(para_lines)})
    return blocks


def strip_md_formatting(text: str) -> str:
    """Remove **bold**, *italic*, `code`, [link](url) markdown markers."""
    # Links [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Bold+italic ***text***
    text = re.sub(r"\*\*\*([^*]+)\*\*\*", r"\1", text)
    # Bold **text**
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    # Italic *text* or _text_
    text = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", text)
    # Inline code `text` → text
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text


def insert_blocks_before(doc: Document, before_para, blocks: list[dict]) -> None:
    """Insert blocks (parsed from md) before given paragraph in doc."""
    for block in blocks:
        btype = block["type"]
        content = block.get("content")

        if btype == "h1":
            # Main chapter heading (CHƯƠNG X)
            p = before_para.insert_paragraph_before("", style="Heading 1")
            run = p.add_run(strip_md_formatting(content))
            run.font.size = Pt(14)
            run.bold = True
        elif btype == "h2":
            p = before_para.insert_paragraph_before("", style="Heading 2")
            run = p.add_run(strip_md_formatting(content))
            run.font.size = Pt(13)
            run.bold = True
        elif btype == "h3":
            p = before_para.insert_paragraph_before("", style="Heading 3")
            run = p.add_run(strip_md_formatting(content))
            run.font.size = Pt(12)
            run.bold = True
        elif btype == "para":
            text = strip_md_formatting(content)
            before_para.insert_paragraph_before(text, style="Normal")
        elif btype == "list":
            for item in content:
                item_text = "•  " + strip_md_formatting(item)
                before_para.insert_paragraph_before(item_text, style="Normal")
        elif btype == "code":
            # Code block → plain paragraph with monospace feel
            p = before_para.insert_paragraph_before(content, style="Normal")
            for run in p.runs:
                run.font.name = "Consolas"
                run.font.size = Pt(10)
        elif btype == "table":
            insert_table_before(doc, before_para, content)
        elif btype == "image":
            insert_image_before(doc, before_para, block["path"], block["caption"])


def insert_image_before(doc: Document, before_para, img_path: str,
                        caption: str, width_inches: float = 5.5) -> None:
    """Insert a centered image + caption before given paragraph.

    img_path can be relative to the markdown file (docs/tieuluan_V3/) or absolute.
    """
    # Resolve path relative to markdown dir
    md_dir = MD_FILE.parent
    img_file = (md_dir / img_path).resolve() if not Path(img_path).is_absolute() else Path(img_path)
    if not img_file.exists():
        print(f"      [WARN] Image not found, skip: {img_file}")
        return

    # Add paragraph for image (centered)
    img_para = before_para.insert_paragraph_before("", style="Normal")
    img_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    run = img_para.add_run()
    run.add_picture(str(img_file), width=Inches(width_inches))

    # Add caption below (centered, italic)
    caption_para = before_para.insert_paragraph_before("", style="Normal")
    caption_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    caption_run = caption_para.add_run(caption)
    caption_run.italic = True
    caption_run.font.size = Pt(11)


def insert_table_before(doc: Document, before_para, rows: list[list[str]]) -> None:
    """Insert a table before given paragraph."""
    if not rows:
        return
    n_cols = len(rows[0])
    n_rows = len(rows)
    # Create table at end of doc first, then move its XML before target
    table = doc.add_table(rows=n_rows, cols=n_cols)
    table.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, cell_text in enumerate(row):
            if j < n_cols:
                cell = table.rows[i].cells[j]
                cell.text = ""
                run = cell.paragraphs[0].add_run(strip_md_formatting(cell_text))
                if i == 0:
                    run.bold = True
    # Move table XML before target paragraph
    tbl_element = table._element
    tbl_element.getparent().remove(tbl_element)
    before_para._element.addprevious(tbl_element)
    # Add spacing paragraph after table
    before_para.insert_paragraph_before("", style="Normal")


def main() -> int:
    assert SRC_DOCX.exists(), f"Thiếu {SRC_DOCX}"
    assert MD_FILE.exists(), f"Thiếu {MD_FILE}"

    print(f"[1/5] Load docx gốc: {SRC_DOCX.name}")
    doc = Document(str(SRC_DOCX))
    print(f"      Tổng paragraphs: {len(doc.paragraphs)}")

    print(f"[2/5] Tìm vị trí Chương 3 (content body) + TÀI LIỆU THAM KHẢO...")
    markers = find_marker_indices(doc, ["CHƯƠNG 3", "TÀI LIỆU THAM KHẢO"])
    print(f"      Markers found: {markers}")

    if "CHƯƠNG 3" not in markers or "TÀI LIỆU THAM KHẢO" not in markers:
        print("[ERROR] Không tìm được marker. Abort.")
        return 1

    ch3_start = markers["CHƯƠNG 3"]
    tltk_start = markers["TÀI LIỆU THAM KHẢO"]
    print(f"      Chương 3 từ paragraph {ch3_start}, TLTK bắt đầu {tltk_start}")
    print(f"      Sẽ xoá {tltk_start - ch3_start} paragraphs cũ")

    print(f"[3/5] Parse markdown: {MD_FILE.name}")
    md_text = MD_FILE.read_text()
    blocks = parse_md_to_blocks(md_text)
    print(f"      Parsed {len(blocks)} blocks")

    print(f"[4/5] Xoá Chương 3 cũ...")
    # Lưu lại reference tới paragraph TÀI LIỆU THAM KHẢO
    tltk_para = doc.paragraphs[tltk_start]
    delete_paragraphs_range(doc, ch3_start, tltk_start)
    print(f"      Xoá xong, còn {len(doc.paragraphs)} paragraphs")

    print(f"[5/5] Insert Chương 3 + Chương 4 mới trước TÀI LIỆU THAM KHẢO...")
    insert_blocks_before(doc, tltk_para, blocks)
    print(f"      Insert xong, tổng paragraphs: {len(doc.paragraphs)}")

    print(f"\n[SAVE] {NEW_DOCX}")
    doc.save(str(NEW_DOCX))
    print(f"      Size: {NEW_DOCX.stat().st_size / 1024:.1f} KB")

    print(f"\n[DONE] Mở trong Word:")
    print(f"  {NEW_DOCX}")
    print(f"\nFile gốc giữ nguyên: {SRC_DOCX.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
