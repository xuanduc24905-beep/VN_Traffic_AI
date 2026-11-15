#!/usr/bin/env python3
"""Extract all paragraphs + tables from the report .docx into a structured JSON.

Usage: python 00_extract_docx.py <path-to-docx> <out-json>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


NUM_RE = re.compile(r"[-+]?\d[\d\.,]*%?")


def iter_block_items(parent):
    """Yield ('p', paragraph) or ('t', table) in document order."""
    body = parent.element.body
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield "p", Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield "t", Table(child, parent)


def extract_numbers(text: str):
    nums = []
    for m in NUM_RE.finditer(text):
        tok = m.group(0)
        if tok in {"-", "+", "."}:
            continue
        if not any(c.isdigit() for c in tok):
            continue
        nums.append(tok)
    return nums


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = Document(str(src))

    paragraphs = []
    tables = []
    all_numbers = []

    p_idx = 0
    t_idx = 0
    cur_heading = None
    for kind, obj in iter_block_items(doc):
        if kind == "p":
            text = obj.text
            style = obj.style.name if obj.style else ""
            if style.startswith("Heading") or style.startswith("Tiêu đề"):
                cur_heading = text
            nums = extract_numbers(text)
            paragraphs.append({
                "idx": p_idx,
                "style": style,
                "heading": cur_heading,
                "text": text,
                "numbers": nums,
            })
            for n in nums:
                all_numbers.append({
                    "where": f"paragraph[{p_idx}] (heading='{cur_heading}')",
                    "number": n,
                    "context": text[:240],
                })
            p_idx += 1
        else:  # table
            rows = []
            for r_idx, row in enumerate(obj.rows):
                cells = []
                for c_idx, cell in enumerate(row.cells):
                    txt = cell.text.strip()
                    cells.append(txt)
                    nums = extract_numbers(txt)
                    for n in nums:
                        all_numbers.append({
                            "where": f"table[{t_idx}] row {r_idx} col {c_idx} (heading='{cur_heading}')",
                            "number": n,
                            "context": txt[:240],
                        })
                rows.append(cells)
            tables.append({
                "idx": t_idx,
                "heading": cur_heading,
                "rows": rows,
            })
            t_idx += 1

    out_obj = {
        "source": str(src),
        "n_paragraphs": len(paragraphs),
        "n_tables": len(tables),
        "n_numbers": len(all_numbers),
        "paragraphs": paragraphs,
        "tables": tables,
        "all_numbers": all_numbers,
    }
    out.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2))
    print(f"OK — {len(paragraphs)} paragraphs, {len(tables)} tables, {len(all_numbers)} numeric tokens -> {out}")


if __name__ == "__main__":
    main()
