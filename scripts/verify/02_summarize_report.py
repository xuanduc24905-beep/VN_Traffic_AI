#!/usr/bin/env python3
"""Print headings + per-table brief so we can map numbers to verification tasks."""
from __future__ import annotations

import json
from pathlib import Path

d = json.loads(Path("outputs/verify/report_extract.json").read_text())

print("=" * 80)
print("HEADINGS")
print("=" * 80)
for p in d["paragraphs"]:
    st = p.get("style", "")
    if st and ("Heading" in st or "Title" in st or "Tiêu" in st):
        print(f"  [{p['idx']:4d}] {st:20s} | {p['text']}")

print()
print("=" * 80)
print("TABLES (rows x cols, first row = header)")
print("=" * 80)
for t in d["tables"]:
    rows = t["rows"]
    nrows = len(rows)
    ncols = max((len(r) for r in rows), default=0)
    hdr = rows[0] if rows else []
    # Find a caption-like paragraph before (heuristic: iterate paragraphs by idx; we don't track interleave here)
    print(f"  TBL #{t['idx']:2d}  {nrows}x{ncols}  heading='{t['heading']}'")
    print(f"      header: {hdr}")
    # Second row preview to understand data
    if nrows > 1:
        print(f"      row1:   {rows[1]}")
    if nrows > 2:
        print(f"      row2:   {rows[2]}")
