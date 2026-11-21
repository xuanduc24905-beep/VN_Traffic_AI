#!/usr/bin/env python3
"""Prevent Word from stretching short bullet-list lines in the final report."""

from pathlib import Path
from shutil import copy2
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


DOCX = Path("docs/tieuluan_V3/Nhom02_PhanLoaiPhuongTien_UPDATED5 (1).docx")
BACKUP = DOCX.with_name(DOCX.stem + "_before_list_spacing_fix.docx")
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}


def set_left_alignment(ppr: etree._Element) -> None:
    jc = ppr.find("w:jc", NS)
    if jc is None:
        jc = etree.SubElement(ppr, f"{{{W}}}jc")
    jc.set(f"{{{W}}}val", "left")


def main() -> None:
    if not BACKUP.exists():
        copy2(DOCX, BACKUP)

    with ZipFile(DOCX, "r") as source:
        entries = {item.filename: source.read(item.filename) for item in source.infolist()}

    styles = etree.fromstring(entries["word/styles.xml"])
    document = etree.fromstring(entries["word/document.xml"])

    style = styles.xpath(
        './/w:style[@w:type="paragraph" and @w:styleId="ListBullet"]',
        namespaces=NS,
    )
    if style:
        ppr = style[0].find("w:pPr", NS)
        if ppr is None:
            ppr = etree.SubElement(style[0], f"{{{W}}}pPr")
        set_left_alignment(ppr)

    changed = 0
    for paragraph in document.xpath('.//w:p[w:pPr/w:pStyle[@w:val="ListBullet"]]', namespaces=NS):
        ppr = paragraph.find("w:pPr", NS)
        set_left_alignment(ppr)
        changed += 1

    entries["word/styles.xml"] = etree.tostring(
        styles, xml_declaration=True, encoding="UTF-8", standalone=True
    )
    entries["word/document.xml"] = etree.tostring(
        document, xml_declaration=True, encoding="UTF-8", standalone=True
    )

    with NamedTemporaryFile(dir=DOCX.parent, suffix=".docx", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with ZipFile(tmp_path, "w", ZIP_DEFLATED) as target:
            for name, payload in entries.items():
                target.writestr(name, payload)
        tmp_path.replace(DOCX)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    print(f"Left-aligned {changed} ListBullet paragraphs")
    print(f"Backup: {BACKUP}")


if __name__ == "__main__":
    main()
