#!/usr/bin/env python3
"""Force every Word font declaration in the report to Times New Roman."""

from pathlib import Path
from shutil import copy2
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


DOCX = Path("docs/tieuluan_V3/Nhom02_PhanLoaiPhuongTien_UPDATED5 (1).docx")
BACKUP = DOCX.with_name(DOCX.stem + "_before_all_tnr_fix.docx")
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
FONT = "Times New Roman"


def force_fonts(root: etree._Element) -> tuple[int, int]:
    declarations = 0
    for fonts in root.xpath('.//*[local-name()="rFonts"]'):
        for attr in list(fonts.attrib):
            if etree.QName(attr).localname.endswith("Theme") or etree.QName(attr).localname.endswith("theme"):
                del fonts.attrib[attr]
        for key in ("ascii", "hAnsi", "eastAsia", "cs"):
            fonts.set(f"{{{W}}}{key}", FONT)
        declarations += 1

    math_fonts = 0
    for math_font in root.xpath('.//*[local-name()="mathFont"]'):
        math_font.set(f"{{{M}}}val", FONT)
        math_fonts += 1
    return declarations, math_fonts


def main() -> None:
    if not BACKUP.exists():
        copy2(DOCX, BACKUP)

    with ZipFile(DOCX, "r") as source:
        entries = {item.filename: source.read(item.filename) for item in source.infolist()}

    declarations = 0
    math_fonts = 0
    for name, payload in list(entries.items()):
        if not (name.startswith("word/") and name.endswith(".xml")):
            continue
        try:
            root = etree.fromstring(payload)
        except etree.XMLSyntaxError:
            continue
        changed, changed_math = force_fonts(root)
        if changed or changed_math:
            entries[name] = etree.tostring(
                root, xml_declaration=True, encoding="UTF-8", standalone=True
            )
            declarations += changed
            math_fonts += changed_math

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

    print(f"Updated {declarations} Word font declarations")
    print(f"Updated {math_fonts} equation font declarations")
    print(f"Backup: {BACKUP}")


if __name__ == "__main__":
    main()
