from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

from lxml import etree
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "docs/tieuluan_V3/Nhom02_PhanLoaiPhuongTien_UPDATED5 (1).docx"
BACKUP = ROOT / "docs/tieuluan_V3/Nhom02_PhanLoaiPhuongTien_UPDATED5 (1).before_PROOFING_CLEAN.docx"
MATH_SYMBOLS = set("∈≤≥≈±×→↔∞∑∫λσ−")


def set_rfonts(rpr, name: str) -> None:
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
        rfonts.set(qn(f"w:{attr}"), name)


def set_lang(rpr, lang: str) -> None:
    node = rpr.find(qn("w:lang"))
    if node is None:
        node = OxmlElement("w:lang")
        rpr.append(node)
    node.set(qn("w:val"), lang)
    node.set(qn("w:eastAsia"), lang)


def configure_proofing(doc: Document) -> None:
    settings = doc.settings._element
    for tag in ("w:hideSpellingErrors", "w:hideGrammaticalErrors"):
        node = settings.find(qn(tag))
        if node is None:
            node = OxmlElement(tag)
            settings.append(node)
        node.set(qn("w:val"), "true")
    proof = settings.find(qn("w:proofState"))
    if proof is None:
        proof = OxmlElement("w:proofState")
        settings.append(proof)
    proof.set(qn("w:spelling"), "clean")
    proof.set(qn("w:grammar"), "clean")

    defaults = doc.styles.element.find(qn("w:docDefaults"))
    rpr = defaults.find(f"{qn('w:rPrDefault')}/{qn('w:rPr')}")
    set_lang(rpr, "vi-VN")
    for style_name in ("Normal", "List Bullet", "Heading 1", "Heading 2", "Heading 3", "Chu thich hinh", "Chu thich bang"):
        style = doc.styles[style_name]
        set_lang(style.element.get_or_add_rPr(), "vi-VN")


def document_roots(doc: Document):
    roots = [doc.element]
    seen = {id(doc.element)}
    for section in doc.sections:
        for part in (section.header, section.footer):
            if id(part._element) not in seen:
                roots.append(part._element)
                seen.add(id(part._element))
    return roots


def remove_proof_errors(doc: Document) -> int:
    count = 0
    for root in document_roots(doc):
        for node in list(root.xpath('.//w:proofErr')):
            parent = node.getparent()
            if parent is not None:
                parent.remove(node)
                count += 1
    return count


def simple_text_run(run):
    children = list(run)
    allowed = {qn("w:rPr"), qn("w:t")}
    if not children or any(c.tag not in allowed for c in children):
        return None
    texts = [c for c in children if c.tag == qn("w:t")]
    return texts[0] if len(texts) == 1 else None


def rpr_key(run) -> bytes:
    rpr = run.find(qn("w:rPr"))
    if rpr is None:
        return b""
    return etree.tostring(rpr, method="c14n")


def merge_adjacent_runs(doc: Document) -> int:
    merged = 0
    for root in document_roots(doc):
        for p in root.xpath('.//w:p'):
            changed = True
            while changed:
                changed = False
                children = list(p)
                for i in range(len(children) - 1):
                    a, b = children[i], children[i + 1]
                    if a.tag != qn("w:r") or b.tag != qn("w:r"):
                        continue
                    ta, tb = simple_text_run(a), simple_text_run(b)
                    if ta is None or tb is None or rpr_key(a) != rpr_key(b):
                        continue
                    ta.text = (ta.text or "") + (tb.text or "")
                    if ta.text.startswith(" ") or ta.text.endswith(" ") or "  " in ta.text:
                        ta.set(qn("xml:space"), "preserve")
                    p.remove(b)
                    merged += 1
                    changed = True
                    break
    return merged


def normalize_run_language_and_symbols(doc: Document) -> int:
    fixed = 0
    for pi, p in enumerate(doc.paragraphs):
        lang = "en-US" if 5 <= pi <= 9 else "vi-VN"
        for run in p.runs:
            rpr = run._r.get_or_add_rPr()
            set_lang(rpr, lang)
            if any(ch in MATH_SYMBOLS for ch in run.text) and run.font.name == "Consolas":
                run.font.name = "Times New Roman"
                run.font.size = Pt(13)
                set_rfonts(rpr, "Times New Roman")
                fixed += 1
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for run in p.runs:
                        set_lang(run._r.get_or_add_rPr(), "vi-VN")
    return fixed


def rebuild_membership_paragraph(doc: Document) -> None:
    for p in doc.paragraphs:
        if p.text.startswith("Trong đó class_id ∈ {0, 1, 2, 3}"):
            text = p.text
            p._p.clear_content()
            run = p.add_run(text)
            run.font.name = "Times New Roman"
            run.font.size = Pt(13)
            rpr = run._r.get_or_add_rPr()
            set_rfonts(rpr, "Times New Roman")
            set_lang(rpr, "vi-VN")
            break


def main() -> None:
    if not BACKUP.exists():
        shutil.copy2(DOCX, BACKUP)
    doc = Document(DOCX)
    configure_proofing(doc)
    removed = remove_proof_errors(doc)
    fixed_symbols = normalize_run_language_and_symbols(doc)
    rebuild_membership_paragraph(doc)
    merged = merge_adjacent_runs(doc)
    doc.save(DOCX)
    print(f"Removed proofErr: {removed}")
    print(f"Merged adjacent runs: {merged}")
    print(f"Fixed symbol-font runs: {fixed_symbols}")
    print(f"Updated {DOCX}")


if __name__ == "__main__":
    main()
