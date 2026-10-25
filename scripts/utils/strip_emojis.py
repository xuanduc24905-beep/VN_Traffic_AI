"""Xoá emoji + icon khỏi source files (KHÔNG đụng indentation Python).

Bug cũ: collapse multi-space → hỏng indent. Fix: chỉ xoá char emoji rồi
strip trailing spaces (không đụng leading whitespace).

Chạy: python scripts/strip_emojis.py [--dry-run]
"""
import re
import sys
from pathlib import Path

EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U0001F000-\U0001F2FF"
    "\U00002600-\U000027BF"
    "\U00002B00-\U00002BFF"
    "\U0001F1E0-\U0001F1FF"
    "︀-️"
    "‍"
    "]+",
    flags=re.UNICODE,
)

EXTRA_CHARS = "★⭐◆◇●○▪▫→←↑↓↔⏎⏭⏰⌛"

EXTS = {".py", ".md", ".ipynb", ".sh", ".json"}
SKIP_DIRS = {".git", "runs", "logs", "weights", "data",
             ".venv", "venv", "__pycache__", "node_modules",
             "raw_vn_cantho"}
SKIP_FILES = {"strip_emojis.py"}


def clean(text: str) -> tuple[str, int]:
    n_before = len(text)
    text = EMOJI_RE.sub("", text)
    for ch in EXTRA_CHARS:
        text = text.replace(ch, "")
    # CHỈ xoá trailing whitespace, GIỮ NGUYÊN indent
    text = re.sub(r"[ \t]+\n", "\n", text)
    # Xoá space liên tiếp GIỮA từ (không phải đầu dòng)
    lines = []
    for line in text.split("\n"):
        # Tách indent
        m = re.match(r"^([ \t]*)(.*)$", line)
        indent, rest = m.group(1), m.group(2)
        # Collapse chỉ trong phần rest
        rest = re.sub(r" {2,}", " ", rest)
        lines.append(indent + rest)
    text = "\n".join(lines)
    return text, n_before - len(text)


def should_skip(p: Path, root: Path) -> bool:
    if p.name in SKIP_FILES:
        return True
    for part in p.relative_to(root).parts:
        if part in SKIP_DIRS:
            return True
    return p.suffix.lower() not in EXTS


def main():
    dry = "--dry-run" in sys.argv
    root = Path(__file__).resolve().parent.parent

    changed = []
    total = 0
    for p in root.rglob("*"):
        if not p.is_file() or should_skip(p, root):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        cleaned, n = clean(text)
        if n == 0:
            continue
        changed.append((p, n))
        total += n
        if not dry:
            p.write_text(cleaned, encoding="utf-8")

    print(f"{'[DRY] ' if dry else ''}{len(changed)} file, {total} char emoji xoá")
    for p, n in sorted(changed, key=lambda x: -x[1])[:30]:
        print(f"  {n:5d}  {p.relative_to(root)}")


if __name__ == "__main__":
    main()
