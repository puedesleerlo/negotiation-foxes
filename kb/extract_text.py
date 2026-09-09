"""Task 1.4 (a) — Section-wise text extraction from the downloaded PDFs.

Uses PyMuPDF with font information: the most frequent font size is the body; lines with a
larger or bold font and few words are marked as headings. GROBID is not available in this
environment (it needs a separate service); if installed, this function can be swapped behind
the same interface.

Output: knowledge/extracted/<paper_id>.md with `##` headings and the body in paragraphs.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pymupdf

from kb.schemas import Reference
from kb.text import dehyphenate_join, normalize_chars

HEAD_HINT = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+)?(abstract|introduction|related work|background|method(?:s|ology)?|"
    r"model|approach|experiment(?:s|al setup)?|evaluation|result(?:s)?|discussion|"
    r"conclusion(?:s)?|references|acknowledg(?:e)?ments|appendix)\b",
    re.I,
)


def _join_spans(spans: list[tuple[float, float, str, float, bool]]) -> tuple[str, float, bool]:
    """Join the fragments of one line, inserting spaces according to the real gap."""
    spans = sorted(spans, key=lambda s: s[0])
    text = ""
    prev_x1 = None
    sizes, bolds = [], []
    for x0, x1, frag, size, bold in spans:
        if prev_x1 is not None and not text.endswith(" ") and not frag.startswith(" "):
            if x0 - prev_x1 > 0.22 * max(size, 1.0):
                text += " "
        text += frag
        prev_x1 = x1
        sizes.append(size)
        bolds.append(bold)
    return (normalize_chars(text).strip(),
            sum(sizes) / max(len(sizes), 1),
            any(bolds))


def page_lines(page: pymupdf.Page, y_tol: float = 2.5) -> list[tuple[str, float, bool]]:
    """[(text, mean font size, is bold)] per visual line.

    Some PDFs (e.g. Faratin 1998) place every word as its own 'line', so lines are rebuilt by
    grouping the fragments by vertical coordinate *within each block* — the block preserves
    column order in two-column documents.
    """
    out: list[tuple[str, float, bool]] = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        items: list[tuple[float, float, float, str, float, bool]] = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                frag = span.get("text", "")
                if not frag.strip():
                    continue
                x0, y0, x1, y1 = span.get("bbox", (0, 0, 0, 0))
                items.append(((y0 + y1) / 2, x0, x1, frag, span.get("size", 0.0),
                              "bold" in (span.get("font", "") or "").lower()))
        if not items:
            continue
        items.sort(key=lambda r: (r[0], r[1]))
        group: list[tuple[float, float, str, float, bool]] = []
        base_y = items[0][0]
        for y, x0, x1, frag, size, bold in items:
            if abs(y - base_y) > y_tol and group:
                out.append(_join_spans(group))
                group = []
                base_y = y
            group.append((x0, x1, frag, size, bold))
        if group:
            out.append(_join_spans(group))
    return [line for line in out if line[0]]


# Frequent English words whose ligature (fi, ff, fl) extraction eats:
# su[ffi]cient -> sucient, arti[fi]cial -> articial, con[fl]ict -> conict...
LIGATURE_LOSS = re.compile(
    r"\b(articial|scientic|sucient|conict|dierent|dierence|eective|eciency|ecient|specic|"
    r"identied|classication|signicant|nding|ndings|rst|nal|dene|dened|denition|"
    r"benet|prot|conguration|coecient)\b", re.I)


def quality_flag(text: str) -> str:
    """Flag degraded text: PDFs with bitmap fonts or scans lose ligatures.

    It does not use the density of one-letter tokens: in papers with mathematical notation it
    is legitimately high. It uses frequent English words with the ligature eaten.
    """
    tokens = text.split()
    if not tokens:
        return "empty"
    hits = len(LIGATURE_LOSS.findall(text))
    if hits >= 3 and hits / max(len(tokens), 1) > 1e-4:
        return "low"
    singles = sum(1 for t in tokens if len(t) == 1 and t.isascii() and t.isalpha()
                  and t.lower() not in ("a", "i"))
    return "low" if singles / len(tokens) > 0.08 else "ok"


def extract(pdf_path: Path) -> str:
    doc = pymupdf.open(pdf_path)
    lines: list[tuple[str, float, bool]] = []
    for page in doc:
        lines.extend(page_lines(page))
    if not lines:
        return ""
    body_size = Counter(round(sz, 1) for _, sz, _ in lines).most_common(1)[0][0]
    # Some old PDFs (dvips Type3) report absurd sizes (0.2): there, heading detection by font
    # size is useless and only the lexical pattern is used.
    use_font_size = body_size >= 4.0

    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            joined = " ".join(dehyphenate_join(buf))
            out.append(re.sub(r"\s+", " ", joined).strip())
            buf.clear()

    for text, size, bold in lines:
        words = len(text.split())
        is_head = (
            use_font_size
            and words <= 14
            and (size >= body_size * 1.12 or (bold and size >= body_size * 0.98))
            and not text.endswith((".", ",", ";"))
        ) or bool(HEAD_HINT.match(text))
        if is_head and words <= 14:
            flush()
            out.append(f"\n## {text.strip()}\n")
        else:
            buf.append(text)
    flush()
    return "\n\n".join(p for p in out if p.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract text from papers (task 1.4)")
    ap.add_argument("--refs", default="knowledge/db/references.jsonl")
    ap.add_argument("--out-dir", default="knowledge/extracted")
    args = ap.parse_args()

    refs = [Reference(**json.loads(l))
            for l in Path(args.refs).read_text().splitlines() if l.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    for ref in refs:
        if ref.fetch_status != "downloaded" or not ref.pdf_path or ref.paper_id in done:
            continue
        done.add(ref.paper_id or "")
        text = extract(Path(ref.pdf_path))
        flag = quality_flag(text)
        target = out_dir / f"{ref.paper_id}.md"
        title = ref.resolved_title or ref.title or ref.paper_id
        header = (f"---\npaper_id: {ref.paper_id}\ntitle: {title}\n"
                  f"year: {ref.resolved_year or ref.year}\ndoi: {ref.doi}\n"
                  f"source_pdf: {ref.pdf_path}\nextraction_quality: {flag}\n---\n\n")
        target.write_text(header + text, encoding="utf-8")
        heads = text.count("\n## ")
        print(f"{ref.paper_id:22} {len(text.split()):>7} words, {heads:>3} headings, "
              f"quality={flag}")
    print(f"\n{len(done)} papers extracted to {out_dir}")


if __name__ == "__main__":
    main()
