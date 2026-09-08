"""Normalisation of text extracted from PDFs: de-hyphenation, paragraphs, headings."""
from __future__ import annotations

import re
import unicodedata

PAGE_MARK = re.compile(r"^<<<PAGE \d+>>>$")


def normalize_chars(text: str) -> str:
    """Unify quotes, dashes and odd spaces without destroying accents."""
    text = unicodedata.normalize("NFC", text)
    for bad, good in [
        ("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
        (" ", " "), ("ﬁ", "fi"), ("ﬂ", "fl"),
    ]:
        text = text.replace(bad, good)
    return text


def dehyphenate_join(lines: list[str]) -> list[str]:
    """Join lines split by a soft hyphen ('hypothe-' + 'sis' -> 'hypothesis')."""
    out: list[str] = []
    buf = ""
    for line in lines:
        line = line.rstrip()
        if not buf:
            buf = line
            continue
        if buf.endswith("-") and not buf.endswith("--") and line[:1].islower():
            buf = buf[:-1] + line
        else:
            out.append(buf)
            buf = line
    if buf:
        out.append(buf)
    return out


PARA_END = re.compile(r"[.!?:][\"')\]]?$")
# Lines that always open a paragraph: bibliography entries and headings.
BREAK_BEFORE = re.compile(
    r"^(?:\[\d+\.\d+\]|##\s|#\s|L[123]\s*[—-]\s|\d+\.\s+[A-Z]|References$|Results$"
    r"|Objective$|Query\.$|Disagreements)"
)


def paragraphs(lines: list[str], width_ratio: float = 0.92) -> list[str]:
    """Rebuild paragraphs from justified-column text.

    A line closes a paragraph only if it is short *and* ends with final punctuation: a short
    line ending mid-sentence ('... Ames and') is a line break, not a paragraph end.
    """
    lengths = [len(l) for l in lines if l.strip()]
    if not lengths:
        return []
    width = max(lengths)
    cutoff = width * width_ratio
    paras: list[str] = []
    buf: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            if buf:
                paras.append(" ".join(buf))
                buf = []
            continue
        if BREAK_BEFORE.match(s) and buf:
            paras.append(" ".join(buf))
            buf = []
        buf.append(s)
        closes = (len(line.rstrip()) < cutoff and PARA_END.search(s)) or (
            len(s) < 70 and not PARA_END.search(s) and BREAK_BEFORE.match(s)
        )
        if closes:
            paras.append(" ".join(buf))
            buf = []
    if buf:
        paras.append(" ".join(buf))
    return [re.sub(r"\s+", " ", p).strip() for p in paras if p.strip()]


SENTENCE_END = re.compile(r"(?<=[.!?])[\"']?\s+(?=[A-Z(\[\"'])")
ABBREV = re.compile(
    r"\b(?:et al|e\.g|i\.e|cf|vs|Fig|Prof|Dr|Mr|Ms|St|Int|J|Hum|Comput|Stud|Syst|Auton|approx)\.$"
)


def sentences(paragraph: str) -> list[str]:
    """Sentence split tolerant of the abbreviations frequent in bibliographies."""
    parts = SENTENCE_END.split(paragraph)
    out: list[str] = []
    for part in parts:
        if out and ABBREV.search(out[-1]):
            out[-1] = out[-1] + " " + part
        else:
            out.append(part)
    return [p.strip() for p in out if p.strip()]
