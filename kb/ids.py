"""Stable identifiers of the project."""
from __future__ import annotations

import re


def slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")[:n]


def make_paper_id(authors: list[str], year: int | None, title: str | None) -> str:
    """`<surname><year><4 letters of the title>`, e.g. baarslag2016lear."""
    # Walk the list until a surname transliterable to ascii is found: OpenAlex returns some
    # names in their original script ("崔宗琦").
    surname = "unknown"
    for author in authors or []:
        parts = [p for p in re.split(r"[\s,]+", author) if p]
        if not parts:
            continue
        candidate = re.sub(r"[^a-z]", "", parts[-1].lower())
        if candidate:
            surname = candidate
            break
    year_part = str(year) if year and 1900 <= year <= 2030 else "0000"
    title_part = re.sub(r"[^a-z]", "", (title or "").lower())[:4] or "xxxx"
    return f"{surname}{year_part}{title_part}"
