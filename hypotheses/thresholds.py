"""Turning a hypothesis statement into something a truth can be compared against.

The planner is asked to produce a structured `threshold` on every numeric hypothesis. This is
the conservative fallback for statements that lack one: it extracts a bound only when the
wording is unambiguous, in either order the language allows ("at least 20,000" / "20,000 or
less"), and returns `None` otherwise. Inventing a bound to be able to score a hypothesis would be
worse than not scoring it.
"""
from __future__ import annotations

import re
from typing import Optional

# Direction before the number ("at least 20 000", "above 30 000") ...
PREFIX = re.compile(
    # negated forms first: the regex takes the leftmost match, and "does not exceed" starts
    # before "exceed" does
    r"(?P<dir>does not exceed|not exceed|not more than|not above|not over|"
    r"at least|at most|no less than|no more than|above|below|over|under|"
    r"more than|less than|greater than|lower than|higher than|exceeds?|"
    r"al menos|como m[ií]nimo|por encima de|superior a|supera los|"
    r"como m[áa]ximo|por debajo de|inferior a|no supera|no m[áa]s de)\D{0,20}"
    r"(?P<value>\d[\d\.,\s]{2,})", re.I)
# ... or after it ("20 000 or less", "30 000 or more").
SUFFIX = re.compile(
    r"(?P<value>\d[\d\.,\s]{2,})\s*(?:USD|dollars|d[óo]lares)?\s*"
    r"(?P<dir>or less|or more|or lower|or higher|or above|or below|"
    r"o menos|o m[áa]s|o superior|o inferior|o menor|o mayor)", re.I)
NEGATED_LTE = ("does not exceed", "not exceed", "not more than", "not above", "not over")
GTE_WORDS = ("at least", "no less than", "above", "over", "more than", "greater than",
             "higher than", "exceed", "or more", "or higher", "or above",
             "al menos", "como mínimo", "como minimo", "por encima de", "superior a",
             "supera los", "o más", "o mas", "o superior", "o mayor")


def threshold_from_prose(statement: str) -> Optional[tuple[str, float]]:
    """('gte' | 'lte', value) when the statement carries an unambiguous bound, else None."""
    match = PREFIX.search(statement or "") or SUFFIX.search(statement or "")
    if not match:
        return None
    raw = re.sub(r"[\s,\.]", "", match.group("value"))
    try:
        value = float(raw)
    except ValueError:
        return None
    direction = match.group("dir").lower()
    # A negated phrase ("does not exceed") contains a GTE word as a substring; the negation
    # must be recognised before the substring test, or the bound flips.
    if any(direction.startswith(w) for w in NEGATED_LTE):
        return "lte", value
    return ("gte" if any(w in direction for w in GTE_WORDS) else "lte"), value
