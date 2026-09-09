"""Task 2.0 — Ingest a two-sided case into the gym's corpus structure.

Extracts the text of each role's confidential instructions to `roles/<role_id>/` and leaves the
public corpus in `public/`. It also inserts a **canary** per role: a unique string that appears
nowhere else, so the isolation test can prove that one party's corpus never reaches the
other's artifacts.

Usage:  python scripts/ingest_case.py --case parker_gibson
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kb.extract_text import extract  # noqa: E402

# Which PDF in `source/` belongs to which role.
ROLE_SOURCES = {
    "parker_gibson": {
        "parkers": "Parker-Gibson - Parker confidential.pdf",
        "gibsons": "Parker-Gibson - Gibson confidential.pdf",
    },
}


def canary_for(case_id: str, role_id: str) -> str:
    """A unique, stable string per case and role."""
    digest = hashlib.sha256(f"{case_id}/{role_id}/canary".encode()).hexdigest()[:12]
    return f"CANARY-{role_id.upper()}-{digest}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="parker_gibson")
    args = ap.parse_args()

    base = Path("cases") / args.case
    sources = ROLE_SOURCES.get(args.case)
    if not sources:
        raise SystemExit(f"I do not know which PDF belongs to which role in {args.case}; "
                         f"add it to ROLE_SOURCES")

    for role_id, filename in sources.items():
        pdf = base / "source" / filename
        if not pdf.exists():
            raise SystemExit(f"missing {pdf}")
        text = extract(pdf)
        canary = canary_for(args.case, role_id)
        target = base / "roles" / role_id / "confidential.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            f"---\ncase_id: {args.case}\nrole_id: {role_id}\n"
            f"visibility: confidential\ncanary: {canary}\n"
            f"source_pdf: {pdf}\n---\n\n"
            f"<!-- {canary} -->\n\n"
            f"# Confidential instructions — {role_id}\n\n{text}\n",
            encoding="utf-8")
        print(f"{role_id}: {len(text.split()):>5} words -> {target}  (canary {canary})")


if __name__ == "__main__":
    main()
