"""Generate the human review queue of references (tasks 1.2 and 1.3).

Output: knowledge/db/review_queue.md — one entry per reference that needs a human eye, with the
report's citation context so the review does not require opening the PDF.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kb.schemas import KosmosClaim, Reference


def scope(ref: Reference) -> str:
    return "L1/L2" if set(ref.queries) & {"L1", "L2"} else "L3"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="knowledge/db/references.jsonl")
    ap.add_argument("--claims", default="knowledge/db/kosmos_claims.jsonl")
    ap.add_argument("--out", default="knowledge/db/review_queue.md")
    args = ap.parse_args()

    refs = [Reference(**json.loads(l)) for l in Path(args.refs).read_text().splitlines() if l.strip()]
    claims = {c.claim_id: c for c in (
        KosmosClaim(**json.loads(l)) for l in Path(args.claims).read_text().splitlines() if l.strip()
    )}

    pending = [r for r in refs if r.needs_review or r.fetch_status in ("missing", "paywalled", "failed")]
    pending.sort(key=lambda r: (scope(r), -r.n_citations))

    lines = [
        "# Reference review queue",
        "",
        f"Generated automatically. {len(pending)} of {len(refs)} references need review.",
        "",
        "Possible reasons: metadata not resolved with confidence (`ambiguous`/`not_found`), "
        "or PDF not downloadable through legitimate routes (`missing`/`paywalled`).",
        "Those in scope **L1/L2** feed E1's fox catalog; **L3** ones concern credit assignment, "
        "out of scope in E1 (DECISIONS D-010).",
        "",
    ]
    current_scope = None
    for ref in pending:
        if scope(ref) != current_scope:
            current_scope = scope(ref)
            lines += ["", f"## Scope {current_scope}", ""]
        label = ref.mention_string or ref.title or ref.ref_id
        lines.append(f"### {label}  ·  `{ref.ref_id}`")
        lines.append("")
        lines.append(f"- **Kind:** {ref.ref_kind} · **citations in the report:** {ref.n_citations} "
                     f"· **sections:** {', '.join(ref.queries)}")
        lines.append(f"- **Metadata status:** {ref.resolve_status}"
                     + (f" (score {ref.match_score})" if ref.match_score is not None else ""))
        if ref.resolved_title:
            lines.append(f"- **Best candidate:** {ref.resolved_title} "
                         f"({ref.resolved_year}) — {ref.doi or 'no DOI'}")
        if ref.fetch_status != "downloaded":
            lines.append(f"- **PDF:** {ref.fetch_status}"
                         + (f" — {ref.fetch_note}" if ref.fetch_note else ""))
        if ref.review_note:
            lines.append(f"- **Note:** {ref.review_note.strip(' |')}")
        ctx = [claims[cid].text for cid in ref.claim_ids[:2] if cid in claims]
        if ctx:
            lines.append("- **What the report attributes to it:**")
            for text in ctx:
                lines.append(f"  > {text}")
        lines.append("")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(pending)} references in the queue -> {args.out}")


if __name__ == "__main__":
    main()
