"""Task 1.1 — Ingestion of the Kosmos reports.

For each report in knowledge/kosmos/ it extracts:
  * the normalised text                -> knowledge/extracted/kosmos/<report_id>.txt
  * the claims with their citations    -> knowledge/db/kosmos_claims.jsonl
  * the references (listed + inline)   -> knowledge/db/references.jsonl

Two kinds of reference are distinguished (see DECISIONS D-009):
  listed  = formal entry in the report's References section (title, authors, DOI/URL).
  inline  = work named in the body (author + year) without a formal entry; resolved later.

Usage:  python -m kb.ingest_kosmos [--kosmos-dir knowledge/kosmos] [--out-dir knowledge/db]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pymupdf

from kb.ids import make_paper_id
from kb.schemas import KosmosClaim, Reference
from kb.text import dehyphenate_join, normalize_chars, paragraphs, sentences

MARKER = re.compile(r"\[(\d+\.\d+)\]")
REF_ENTRY = re.compile(r"^\[(\d+\.\d+)\]\s+(.*)$", re.S)
URL_RE = re.compile(r"https?://\S+")
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+")
YEAR_TAIL = re.compile(r"\((\d{4})\)\.?\s*$")
YEAR_BARE = re.compile(r"(?:^|\s)((?:19|20)\d{2})\.\s*$")

# Headings in the body of the report.
H_QUERY = re.compile(r"^(?:##\s*)?(L[123])\s*[—-]\s*(.+)$")
H_NUM = re.compile(r"^(\d+)\.\s+([A-Z].*)$")
H_PLAIN = {"Results", "Objective", "References", "Query.",
           "Disagreements, Failures to Replicate, and Open Questions"}

# Inline mentions: "Baarslag et al. 2016", "Zeng and Sycara 1998",
# "Leahu, Kaisers, and Baarslag 2019", "Wolpert and Tumer, 2002".
NAME = r"[A-Z][A-Za-zÀ-ſ\-]+"
INLINE_MENTION = re.compile(
    rf"\b({NAME}(?:,\s+{NAME})*(?:,?\s+(?:and|&)\s+{NAME})?(?:\s+et\s+al\.)?)"
    rf"(?:'s)?,?\s+\(?((?:19|20)\d{{2}})\)?"
)
# Sentence-initial words that are not surnames.
STOPNAMES = {
    "The", "This", "These", "Those", "In", "For", "From", "By", "At", "Since", "While",
    "When", "Where", "What", "Which", "It", "Its", "Their", "They", "There", "A", "An",
    "As", "But", "And", "Or", "Not", "No", "Both", "Two", "One", "Three", "Multiple",
    "Several", "Under", "Over", "With", "Without", "Within", "Across", "After", "Before",
    "During", "However", "Although", "Because", "Thus", "Therefore", "Here", "Also",
    "Bayesian", "Gaussian", "Particle", "Regression", "Evaluation", "Open", "Whether",
    "Empirical", "Section", "Figure", "Table", "COVID", "ANAC", "GENIUS", "MESOs",
    "Anchor", "Offers", "Optimism", "Integrative", "Information", "Credit", "Mechanism",
    "Diversity", "Incentive", "Temporal", "Exact", "Difference", "Shapley", "Hanson",
    "Exponential", "Repeated", "Retrospective", "Crucially", "Recently", "More",
}


def slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:n]


def read_pdf(path: Path) -> list[str]:
    doc = pymupdf.open(path)
    lines: list[str] = []
    for page in doc:
        lines.extend(normalize_chars(page.get_text("text")).splitlines())
    return lines


def is_heading(par: str) -> bool:
    if len(par) > 130 or MARKER.search(par):
        return False
    if par in H_PLAIN or H_QUERY.match(par) or H_NUM.match(par):
        return True
    return False


def merge_wrapped_headings(paras: list[str]) -> list[str]:
    """Join headings split over two lines ('L1 — Measuring ... Hidden' + 'Parameters')."""
    out: list[str] = []
    i = 0
    while i < len(paras):
        cur = paras[i]
        if is_heading(cur) and not cur.endswith(".") and i + 1 < len(paras):
            nxt = paras[i + 1]
            if len(nxt) < 60 and not nxt.endswith(".") and not is_heading(nxt) \
                    and not MARKER.search(nxt) and nxt[:1].isupper():
                out.append(f"{cur} {nxt}")
                i += 2
                continue
        out.append(cur)
        i += 1
    return out


def split_document(paras: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Return (preamble/queries, results, references)."""
    try:
        i_res = next(i for i, p in enumerate(paras) if p.strip() == "Results")
    except StopIteration:
        i_res = 0
    try:
        i_ref = next(i for i, p in enumerate(paras) if p.strip() == "References" and i > i_res)
    except StopIteration:
        i_ref = len(paras)
    return paras[:i_res], paras[i_res + 1:i_ref], paras[i_ref + 1:]


def parse_reference_entry(marker: str, body: str) -> dict:
    """`TITLE. AUTHORS. [VENUE ](YEAR). URL Context: "..."` with variants."""
    out: dict = {"markers": [marker], "raw_entry": f"[{marker}] {body}".strip()}
    context = None
    if " Context: " in body:
        head, context = body.split(" Context: ", 1)
        context = context.strip().strip('"')
    else:
        head = body
    urls = URL_RE.findall(head)
    if urls:
        out["url"] = urls[0].rstrip(".,;")
        head = URL_RE.sub("", head)
    if out.get("url"):
        m = DOI_RE.search(out["url"])
        if m:
            out["doi"] = m.group(0).rstrip(".")
    head = head.strip()
    year = None
    m = YEAR_TAIL.search(head)
    if m:
        year = int(m.group(1))
        head = head[: m.start()].strip()
    else:
        m = YEAR_BARE.search(head)
        if m:
            year = int(m.group(1))
            head = head[: m.start()].strip()
    head = head.rstrip(". ").strip()
    raw_parts = [p.strip() for p in head.split(". ") if p.strip()]
    parts: list[str] = []
    for part in raw_parts:
        # "Mark J" + "C" + "Hendrikx, ..." -> "Mark J. C. Hendrikx, ..."
        if parts and re.search(r"\b[A-Z]$", parts[-1]):
            parts[-1] = f"{parts[-1]}. {part}"
        else:
            parts.append(part)
    if parts:
        out["title"] = parts[0].rstrip(".")
    if len(parts) > 1:
        authors_raw = parts[1]
        out["authors"] = [a.strip() for a in re.split(r",| and ", authors_raw) if a.strip()]
    if len(parts) > 2:
        out["venue"] = ". ".join(parts[2:]).strip()
    out["year"] = year
    if context:
        out["context_quotes"] = [context]
    return out


def parse_references(paras: list[str]) -> dict[str, dict]:
    """Group [n.m] entries by group n: one group = one work."""
    entries: list[tuple[str, str]] = []
    cur_marker, cur_body = None, []
    for par in paras:
        m = REF_ENTRY.match(par)
        if m:
            if cur_marker:
                entries.append((cur_marker, " ".join(cur_body)))
            cur_marker, cur_body = m.group(1), [m.group(2)]
        elif cur_marker:
            cur_body.append(par)
    if cur_marker:
        entries.append((cur_marker, " ".join(cur_body)))

    groups: dict[str, dict] = {}
    for marker, body in entries:
        parsed = parse_reference_entry(marker, body)
        gid = marker.split(".")[0]
        if gid not in groups:
            groups[gid] = parsed
            groups[gid]["marker_group"] = gid
        else:
            g = groups[gid]
            g["markers"].append(marker)
            g.setdefault("context_quotes", []).extend(parsed.get("context_quotes", []))
            for field in ("title", "authors", "year", "venue", "url", "doi"):
                if not g.get(field) and parsed.get(field):
                    g[field] = parsed[field]
    return groups


def find_inline_mentions(text: str) -> list[str]:
    found: list[str] = []
    for names, year in INLINE_MENTION.findall(text):
        head = names.split(",")[0].split(" ")[0]
        if head in STOPNAMES:
            continue
        norm = re.sub(r"\s+", " ", names).strip().rstrip(",")
        found.append(f"{norm} {year}")
    return found


def ingest_report(pdf: Path, report_id: str, extracted_dir: Path) -> tuple[list[KosmosClaim], list[Reference]]:
    raw_lines = dehyphenate_join(read_pdf(pdf))
    raw_lines = [l for l in raw_lines if not l.startswith("<<<PAGE")]
    paras = merge_wrapped_headings(paragraphs(raw_lines))
    extracted_dir.mkdir(parents=True, exist_ok=True)
    (extracted_dir / f"{report_id}.txt").write_text("\n\n".join(paras), encoding="utf-8")

    preamble, results, refs = split_document(paras)
    ref_groups = parse_references(refs)

    claims: list[KosmosClaim] = []
    query, section = "meta", "Objective"
    for par in results:
        if is_heading(par):
            mq = H_QUERY.match(par)
            if mq:
                query, section = mq.group(1), mq.group(2).strip()
                continue
            mn = H_NUM.match(par)
            if mn:
                section = f"{mn.group(1)}. {mn.group(2).strip()}"
                continue
            section = par
            continue
        for sent in sentences(par):
            markers = MARKER.findall(sent)
            mentions = find_inline_mentions(sent)
            if not markers and not mentions:
                continue
            cid = f"{report_id}-c{len(claims) + 1:03d}"
            claims.append(KosmosClaim(
                claim_id=cid, report_id=report_id, query=query, section=section,
                text=sent, markers=markers,
                marker_groups=sorted({m.split('.')[0] for m in markers}, key=int),
                inline_mentions=mentions,
            ))

    references: list[Reference] = []
    seen_inline: dict[str, Reference] = {}
    for gid, data in sorted(ref_groups.items(), key=lambda kv: int(kv[0])):
        cited_by = [c for c in claims if gid in c.marker_groups]
        title = data.get("title") or ""
        year = data.get("year")
        ref = Reference(
            ref_id=f"{report_id}-g{gid}",
            report_id=report_id, ref_kind="listed", marker_group=gid,
            markers=data.get("markers", []), raw_entry=data.get("raw_entry"),
            title=title or None, authors=data.get("authors", []), year=year,
            venue=data.get("venue"), doi=data.get("doi"), url=data.get("url"),
            context_quotes=data.get("context_quotes", []),
            claim_ids=[c.claim_id for c in cited_by],
            queries=sorted({c.query for c in cited_by}),
            n_citations=sum(len([m for m in c.markers if m.split(".")[0] == gid]) for c in cited_by),
            needs_review=not title or year is None or (year is not None and not 1900 <= year <= 2030),
            review_note=None if title and year and 1900 <= year <= 2030 else "title or year not parseable",
            paper_id=make_paper_id(data.get("authors", []), year, title),
        )
        references.append(ref)

    for claim in claims:
        for mention in claim.inline_mentions:
            key = mention.lower()
            if key in seen_inline:
                seen_inline[key].claim_ids.append(claim.claim_id)
                seen_inline[key].n_citations += 1
                if claim.query not in seen_inline[key].queries:
                    seen_inline[key].queries.append(claim.query)
                continue
            # Surnames in the mention: "Leahu, Kaisers, and Baarslag 2019" -> all three.
            names = re.sub(r"\s+et\s+al\.?", "", mention.rsplit(" ", 1)[0])
            surnames = [n.strip().lower() for n in re.split(r",|\band\b|&", names) if n.strip()]
            year = int(mention.split()[-1])

            def author_surnames(ref: Reference) -> set[str]:
                return {a.split()[-1].lower() for a in ref.authors if a.split()}

            # *All* the mentioned surnames must be among the authors: otherwise
            # "Ding and Lehrer 2026" would link to a Konishi et al. 2026 that includes a Ding.
            match = next(
                (r for r in references
                 if r.ref_kind == "listed" and r.year == year
                 and all(sn in author_surnames(r) for sn in surnames)),
                None,
            )
            surname = surnames[0] if surnames else ""
            ref = Reference(
                ref_id=f"{report_id}-i{len(seen_inline) + 1:03d}",
                report_id=report_id, ref_kind="inline",
                mention_string=mention, year=year,
                authors=[surname.capitalize()],
                claim_ids=[claim.claim_id], queries=[claim.query], n_citations=1,
                needs_review=match is None,
                resolve_status="resolved" if match else "unresolved",
                title=match.title if match else None,
                doi=match.doi if match else None,
                url=match.url if match else None,
                paper_id=match.paper_id if match else None,
                review_note=(f"named in the body; matches entry {match.ref_id}"
                             if match else "named in the body; no formal entry"),
            )
            seen_inline[key] = ref
    references.extend(seen_inline.values())
    return claims, references


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest the Kosmos reports (task 1.1)")
    ap.add_argument("--kosmos-dir", default="knowledge/kosmos")
    ap.add_argument("--out-dir", default="knowledge/db")
    ap.add_argument("--extracted-dir", default="knowledge/extracted/kosmos")
    args = ap.parse_args()

    kosmos_dir, out_dir = Path(args.kosmos_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(kosmos_dir.rglob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs in {kosmos_dir}")

    all_claims: list[KosmosClaim] = []
    all_refs: list[Reference] = []
    for pdf in pdfs:
        report_id = slug(pdf.stem)
        claims, refs = ingest_report(pdf, report_id, Path(args.extracted_dir))
        all_claims.extend(claims)
        all_refs.extend(refs)
        listed = sum(1 for r in refs if r.ref_kind == "listed")
        inline = sum(1 for r in refs if r.ref_kind == "inline")
        print(f"{pdf.name}: {len(claims)} claims, {listed} listed references, {inline} inline")

    with (out_dir / "kosmos_claims.jsonl").open("w", encoding="utf-8") as fh:
        for claim in all_claims:
            fh.write(claim.model_dump_json() + "\n")
    with (out_dir / "references.jsonl").open("w", encoding="utf-8") as fh:
        for ref in all_refs:
            fh.write(ref.model_dump_json() + "\n")
    print(f"-> {out_dir/'kosmos_claims.jsonl'} and {out_dir/'references.jsonl'}")


if __name__ == "__main__":
    main()
