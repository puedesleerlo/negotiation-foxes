"""Task 1.3 — Download PDFs through legitimate routes.

Order of attempts per reference:
  1. OpenAlex's `oa_pdf_url` (best_oa_location: the publisher's or a repository's open-access version).
  2. arXiv, when the DOI is 10.48550/arXiv.XXXX or the URL points to arxiv.org.
  3. The URL carried by the Kosmos entry itself, if it is a direct link to a file
     (institutional repositories: tudelft, ucalgary, etc.).

No paywall is circumvented: without an open route the reference stays `missing` (or
`paywalled` if the publisher answered 403) and goes to the manual review queue.

Usage:  python -m kb.fetch_papers [--limit N] [--only-scope L1L2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import httpx

from kb.schemas import Reference

PAPERS_DIR = Path("knowledge/papers")
USER_AGENT = "negotiation-foxes-kb/0.1 (research prototype; respects robots and paywalls)"
ARXIV_DOI = re.compile(r"10\.48550/arxiv\.([\w.\-/]+)", re.I)
ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/([\w.\-/]+?)(?:v\d+)?(?:\.pdf)?$", re.I)


SEMANTIC_SCHOLAR = "https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
ARXIV_API = "http://export.arxiv.org/api/query"


def arxiv_by_title(title: str, surname: str, client: httpx.Client) -> str | None:
    """Search arXiv by title; the surname must appear among the authors."""
    if not title:
        return None
    try:
        resp = client.get(ARXIV_API, params={
            "search_query": f'ti:"{title[:120]}"', "max_results": 3,
        })
        if resp.status_code != 200:
            return None
        body = resp.text
    except Exception:
        return None
    entries = re.findall(r"<entry>(.*?)</entry>", body, re.S)
    for entry in entries:
        authors = " ".join(re.findall(r"<name>(.*?)</name>", entry)).lower()
        if surname and surname.lower() not in authors:
            continue
        m = re.search(r'<link[^>]*title="pdf"[^>]*href="([^"]+)"', entry)
        if m:
            return m.group(1)
        m = re.search(r"<id>(http[^<]*arxiv.org/abs/[^<]+)</id>", entry)
        if m:
            return m.group(1).replace("/abs/", "/pdf/")
    return None


def semantic_scholar_pdf(doi: str, client: httpx.Client) -> str | None:
    """Ask Semantic Scholar for the open-access route it knows (no key needed)."""
    try:
        resp = client.get(SEMANTIC_SCHOLAR.format(doi=doi), params={"fields": "openAccessPdf"})
        if resp.status_code != 200:
            return None
        return ((resp.json() or {}).get("openAccessPdf") or {}).get("url")
    except Exception:
        return None


def candidate_urls(ref: Reference) -> list[tuple[str, str]]:
    """[(source, url)] in order of preference."""
    out: list[tuple[str, str]] = []
    if ref.oa_pdf_url:
        out.append(("openalex_oa", ref.oa_pdf_url))
    for url in ref.alt_pdf_urls:
        out.append(("openalex_loc", url))
    arxiv_id = None
    if ref.doi and (m := ARXIV_DOI.search(ref.doi)):
        arxiv_id = m.group(1)
    elif ref.url and (m := ARXIV_URL.search(ref.url)):
        arxiv_id = m.group(1)
    if arxiv_id:
        out.append(("arxiv", f"https://arxiv.org/pdf/{arxiv_id}"))
    if ref.url and "doi.org" not in ref.url and ref.url not in [u for _, u in out]:
        out.append(("kosmos_entry", ref.url))
    return out


def fetch_one(ref: Reference, client: httpx.Client, force: bool = False) -> Reference:
    if ref.fetch_status == "downloaded" and not force and ref.pdf_path and Path(ref.pdf_path).exists():
        return ref
    paper_id = ref.paper_id or ref.ref_id
    target = PAPERS_DIR / f"{paper_id}.pdf"
    notes: list[str] = []
    candidates = candidate_urls(ref)
    if ref.resolve_status != "resolved":
        # With unconfirmed metadata, any PDF that comes from a search may belong to another
        # work. Only the URL carried by the Kosmos entry itself is accepted.
        candidates = [(src, url) for src, url in candidates if src == "kosmos_entry"]
        if not candidates:
            ref.fetch_status = "missing"
            ref.fetch_note = "metadata unresolved: not downloading, to avoid attaching the wrong PDF"
            return ref
    if not any(src in ("openalex_oa", "openalex_loc", "arxiv") for src, _ in candidates):
        if ref.doi:
            s2 = semantic_scholar_pdf(ref.doi, client)
            if s2:
                candidates.insert(0, ("semantic_scholar", s2))
        surname = (ref.authors or [""])[0].split()[-1] if ref.authors else (
            (ref.mention_string or "").split(",")[0].split(" ")[0])
        ax = arxiv_by_title(ref.resolved_title or ref.title or "", surname, client)
        if ax:
            candidates.append(("arxiv_search", ax))
    for source, url in candidates:
        try:
            resp = client.get(url)
        except Exception as exc:  # network, DNS, TLS
            notes.append(f"{source}: network error ({type(exc).__name__})")
            continue
        if resp.status_code in (401, 402, 403):
            notes.append(f"{source}: {resp.status_code} (restricted access)")
            continue
        if resp.status_code != 200:
            notes.append(f"{source}: HTTP {resp.status_code}")
            continue
        content = resp.content
        if not content.startswith(b"%PDF"):
            notes.append(f"{source}: response is not a PDF ({resp.headers.get('content-type','?')})")
            continue
        PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        ref.pdf_path = str(target)
        ref.pdf_sha256 = hashlib.sha256(content).hexdigest()
        ref.fetch_status = "downloaded"
        ref.fetch_note = f"{source} ({len(content) // 1024} KB)"
        time.sleep(0.3)
        return ref
    paywalled = any("restricted access" in n for n in notes)
    ref.fetch_status = "paywalled" if paywalled else ("failed" if notes else "missing")
    ref.fetch_note = "; ".join(notes) or "no candidate URL (metadata not resolved)"
    return ref


def main() -> None:
    ap = argparse.ArgumentParser(description="Download PDFs (task 1.3)")
    ap.add_argument("--refs", default="knowledge/db/references.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    path = Path(args.refs)
    refs = [Reference(**json.loads(l)) for l in path.read_text().splitlines() if l.strip()]
    # One work can appear as several references (different groups, inline mention): it is
    # downloaded once per paper_id.
    seen: dict[str, Reference] = {}
    todo: list[Reference] = []
    for ref in refs:
        key = ref.paper_id or ref.ref_id
        if key in seen:
            continue
        seen[key] = ref
        todo.append(ref)
    if args.limit:
        todo = todo[: args.limit]

    client = httpx.Client(timeout=60.0, follow_redirects=True, headers={"User-Agent": USER_AGENT})
    print(f"trying to download {len(todo)} distinct works…")
    for i, ref in enumerate(todo, 1):
        fetch_one(ref, client, force=args.force)
        mark = {"downloaded": "ok", "paywalled": "$", "missing": "-", "failed": "x"}[ref.fetch_status]
        label = (ref.title or ref.mention_string or ref.ref_id)[:50]
        print(f"[{i:>2}/{len(todo)}] {mark} {label:52} {ref.fetch_note or ''}"[:150])

    # Propagate the result to every reference that points to the same work.
    by_paper = {r.paper_id or r.ref_id: r for r in todo}
    for ref in refs:
        src = by_paper.get(ref.paper_id or ref.ref_id)
        if src and src is not ref:
            ref.fetch_status, ref.pdf_path = src.fetch_status, src.pdf_path
            ref.pdf_sha256, ref.fetch_note = src.pdf_sha256, src.fetch_note

    with path.open("w", encoding="utf-8") as fh:
        for ref in refs:
            fh.write(ref.model_dump_json() + "\n")
    ok = sum(1 for r in todo if r.fetch_status == "downloaded")
    print(f"\ndownloaded {ok}/{len(todo)} PDFs; "
          f"{sum(1 for r in todo if r.fetch_status == 'paywalled')} with restricted access, "
          f"{sum(1 for r in todo if r.fetch_status in ('missing', 'failed'))} without an open route")


if __name__ == "__main__":
    main()
