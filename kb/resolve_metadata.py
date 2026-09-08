"""Task 1.2 — Resolve reference metadata against OpenAlex and Crossref.

Two paths:
  * `listed` with a DOI  -> direct lookup by DOI (exact).
  * `listed` without DOI or `inline` -> text search (surname + terms from the claim that cites
    it), filtered by author surname and year ±1. The match is scored, and anything below the
    threshold stays `ambiguous`/`not_found` with `needs_review`.

No personal e-mail is sent to these services (see DECISIONS D-008).

Usage:  python -m kb.resolve_metadata [--limit N] [--refresh]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from kb.ids import make_paper_id
from kb.schemas import KosmosClaim, Reference

OPENALEX = "https://api.openalex.org/works"
CROSSREF = "https://api.crossref.org/works"
CACHE_PATH = Path("knowledge/db/resolve_cache.json")
USER_AGENT = "negotiation-foxes-kb/0.1 (research prototype; contact via repository owner)"

STOPWORDS = set("""a an the of and or in on for to with by from as at is are was were be been
that this these those which who whom whose what when where how why not no than then thus
their its his her our your it they them he she we you i also more most such some any all
using used use uses study studies work works paper papers approach approaches method methods
model models framework result results finding findings show shows showed found reported
report reports describe describes described introduce introduces introduced propose proposes
proposed apply applies applied et al""".split())


def content_words(text: str, limit: int = 10) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text)
    out: list[str] = []
    for w in words:
        lw = w.lower()
        if lw in STOPWORDS or lw in out:
            continue
        out.append(lw)
        if len(out) >= limit:
            break
    return out


class Resolver:
    def __init__(self, refresh: bool = False) -> None:
        self.cache: dict = {} if refresh else self._load_cache()
        self.client = httpx.Client(timeout=30.0, headers={"User-Agent": USER_AGENT},
                                   follow_redirects=True)

    @staticmethod
    def _load_cache() -> dict:
        if CACHE_PATH.exists():
            return json.loads(CACHE_PATH.read_text())
        return {}

    def save_cache(self) -> None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(self.cache, indent=1, sort_keys=True))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20))
    def _get(self, url: str, params: dict | None = None) -> dict | None:
        resp = self.client.get(url, params=params)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        time.sleep(0.15)  # courtesy towards the public pool
        return resp.json()

    def openalex_by_doi(self, doi: str) -> dict | None:
        key = f"oa_doi:{doi.lower()}"
        if key not in self.cache:
            self.cache[key] = self._get(f"{OPENALEX}/https://doi.org/{doi.lower()}")
        return self.cache[key]

    def openalex_search(self, query: str, year: int | None) -> list[dict]:
        key = f"oa_q:{query}|{year}"
        if key not in self.cache:
            params = {"search": query, "per-page": 8}
            if year:
                params["filter"] = f"from_publication_date:{year - 1}-01-01," \
                                   f"to_publication_date:{year + 1}-12-31"
            data = self._get(OPENALEX, params)
            self.cache[key] = (data or {}).get("results", [])
        return self.cache[key]

    def crossref_search(self, query: str, year: int | None) -> list[dict]:
        key = f"cr_q:{query}|{year}"
        if key not in self.cache:
            data = self._get(CROSSREF, {"query.bibliographic": query, "rows": 5})
            self.cache[key] = (data or {}).get("message", {}).get("items", [])
        return self.cache[key]


def oa_authors(work: dict) -> list[str]:
    return [a.get("author", {}).get("display_name", "") for a in work.get("authorships", [])]


def oa_pdf(work: dict) -> tuple[bool, str | None]:
    loc = work.get("best_oa_location") or work.get("primary_location") or {}
    return bool(work.get("open_access", {}).get("is_oa")), loc.get("pdf_url")


def all_pdf_urls(work: dict) -> list[str]:
    """Every location with a PDF that OpenAlex declares, without duplicates."""
    urls: list[str] = []
    for loc in (work.get("locations") or []):
        url = loc.get("pdf_url")
        if url and url not in urls:
            urls.append(url)
    oa_url = (work.get("open_access") or {}).get("oa_url")
    if oa_url and oa_url not in urls:
        urls.append(oa_url)
    return urls


def title_overlap(work: dict, title_hint: str | None) -> float:
    """Fraction of the hint's content words that appear in the candidate title."""
    if not title_hint:
        return 0.0
    hint = set(content_words(title_hint, 14))
    if not hint:
        return 0.0
    title = (work.get("title") or work.get("display_name") or "").lower()
    return len(hint & set(content_words(title, 30))) / len(hint)


def score_match(work: dict, surname: str, year: int | None, title_hint: str | None) -> float:
    """0–1. The surname alone does not reach the threshold: topical overlap is required.

    Weights: surname 0.35, year 0.15, title overlap 0.50. A common surname ('Zhang 2025') with
    no topical relation stays at 0.50 and does not cross the 0.55 threshold.
    """
    score = 0.0
    authors = " ".join(oa_authors(work)).lower()
    if surname and surname.lower() in authors:
        score += 0.35
    wyear = work.get("publication_year")
    if year and wyear == year:
        score += 0.15
    elif year and wyear and abs(wyear - year) <= 2:
        score += 0.08
    score += 0.50 * title_overlap(work, title_hint)
    return round(min(score, 1.0), 3)


def crossref_to_oa(item: dict) -> dict:
    """Normalise a Crossref item to the shape the rest of the module uses."""
    authors = [
        {"author": {"display_name": f"{a.get('given', '')} {a.get('family', '')}".strip()}}
        for a in item.get("author", [])
    ]
    year = None
    parts = (item.get("issued") or {}).get("date-parts") or [[]]
    if parts and parts[0]:
        year = parts[0][0]
    return {
        "id": None,
        "title": (item.get("title") or [""])[0],
        "authorships": authors,
        "publication_year": year,
        "doi": item.get("DOI"),
        "primary_location": {"source": {"display_name": (item.get("container-title") or [""])[0]}},
        "open_access": {"is_oa": False},
        "_source": "crossref",
    }


MIN_OVERLAP_LISTED = 0.60   # we know the exact title: the candidate must reflect it
MIN_OVERLAP_INLINE = 0.25   # we only have author+year: real topical overlap is required


def resolve_reference(ref: Reference, claims_by_id: dict[str, KosmosClaim],
                      resolver: Resolver, threshold: float = 0.55,
                      hints: dict[str, dict] | None = None) -> Reference:
    if ref.doi:
        work = resolver.openalex_by_doi(ref.doi)
        if work:
            return apply_work(ref, work, 1.0, "resolved")

    surname = ""
    if ref.ref_kind == "listed" and ref.authors:
        surname = ref.authors[0].split()[-1]
    elif ref.mention_string:
        surname = ref.mention_string.split(",")[0].split(" ")[0]

    claim_text = " ".join(
        claims_by_id[cid].text for cid in ref.claim_ids[:2] if cid in claims_by_id
    )
    hint = (hints or {}).get((ref.mention_string or "").lower())
    hint_title = hint.get("title") if hint else None
    title_hint = ref.title or hint_title or claim_text

    # Several query strategies: the title if there is one, and the surname with the terms of
    # the claim that cites it. No year filter on the second pass (the reports mix preprint and
    # publication years).
    queries: list[tuple[str, int | None]] = []
    if hint_title:
        queries.append((hint_title, None))
    if ref.title:
        queries.append((ref.title, ref.year))
        queries.append((ref.title, None))
    q_claim = " ".join([surname] + content_words(claim_text, 9)).strip()
    if q_claim:
        queries.append((q_claim, ref.year))
        queries.append((q_claim, None))

    candidates: list[dict] = []
    for query, year_filter in queries:
        if not query.strip():
            continue
        candidates.extend(resolver.openalex_search(query, year_filter))
        if len(candidates) >= 8 and any(
            score_match(w, surname, ref.year, title_hint) >= threshold for w in candidates
        ):
            break
    if not any(score_match(w, surname, ref.year, title_hint) >= threshold for w in candidates):
        for query, _ in queries[:3]:
            candidates.extend(crossref_to_oa(i) for i in resolver.crossref_search(query, ref.year))

    scored = sorted(
        ((score_match(w, surname, ref.year, title_hint), w) for w in candidates),
        key=lambda t: -t[0],
    )
    tried = "; ".join(q for q, _ in queries[:2])
    if not scored or scored[0][0] < threshold:
        ref.resolve_status = "not_found" if not scored else "ambiguous"
        ref.needs_review = True
        ref.match_score = scored[0][0] if scored else None
        ref.review_note = (ref.review_note or "") + f" | no reliable match (queries: {tried})"
        return ref
    best_score, best = scored[0]
    overlap = title_overlap(best, title_hint)
    min_overlap = MIN_OVERLAP_LISTED if ref.ref_kind == "listed" else MIN_OVERLAP_INLINE
    if overlap < min_overlap:
        ref.needs_review = True
        ref.review_note = (ref.review_note or "") + (
            f" | best candidate has low title overlap ({overlap:.2f} < {min_overlap})")
        return apply_work(ref, best, best_score, "ambiguous")
    # Two records of the same work (preprint + published version, or duplicate indices) are
    # not an ambiguity: only a candidate with a materially different title is.
    def same_work(a: dict, b: dict) -> bool:
        if (a.get("doi") and a.get("doi") == b.get("doi")) or (a.get("id") and a.get("id") == b.get("id")):
            return True
        ta, tb = set(content_words(a.get("title") or "", 20)), set(content_words(b.get("title") or "", 20))
        if not ta or not tb:
            return False
        return len(ta & tb) / len(ta | tb) >= 0.7

    distinct = [t for t in scored if not same_work(t[1], best)]
    if distinct and distinct[0][0] >= best_score - 0.05:
        ref.review_note = (ref.review_note or "") + " | two candidates with similar scores"
        ref.needs_review = True
        return apply_work(ref, best, best_score, "ambiguous")
    if hint_title:
        ref.review_note = (ref.review_note or "") + " | resolved with a verified manual hint"
    return apply_work(ref, best, best_score, "resolved")


def apply_work(ref: Reference, work: dict, score: float, status: str) -> Reference:
    is_oa, pdf_url = oa_pdf(work)
    ref.openalex_id = work.get("id")
    ref.resolved_title = work.get("title") or work.get("display_name")
    ref.resolved_authors = oa_authors(work)
    ref.resolved_year = work.get("publication_year")
    ref.resolved_venue = ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
    ref.is_oa = is_oa
    ref.oa_pdf_url = pdf_url
    ref.alt_pdf_urls = [u for u in all_pdf_urls(work) if u != pdf_url]
    ref.match_score = score
    ref.resolve_status = status  # type: ignore[assignment]
    if not ref.doi and work.get("doi"):
        ref.doi = work["doi"].replace("https://doi.org/", "")
    if status == "resolved":
        ref.needs_review = False
    if not ref.year and ref.resolved_year:
        ref.year = ref.resolved_year
    if status == "resolved":
        # The paper id is fixed from the confirmed metadata: the report's year can be wrong
        # (e.g. "ArXiv (2605)" from the identifier 2605.04507).
        ref.paper_id = make_paper_id(
            ref.resolved_authors or ref.authors,
            ref.resolved_year or ref.year,
            ref.resolved_title or ref.title,
        )
    return ref


def main() -> None:
    ap = argparse.ArgumentParser(description="Resolve reference metadata (task 1.2)")
    ap.add_argument("--refs", default="knowledge/db/references.jsonl")
    ap.add_argument("--claims", default="knowledge/db/kosmos_claims.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--hints", default="knowledge/db/manual_hints.yaml")
    args = ap.parse_args()

    refs = [Reference(**json.loads(l)) for l in Path(args.refs).read_text().splitlines() if l.strip()]
    claims = {c.claim_id: c for c in (
        KosmosClaim(**json.loads(l)) for l in Path(args.claims).read_text().splitlines() if l.strip()
    )}
    hints: dict[str, dict] = {}
    hints_path = Path(args.hints)
    if hints_path.exists():
        import yaml
        raw = yaml.safe_load(hints_path.read_text()) or {}
        hints = {k.lower(): v for k, v in raw.items()}
        print(f"manual hints loaded: {len(hints)}")
    resolver = Resolver(refresh=args.refresh)

    todo = [r for r in refs if r.resolve_status == "unresolved"]
    if args.limit:
        todo = todo[: args.limit]
    print(f"resolving {len(todo)} of {len(refs)} references…")
    try:
        for i, ref in enumerate(todo, 1):
            resolve_reference(ref, claims, resolver, hints=hints)
            flag = {"resolved": "ok", "ambiguous": "~", "not_found": "x"}.get(ref.resolve_status, "?")
            label = ref.mention_string or (ref.title or "")[:48]
            print(f"[{i:>2}/{len(todo)}] {flag} {label[:52]:52} -> "
                  f"{(ref.resolved_title or '')[:44]} ({ref.match_score})")
    finally:
        resolver.save_cache()

    # One work = one paper_id: several references (formal group + inline mentions) can point
    # to the same DOI with different years (online-first vs printed issue).
    canonical: dict[str, str] = {}
    for ref in sorted(refs, key=lambda r: (r.ref_kind != "listed", r.ref_id)):
        key = (ref.doi or "").lower() or (ref.openalex_id or "")
        if not key or not ref.paper_id:
            continue
        canonical.setdefault(key, ref.paper_id)
    merged = 0
    for ref in refs:
        key = (ref.doi or "").lower() or (ref.openalex_id or "")
        if key in canonical and ref.paper_id != canonical[key]:
            ref.paper_id = canonical[key]
            merged += 1
    if merged:
        print(f"unified {merged} duplicate paper_ids (same work, different record)")

    with Path(args.refs).open("w", encoding="utf-8") as fh:
        for ref in refs:
            fh.write(ref.model_dump_json() + "\n")
    ok = sum(1 for r in refs if r.resolve_status == "resolved")
    print(f"\nresolved {ok}/{len(refs)} ({ok / len(refs):.0%}); "
          f"{sum(1 for r in refs if r.needs_review)} flagged needs_review")


if __name__ == "__main__":
    main()
