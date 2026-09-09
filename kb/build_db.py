"""Task 1.4 (b) — The knowledge base database.

Builds knowledge/db/papers.sqlite from the artifacts of tasks 1.1–1.4a. It is *regenerable*:
it is deleted and rebuilt from references.jsonl, kosmos_claims.jsonl and knowledge/extracted/.
The JSONL files are the source of truth; the database is the query index.

It includes a full-text index (FTS5) over the sections of each paper. For now it stands in for
the KB's GraphRAG, which requires LLM/embedding credentials (DECISIONS D-016).

Usage:  python -m kb.build_db
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import date
from pathlib import Path

from kb.schemas import KosmosClaim, Reference

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT, year INTEGER, venue TEXT, doi TEXT, openalex_id TEXT,
    is_oa INTEGER, fetch_status TEXT, pdf_path TEXT, pdf_sha256 TEXT,
    extracted_path TEXT, extraction_quality TEXT, n_words INTEGER
);
CREATE TABLE authors (author_id INTEGER PRIMARY KEY, name TEXT UNIQUE);
CREATE TABLE paper_authors (
    paper_id TEXT, author_id INTEGER, position INTEGER,
    PRIMARY KEY (paper_id, author_id),
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id),
    FOREIGN KEY (author_id) REFERENCES authors(author_id)
);
CREATE TABLE kosmos_reports (report_id TEXT PRIMARY KEY, source_path TEXT, n_claims INTEGER);
CREATE TABLE kosmos_claims (
    claim_id TEXT PRIMARY KEY, report_id TEXT, query TEXT, section TEXT, text TEXT,
    FOREIGN KEY (report_id) REFERENCES kosmos_reports(report_id)
);
CREATE TABLE kosmos_refs (
    ref_id TEXT PRIMARY KEY, report_id TEXT, ref_kind TEXT, marker_group TEXT,
    markers TEXT, mention_string TEXT, raw_title TEXT, paper_id TEXT,
    resolve_status TEXT, match_score REAL, needs_review INTEGER, review_note TEXT,
    n_citations INTEGER, queries TEXT, fetch_status TEXT,
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
);
CREATE TABLE claim_citations (
    claim_id TEXT, ref_id TEXT, marker TEXT,
    PRIMARY KEY (claim_id, ref_id, marker)
);
CREATE TABLE paper_sections (
    section_id INTEGER PRIMARY KEY, paper_id TEXT, ord INTEGER, heading TEXT, body TEXT,
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
);
-- Filled by later tasks (1.5 annotation, 1.9 datasets, 1.7 catalog).
CREATE TABLE models (
    model_id TEXT PRIMARY KEY, paper_id TEXT, name TEXT, estimates TEXT, phase TEXT,
    inputs_required TEXT, assumptions TEXT, evaluation TEXT, benchmark TEXT,
    code_available TEXT, data_available TEXT, complexity TEXT, relevance TEXT,
    research_question TEXT, notes TEXT,
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
);
CREATE TABLE datasets (
    dataset_id TEXT PRIMARY KEY, name TEXT, path TEXT, description TEXT,
    n_rows INTEGER, generated_by TEXT, seed INTEGER, created_at TEXT
);
CREATE TABLE files (
    file_id INTEGER PRIMARY KEY, paper_id TEXT, kind TEXT, path TEXT, sha256 TEXT, bytes INTEGER
);
CREATE TABLE annotations (
    annotation_id INTEGER PRIMARY KEY, paper_id TEXT, key TEXT, value TEXT,
    evidence TEXT, created_by TEXT, created_at TEXT
);
CREATE VIRTUAL TABLE paper_fts USING fts5(paper_id, heading, body, tokenize='porter');
CREATE INDEX idx_refs_paper ON kosmos_refs(paper_id);
CREATE INDEX idx_claims_query ON kosmos_claims(query);
CREATE INDEX idx_sections_paper ON paper_sections(paper_id);
"""

FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.S)


def parse_extracted(path: Path) -> tuple[dict, list[tuple[str, str]]]:
    """Return (frontmatter, [(heading, body)])."""
    raw = path.read_text(encoding="utf-8")
    meta: dict = {}
    m = FRONTMATTER.match(raw)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        raw = raw[m.end():]
    sections: list[tuple[str, str]] = []
    heading, buf = "(no heading)", []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("## "):
            if buf:
                sections.append((heading, "\n\n".join(buf)))
                buf = []
            heading = block[3:].strip()
        else:
            buf.append(block)
    if buf:
        sections.append((heading, "\n\n".join(buf)))
    return meta, sections


def main() -> None:
    ap = argparse.ArgumentParser(description="Build papers.sqlite (task 1.4)")
    ap.add_argument("--refs", default="knowledge/db/references.jsonl")
    ap.add_argument("--claims", default="knowledge/db/kosmos_claims.jsonl")
    ap.add_argument("--extracted", default="knowledge/extracted")
    ap.add_argument("--db", default="knowledge/db/papers.sqlite")
    args = ap.parse_args()

    db_path = Path(args.db)
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(db_path) + suffix)
        if p.exists():
            p.unlink()
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)

    refs = [Reference(**json.loads(l))
            for l in Path(args.refs).read_text().splitlines() if l.strip()]
    claims = [KosmosClaim(**json.loads(l))
              for l in Path(args.claims).read_text().splitlines() if l.strip()]

    # Reports and claims
    reports = {c.report_id for c in claims}
    for report_id in sorted(reports):
        n = sum(1 for c in claims if c.report_id == report_id)
        src = next(iter(Path("knowledge/kosmos").rglob("*.pdf")), None)
        conn.execute("INSERT INTO kosmos_reports VALUES (?,?,?)",
                     (report_id, str(src) if src else None, n))
    conn.executemany("INSERT INTO kosmos_claims VALUES (?,?,?,?,?)",
                     [(c.claim_id, c.report_id, c.query, c.section, c.text) for c in claims])

    # Papers (one row per distinct resolved work)
    papers: dict[str, Reference] = {}
    for ref in refs:
        if not ref.paper_id:
            continue
        keep = papers.get(ref.paper_id)
        if keep is None or (ref.ref_kind == "listed" and keep.ref_kind != "listed"):
            papers[ref.paper_id] = ref
    author_ids: dict[str, int] = {}
    for paper_id, ref in sorted(papers.items()):
        extracted = Path(args.extracted) / f"{paper_id}.md"
        meta, sections = ({}, [])
        if extracted.exists():
            meta, sections = parse_extracted(extracted)
        n_words = sum(len(b.split()) for _, b in sections)
        conn.execute("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            paper_id, ref.resolved_title or ref.title, ref.resolved_year or ref.year,
            ref.resolved_venue or ref.venue, ref.doi, ref.openalex_id,
            int(bool(ref.is_oa)), ref.fetch_status, ref.pdf_path, ref.pdf_sha256,
            str(extracted) if extracted.exists() else None,
            meta.get("extraction_quality"), n_words,
        ))
        for pos, name in enumerate(ref.resolved_authors or ref.authors):
            if name not in author_ids:
                cur = conn.execute("INSERT OR IGNORE INTO authors(name) VALUES (?)", (name,))
                author_ids[name] = cur.lastrowid or conn.execute(
                    "SELECT author_id FROM authors WHERE name=?", (name,)).fetchone()[0]
            conn.execute("INSERT OR IGNORE INTO paper_authors VALUES (?,?,?)",
                         (paper_id, author_ids[name], pos))
        for ord_, (heading, body) in enumerate(sections):
            conn.execute("INSERT INTO paper_sections(paper_id, ord, heading, body) VALUES (?,?,?,?)",
                         (paper_id, ord_, heading, body))
            conn.execute("INSERT INTO paper_fts VALUES (?,?,?)", (paper_id, heading, body))
        if ref.pdf_path:
            size = Path(ref.pdf_path).stat().st_size if Path(ref.pdf_path).exists() else None
            conn.execute("INSERT INTO files(paper_id, kind, path, sha256, bytes) VALUES (?,?,?,?,?)",
                         (paper_id, "pdf", ref.pdf_path, ref.pdf_sha256, size))
        if extracted.exists():
            conn.execute("INSERT INTO files(paper_id, kind, path, sha256, bytes) VALUES (?,?,?,?,?)",
                         (paper_id, "extracted_text", str(extracted), None,
                          extracted.stat().st_size))

    # References and citations
    for ref in refs:
        conn.execute("INSERT INTO kosmos_refs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            ref.ref_id, ref.report_id, ref.ref_kind, ref.marker_group,
            ",".join(ref.markers), ref.mention_string, ref.title, ref.paper_id,
            ref.resolve_status, ref.match_score, int(ref.needs_review), ref.review_note,
            ref.n_citations, ",".join(ref.queries), ref.fetch_status,
        ))
    by_group = {(r.report_id, r.marker_group): r for r in refs if r.marker_group}
    for claim in claims:
        for marker in claim.markers:
            ref = by_group.get((claim.report_id, marker.split(".")[0]))
            if ref:
                conn.execute("INSERT OR IGNORE INTO claim_citations VALUES (?,?,?)",
                             (claim.claim_id, ref.ref_id, marker))
        for mention in claim.inline_mentions:
            hit = next((r for r in refs
                        if r.ref_kind == "inline" and r.report_id == claim.report_id
                        and (r.mention_string or "").lower() == mention.lower()), None)
            if hit:
                conn.execute("INSERT OR IGNORE INTO claim_citations VALUES (?,?,?)",
                             (claim.claim_id, hit.ref_id, "inline"))

    conn.execute("INSERT INTO annotations(paper_id, key, value, evidence, created_by, created_at)"
                 " VALUES (NULL,?,?,?,?,?)",
                 ("db_build", "ok", f"{len(papers)} papers, {len(claims)} claims",
                  "kb.build_db", date.today().isoformat()))
    conn.commit()

    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in
              ("papers", "authors", "kosmos_claims", "kosmos_refs", "claim_citations",
               "paper_sections", "files")}
    print(f"{db_path}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    conn.close()


if __name__ == "__main__":
    main()
