"""Task 1.5 — Load and *verify* the per-paper annotations.

Validates before writing:
  * the paper_id exists in the papers table;
  * every `kosmos:<claim_id>` evidence exists in kosmos_claims;
  * every `paper:<paper_id>#<section>` evidence corresponds to a real extracted section.
An annotation whose evidence does not resolve is an error, not a warning: the load aborts.

Usage:  python -m kb.load_annotations [--check-only]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date
from pathlib import Path

import yaml


def validate(conn: sqlite3.Connection, data: dict) -> list[str]:
    errors: list[str] = []
    papers = {r[0] for r in conn.execute("SELECT paper_id FROM papers")}
    claims = {r[0] for r in conn.execute("SELECT claim_id FROM kosmos_claims")}
    sections = {(r[0], r[1]) for r in conn.execute("SELECT paper_id, heading FROM paper_sections")}
    # (claim_id, paper_id) pairs that the Kosmos report actually links
    cited = {(r[0], r[1]) for r in conn.execute(
        "SELECT cc.claim_id, kr.paper_id FROM claim_citations cc "
        "JOIN kosmos_refs kr ON kr.ref_id = cc.ref_id WHERE kr.paper_id IS NOT NULL")}

    for paper_id, ann in data.items():
        if paper_id not in papers:
            errors.append(f"{paper_id}: not in the papers table")
        for ev in ann.get("evidence", []):
            if ev.startswith("kosmos:"):
                cid = ev.split(":", 1)[1]
                if cid not in claims:
                    errors.append(f"{paper_id}: evidence does not exist {ev}")
                elif (cid, paper_id) not in cited:
                    errors.append(f"{paper_id}: claim {cid} does not cite that paper "
                                  f"(the evidence points to another work)")
            elif ev.startswith("paper:"):
                body = ev.split(":", 1)[1]
                pid, _, heading = body.partition("#")
                if pid not in papers:
                    errors.append(f"{paper_id}: evidence points to unknown paper {pid}")
                elif heading and (pid, heading) not in sections:
                    near = [h for p, h in sections if p == pid and heading.lower()[:18] in h.lower()]
                    errors.append(f"{paper_id}: section not found '{heading}' in {pid}"
                                  + (f" (did you mean '{near[0]}'?)" if near else ""))
            else:
                errors.append(f"{paper_id}: unrecognised evidence format: {ev}")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default="catalog/paper_annotations.yaml")
    ap.add_argument("--db", default="knowledge/db/papers.sqlite")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    data = yaml.safe_load(Path(args.annotations).read_text()) or {}
    conn = sqlite3.connect(args.db)
    errors = validate(conn, data)
    if errors:
        print(f"{len(errors)} validation problems:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)
    print(f"validation ok: {len(data)} papers, "
          f"{sum(len(a.get('evidence', [])) for a in data.values())} evidence pointers resolved")
    if args.check_only:
        return

    conn.execute("DELETE FROM models")
    conn.execute("DELETE FROM annotations WHERE created_by='kb.load_annotations'")
    today = date.today().isoformat()
    n_models = 0
    for paper_id, ann in data.items():
        evidence = "; ".join(ann.get("evidence", []))
        for i, model in enumerate(ann.get("models", []), 1):
            conn.execute("INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                f"{paper_id}-m{i}", paper_id, model.get("name"),
                json.dumps(model.get("estimates", []), ensure_ascii=False), model.get("phase"),
                json.dumps(model.get("inputs_required", []), ensure_ascii=False),
                json.dumps(model.get("assumptions", []), ensure_ascii=False),
                model.get("evaluation"), model.get("benchmark"),
                str(model.get("code_available")), str(model.get("data_available")),
                model.get("complexity"), model.get("relevance"),
                model.get("research_question"), model.get("notes"),
            ))
            n_models += 1
        conn.execute(
            "INSERT INTO annotations(paper_id, key, value, evidence, created_by, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (paper_id, "annotation", yaml.safe_dump(ann, allow_unicode=True, sort_keys=False),
             evidence, "kb.load_annotations", today))
    conn.commit()
    print(f"loaded {n_models} models and {len(data)} annotations into {args.db}")


if __name__ == "__main__":
    main()
