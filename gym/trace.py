"""The gym's trace: `events.jsonl` as the source of truth (§10.2).

Dual storage: the JSONL is append-only within a run and authoritative; the analytical DuckDB
tables are regenerated from it (`python -m gym.replay --rebuild-db`). Every event carries the
`program_id`, party, step and round, and goes through the leak detector before being written.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from gym.isolation import LeakDetector


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _plain(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):   # numpy scalars
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class Trace:
    def __init__(self, program_id: str, root: Path = Path("runs"),
                 detector: Optional[LeakDetector] = None, fresh: bool = False) -> None:
        self.program_id = program_id
        self.root = root / program_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "events.jsonl"
        self.detector = detector
        self.n_events = 0
        if fresh and self.path.exists():
            # A program is a deterministic function of (case, seed, parameters): running it
            # again reproduces the same trace instead of appending a second copy. Within one
            # run the file stays append-only, which is what the specification asks for.
            self.path.unlink()

    def emit(self, kind: str, *, party: Optional[str] = None, step: Optional[str] = None,
             round: Optional[int] = None, **payload: Any) -> dict:
        event = {
            "seq": self.n_events,
            "ts": datetime.now(timezone.utc).isoformat(),
            "program_id": self.program_id,
            "kind": kind, "party": party, "step": step, "round": round,
            "payload": _plain(payload),
        }
        # No event of one party may contain the other party's canary.
        if self.detector and party:
            self.detector.check(event["payload"], owner_role=party, context=f"event {kind}")
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.n_events += 1
        return event

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    def of_kind(self, kind: str) -> list[dict]:
        return [e for e in self.read() if e["kind"] == kind]

    def write_artifact(self, relative: str, content: str, *, party: Optional[str] = None) -> Path:
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.detector and party:
            self.detector.check(content, owner_role=party, context=f"artifact {relative}")
        target.write_text(content, encoding="utf-8")
        return target

    def write_json(self, relative: str, data: Any, *, party: Optional[str] = None) -> Path:
        return self.write_artifact(relative, json.dumps(_plain(data), indent=2,
                                                        ensure_ascii=False), party=party)


def rebuild_duckdb(program_ids: Iterable[str], runs_dir: Path = Path("runs"),
                   db_path: Path = Path("runs/trace.duckdb")) -> dict:
    """Regenerate the analytical tables from the JSONL files. The trace rules; the DB is derived."""
    import duckdb

    rows: list[dict] = []
    for program_id in program_ids:
        path = runs_dir / program_id / "events.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if db_path.exists():
        db_path.unlink()
    conn = duckdb.connect(str(db_path))
    conn.execute("""CREATE TABLE events (
        seq INTEGER, ts TIMESTAMP, program_id VARCHAR, kind VARCHAR,
        party VARCHAR, step VARCHAR, round INTEGER, payload JSON)""")
    conn.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?)",
                     [(r["seq"], r["ts"], r["program_id"], r["kind"], r["party"], r["step"],
                       r["round"], json.dumps(r["payload"], ensure_ascii=False)) for r in rows])
    conn.execute("""CREATE TABLE actions AS
        SELECT program_id, round, party,
               json_extract_string(payload, '$.action') AS action,
               json_extract_string(payload, '$.offers') AS offers,
               json_extract_string(payload, '$.message') AS message
        FROM events WHERE kind = 'action'""")
    conn.execute("""CREATE TABLE fox_calls AS
        SELECT program_id, round, party,
               json_extract_string(payload, '$.fox_id') AS fox_id,
               json_extract_string(payload, '$.version') AS version,
               CAST(json_extract(payload, '$.scope_ok') AS BOOLEAN) AS scope_ok,
               json_extract_string(payload, '$.summary') AS summary
        FROM events WHERE kind = 'fox_call'""")
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("events", "actions", "fox_calls")}
    conn.close()
    return counts
