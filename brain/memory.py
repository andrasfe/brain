"""Hippocampus — episodic memory backed by SQLite.

Stores episodes (task, action, result, salience) and offers a simple
recency + keyword-overlap retrieval. Swap in a vector store later without
touching the regions that call retrieve()/store().
"""
from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Any

_WORD = re.compile(r"[a-z0-9]+")


class Memory:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        REAL NOT NULL,
                task      TEXT NOT NULL,
                kind      TEXT NOT NULL,
                content   TEXT NOT NULL,
                salience  REAL NOT NULL DEFAULT 0.5
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(ts);
            """
        )
        self.conn.commit()

    def store(self, task: str, kind: str, content: str, salience: float = 0.5) -> int:
        cur = self.conn.execute(
            "INSERT INTO episodes (ts, task, kind, content, salience) VALUES (?,?,?,?,?)",
            (time.time(), task, kind, content, float(salience)),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Rank episodes by keyword overlap with `query`, tie-broken by recency."""
        q_terms = set(_WORD.findall(query.lower()))
        rows = self.conn.execute(
            "SELECT * FROM episodes ORDER BY ts DESC LIMIT 500"
        ).fetchall()
        scored: list[tuple[float, float, sqlite3.Row]] = []
        for row in rows:
            terms = set(_WORD.findall((row["content"] + " " + row["task"]).lower()))
            overlap = len(q_terms & terms)
            if overlap == 0 and q_terms:
                continue
            scored.append((overlap * row["salience"], row["ts"], row))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [dict(r) for _, _, r in scored[:k]]

    def recent(self, k: int = 5) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM episodes ORDER BY ts DESC LIMIT ?", (k,)
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.conn.close()
