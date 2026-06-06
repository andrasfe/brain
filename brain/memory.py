"""Hippocampus-backed memory — typed subsystems on a shared SQLite DB.

Real human memory isn't a single bucket. We carve the rows by `mem_type`:

  episodic    — autobiographical event traces (what I did, what happened to me)
  semantic    — distilled facts about the world or self (Maria is a doctor;
                I always struggle with shell quoting)
  prospective — intentions to do/remember later, with a trigger (when next
                I touch the auth code, recheck token expiry)
  affect      — affect-tagged associations (this address feels unsafe)
  source      — provenance metadata (where this fact came from, who told me)

Procedural memory (habits / motor skills) lives separately in `SkillStore`
because the lookup pattern is different — by percept signature, not by query.

Retrieval blends two ranking strategies:
  - keyword overlap (the original tolerant baseline)
  - TF-IDF cosine over `_tfidf` (richer "semantic" similarity, no deps)
A typed retrieve can restrict candidates to one or more `mem_types`.

Memory stays LLM-free: the consolidation pass that extracts semantic facts
from episodic clusters owns the LLM call (see `brain/consolidator.py`).
Memory just provides cheap, deterministic primitives.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .embeddings import EmbeddingBackend, TfidfBackend

_WORD = re.compile(r"[a-z0-9]+")

# Valid mem_types (centralized so the schema and prompts agree).
EPISODIC = "episodic"
SEMANTIC = "semantic"
PROSPECTIVE = "prospective"
AFFECT = "affect"
SOURCE = "source"
OBSERVATION = "observation"   # passive observation of the user's screen activity
_VALID_TYPES = {EPISODIC, SEMANTIC, PROSPECTIVE, AFFECT, SOURCE, OBSERVATION}


class Memory:
    """SQLite-backed typed memory. Auto-migrates older single-bucket schemas.

    Semantic retrieval is delegated to a pluggable `EmbeddingBackend` (see
    `brain/embeddings.py`). Default is TF-IDF — no deps, no API calls. The
    OpenRouter and sentence-transformers backends persist per-row embeddings
    in the `embedding` BLOB column so they're not recomputed across runs.
    """

    def __init__(self, db_path: Path, refit_every: int = 50,
                 backend: Optional[EmbeddingBackend] = None):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        self._migrate_schema()
        # Default backend = TF-IDF. The orchestrator typically passes in a
        # configured backend at construction (see `make_backend(cfg, llm)`).
        self.backend: EmbeddingBackend = backend or TfidfBackend()
        self._inserts_since_fit = 0
        self._refit_every = refit_every

    # ── schema ──────────────────────────────────────────────────────────────
    def _init_schema(self) -> None:
        """Create the base tables. Indexes that reference migrated columns
        are added in `_migrate_schema` AFTER the column ALTERs run, so a
        legacy single-bucket DB can be upgraded in place."""
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        REAL NOT NULL,
                task      TEXT NOT NULL,
                kind      TEXT NOT NULL,
                content   TEXT NOT NULL,
                salience  REAL NOT NULL DEFAULT 0.5,
                mem_type  TEXT NOT NULL DEFAULT 'episodic',
                affect_json TEXT,
                tags      TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(ts);

            CREATE TABLE IF NOT EXISTS prospective (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_ts      REAL NOT NULL,
                trigger_kind    TEXT NOT NULL,    -- 'keyword' | 'time' | 'percept'
                trigger_pattern TEXT NOT NULL,    -- query string / unix ts / percept gloss
                content         TEXT NOT NULL,
                salience        REAL NOT NULL DEFAULT 0.7,
                fires_after_ts  REAL,             -- for 'time' triggers
                fired_count     INTEGER NOT NULL DEFAULT 0,
                last_fired_ts   REAL,
                done            INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_prospective_kind ON prospective(trigger_kind);
            """
        )
        self.conn.commit()

    def _migrate_schema(self) -> None:
        """Older brains created `episodes` without mem_type/affect_json/tags/
        embedding. ALTER on-startup so prior data stays usable, then build any
        indexes that reference newly-added columns."""
        cols = {row["name"] for row in
                self.conn.execute("PRAGMA table_info(episodes)").fetchall()}
        with self.conn:
            if "mem_type" not in cols:
                self.conn.execute(
                    "ALTER TABLE episodes ADD COLUMN mem_type TEXT NOT NULL DEFAULT 'episodic'")
            if "affect_json" not in cols:
                self.conn.execute("ALTER TABLE episodes ADD COLUMN affect_json TEXT")
            if "tags" not in cols:
                self.conn.execute("ALTER TABLE episodes ADD COLUMN tags TEXT")
            if "embedding" not in cols:
                # BLOB cache for neural-embedding backends; NULL for TF-IDF
                self.conn.execute("ALTER TABLE episodes ADD COLUMN embedding BLOB")
            # Index referencing mem_type — safe to create now
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_type ON episodes(mem_type)")

    # ── insert / update ─────────────────────────────────────────────────────
    def store(self, task: str, kind: str, content: str, salience: float = 0.5,
              mem_type: str = EPISODIC,
              affect_at_encode: Optional[dict] = None,
              tags: Optional[Iterable[str]] = None) -> int:
        if mem_type not in _VALID_TYPES:
            mem_type = EPISODIC
        affect_blob = json.dumps(affect_at_encode) if affect_at_encode else None
        tag_blob = ",".join(sorted({t.strip().lower() for t in (tags or []) if t})) or None
        cur = self.conn.execute(
            "INSERT INTO episodes (ts, task, kind, content, salience, "
            "mem_type, affect_json, tags) VALUES (?,?,?,?,?,?,?,?)",
            (time.time(), task, kind, content, float(salience),
             mem_type, affect_blob, tag_blob),
        )
        self.conn.commit()
        self._inserts_since_fit += 1
        if self._inserts_since_fit >= self._refit_every:
            # Mark the active backend stale. For TF-IDF this triggers a full
            # rebuild; for neural backends, fit() embeds only missing rows.
            if isinstance(self.backend, TfidfBackend):
                self.backend.mark_stale()
            else:
                # Neural backends maintain their own cache; fit() will pick
                # the new row up on the next semantic query. Nothing to do.
                pass
        return int(cur.lastrowid)

    # ── keyword retrieval (kept for cheap default) ──────────────────────────
    def retrieve(self, query: str, k: int = 5,
                 types: Optional[Sequence[str]] = None) -> list[dict[str, Any]]:
        """Rank rows by keyword overlap × salience (tolerant fallback). When
        `types` given, restrict candidates to those mem_types."""
        q_terms = set(_WORD.findall(query.lower()))
        rows = self._candidates(types)
        scored: list[tuple[float, float, sqlite3.Row]] = []
        for row in rows:
            terms = set(_WORD.findall((row["content"] + " " + row["task"]).lower()))
            overlap = len(q_terms & terms)
            if overlap == 0 and q_terms:
                continue
            scored.append((overlap * row["salience"], row["ts"], row))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [self._row_to_dict(r) for _, _, r in scored[:k]]

    # ── semantic retrieval (pluggable backend) ──────────────────────────────
    def retrieve_semantic(self, query: str, k: int = 5,
                           types: Optional[Sequence[str]] = None,
                           min_score: float = 0.05) -> list[dict[str, Any]]:
        """Cosine over the active embedding backend. Persistable backends
        (OpenRouter, sentence-transformers) cache per-row vectors in the
        `embedding` BLOB column to avoid recomputing across runs.

        `types` restricts candidates; `min_score` filters weak matches."""
        rows = self._candidates(types)
        if not rows:
            return []

        # Load any cached embeddings into the neural backend's in-memory map
        # before fit() — so it doesn't waste API calls on rows we've seen.
        if self.backend.persistent:
            self._preload_cached_embeddings(rows)

        # Fit / refresh the backend's view of the eligible corpus.
        self.backend.fit(
            (int(r["id"]), (r["content"] + " " + r["task"])) for r in rows
        )
        self._inserts_since_fit = 0

        # Persist any newly-computed embeddings back to the row cache.
        if self.backend.persistent:
            self._persist_new_embeddings(rows)

        eligible_ids = {int(r["id"]) for r in rows}
        ranked = self.backend.topk(query, k=k * 2, eligible=eligible_ids)
        if not ranked:
            return []
        ids = [doc_id for doc_id, _ in ranked]
        placeholders = ",".join("?" * len(ids))
        rows_by_id = {
            int(r["id"]): r for r in self.conn.execute(
                f"SELECT * FROM episodes WHERE id IN ({placeholders})", ids
            ).fetchall()
        }
        results: list[dict[str, Any]] = []
        for doc_id, score in ranked:
            if score < min_score:
                continue
            row = rows_by_id.get(doc_id)
            if row is None:
                continue
            d = self._row_to_dict(row)
            d["score"] = round(float(score), 4)
            results.append(d)
            if len(results) >= k:
                break
        return results

    # ── embedding cache I/O for persistable backends ────────────────────────
    def _preload_cached_embeddings(self, rows) -> None:
        """If a neural backend, populate its in-memory map from the row's
        `embedding` BLOB column so we skip the API call."""
        bk = self.backend
        if not getattr(bk, "remember", None):
            return
        for r in rows:
            blob = r["embedding"] if "embedding" in r.keys() else None
            if not blob:
                continue
            vec = bk.from_bytes(blob)
            if vec:
                bk.remember(int(r["id"]), vec)

    def _persist_new_embeddings(self, rows) -> None:
        """Write back any embeddings computed by the backend on this fit
        that weren't already in the row's `embedding` BLOB column."""
        bk = self.backend
        vecs = getattr(bk, "_vecs", None)
        if not vecs:
            return
        from .embeddings import _pack_floats
        to_write: list[tuple[bytes, int]] = []
        for r in rows:
            rid = int(r["id"])
            if rid not in vecs:
                continue
            had_blob = "embedding" in r.keys() and r["embedding"]
            if had_blob:
                continue
            to_write.append((_pack_floats(vecs[rid]), rid))
        if not to_write:
            return
        self.conn.executemany(
            "UPDATE episodes SET embedding=? WHERE id=?", to_write)
        self.conn.commit()

    def _candidates(self, types: Optional[Sequence[str]]) -> list[sqlite3.Row]:
        if types:
            valid = [t for t in types if t in _VALID_TYPES]
            if not valid:
                return []
            placeholders = ",".join("?" * len(valid))
            return self.conn.execute(
                f"SELECT * FROM episodes WHERE mem_type IN ({placeholders}) "
                f"ORDER BY ts DESC LIMIT 800",
                valid,
            ).fetchall()
        return self.conn.execute(
            "SELECT * FROM episodes ORDER BY ts DESC LIMIT 800"
        ).fetchall()

    def recent(self, k: int = 5, mem_type: Optional[str] = None) -> list[dict[str, Any]]:
        if mem_type:
            rows = self.conn.execute(
                "SELECT * FROM episodes WHERE mem_type=? ORDER BY ts DESC LIMIT ?",
                (mem_type, k),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM episodes ORDER BY ts DESC LIMIT ?", (k,)
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count(self, mem_type: Optional[str] = None) -> int:
        if mem_type:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM episodes WHERE mem_type=?", (mem_type,)
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM episodes"
            ).fetchone()
        return int(row["n"])

    # ── prospective ─────────────────────────────────────────────────────────
    def prospective_register(self, content: str, trigger_kind: str,
                              trigger_pattern: str,
                              salience: float = 0.75,
                              fires_after_ts: Optional[float] = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO prospective (created_ts, trigger_kind, trigger_pattern, "
            "content, salience, fires_after_ts) VALUES (?,?,?,?,?,?)",
            (time.time(), trigger_kind, trigger_pattern,
             content, float(salience), fires_after_ts),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def prospective_match(self, percept_text: str,
                           now: Optional[float] = None,
                           min_overlap: int = 1) -> list[dict[str, Any]]:
        """Return pending prospective items whose triggers match now.

        - kind='keyword': any token in trigger_pattern appears in percept_text
        - kind='time': fires_after_ts <= now
        - kind='percept': substring match of trigger_pattern in percept_text
        """
        now = now if now is not None else time.time()
        rows = self.conn.execute(
            "SELECT * FROM prospective WHERE done=0 ORDER BY salience DESC LIMIT 50"
        ).fetchall()
        if not rows:
            return []
        text_lower = (percept_text or "").lower()
        text_tokens = set(_WORD.findall(text_lower))

        out: list[dict[str, Any]] = []
        for row in rows:
            kind = row["trigger_kind"]
            patt = (row["trigger_pattern"] or "").lower()
            fired = False
            if kind == "time":
                ft = row["fires_after_ts"]
                if ft is not None and float(ft) <= now:
                    fired = True
            elif kind == "percept":
                if patt and patt in text_lower:
                    fired = True
            elif kind == "keyword":
                patt_tokens = set(_WORD.findall(patt))
                if len(patt_tokens & text_tokens) >= min_overlap:
                    fired = True
            if fired:
                out.append({
                    "id": int(row["id"]),
                    "content": row["content"],
                    "trigger_kind": kind,
                    "trigger_pattern": row["trigger_pattern"],
                    "salience": float(row["salience"]),
                    "fired_count": int(row["fired_count"]),
                })
        return out

    def prospective_mark_fired(self, ids: Sequence[int],
                                mark_done: bool = False) -> None:
        if not ids:
            return
        now = time.time()
        for pid in ids:
            self.conn.execute(
                "UPDATE prospective SET fired_count = fired_count + 1, "
                "last_fired_ts=?, done = done OR ? WHERE id=?",
                (now, 1 if mark_done else 0, int(pid)),
            )
        self.conn.commit()

    def prospective_pending(self, k: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM prospective WHERE done=0 ORDER BY salience DESC, created_ts DESC LIMIT ?",
            (k,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── helpers ─────────────────────────────────────────────────────────────
    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        # Decode affect_json / tags for callers
        if d.get("affect_json"):
            try:
                d["affect"] = json.loads(d["affect_json"])
            except Exception:
                d["affect"] = None
        if d.get("tags"):
            d["tags_list"] = [t for t in (d["tags"] or "").split(",") if t]
        return d

    def close(self) -> None:
        self.conn.close()
