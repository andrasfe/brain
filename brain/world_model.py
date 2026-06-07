"""WorldModelStore — learned forward model of action outcomes.

The brain's first stab at a JEPA-style world model: a k-NN substrate over
`(state, action, outcome)` triples in the configured embedding space. Every
executed action contributes one triple. When the prefrontal is about to
emit an action, it consults this store for *learned* expected outcomes
from past similar (state, action) situations — and threads them into its
prompt as a top-down prediction.

Why this is the LeCun-aligned move (in our context):
  - Predictions live in a **learned latent space**: TF-IDF today, real
    embeddings (`sentence-transformers` / OpenRouter) when configured.
    Either way, the prediction improves with data without changing the
    underlying LLM. That's the JEPA distinction.
  - The prediction is a CHEAP lookup (k-NN), not a generation. The PFC
    still composes the final `expected_result` string but does so with
    grounded evidence — analogous to neocortical predictions being shaped
    by hippocampal/cerebellar forward models.
  - Surprise (existing `prediction_surprise` between expected vs actual)
    is the learning signal: a high-surprise triple gets higher salience
    when stored, so future queries weight it more.

Neuroscience parallel: this is the shared *substrate*. Different regions
consult it differently — PFC for slow goal-conditioned imagination,
the future Cerebellum region for fast motor reflex predictions, the
Dreamer for counterfactual recombination during REM.

The store opens its own SQLite connection but uses the SAME db file as
`Memory` so a single brain has one persistent home for all learning.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from .embeddings import EmbeddingBackend, TfidfBackend, _pack_floats, _unpack_floats

_WORD = re.compile(r"[a-z0-9]+")


# ── helpers ────────────────────────────────────────────────────────────────
def render_state(workspace) -> str:
    """A compact, embeddable description of 'what state was I in' at the
    moment of a decision. Includes the things that should *predict* the
    outcome: percept goal, the spotlight item, mood, and any active
    amygdala interrupt."""
    percept = workspace.latest(kind="percept")
    spotlight = workspace.broadcasts()[:1]
    bits: list[str] = []
    if percept and percept.data:
        bits.append(f"goal={percept.data.get('goal', '')[:80]}")
        ents = percept.data.get("entities") or []
        if ents:
            bits.append(f"entities={','.join(str(e)[:24] for e in ents[:4])}")
    if spotlight:
        s = spotlight[0]
        bits.append(f"spot={s.source}/{s.kind}:{s.content[:60]}")
    bits.append(f"mood={workspace.affect.mood_label}")
    if workspace.interrupt:
        bits.append(f"interrupt={workspace.interrupt[:50]}")
    return "; ".join(bits)


def render_action(effector: str, args: dict) -> str:
    """A short canonical form of the action — used both for embedding and
    for the action-equality match in `predict`."""
    if not args:
        return effector
    # Sort keys, truncate long values for a stable canonical form
    parts: list[str] = []
    for k in sorted(args.keys()):
        v = args[k]
        if isinstance(v, str):
            v = v[:60]
        elif isinstance(v, (list, dict)):
            v = json.dumps(v, sort_keys=True, default=str)[:60]
        parts.append(f"{k}={v}")
    return f"{effector}({', '.join(parts)})"


def _action_overlap(a: str, b: str) -> float:
    """Loose match between two action strings — same effector AND ≥ 1 arg
    token in common is enough for the predict() filter."""
    if not a or not b:
        return 0.0
    a_eff = a.split("(", 1)[0]
    b_eff = b.split("(", 1)[0]
    if a_eff != b_eff:
        return 0.0
    a_toks = set(_WORD.findall(a.lower()))
    b_toks = set(_WORD.findall(b.lower()))
    if not a_toks or not b_toks:
        return 0.5  # effectors match, no args — count as soft match
    overlap = len(a_toks & b_toks) / max(len(a_toks), len(b_toks))
    return overlap


class WorldModelStore:
    """k-NN over (state, action, outcome) in the configured embedding space."""

    def __init__(self, db_path: Path, backend: Optional[EmbeddingBackend] = None,
                 window: int = 600):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        # World model rides the configured backend — when 'auto' resolves to
        # real embeddings, predictions live in a learned latent space.
        self.backend: EmbeddingBackend = backend or TfidfBackend()
        self.window = window
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS world_model (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ts           REAL NOT NULL,
                state_text   TEXT NOT NULL,
                action_text  TEXT NOT NULL,
                outcome_text TEXT NOT NULL,
                ok           INTEGER NOT NULL DEFAULT 1,
                source       TEXT NOT NULL DEFAULT 'observed',
                salience     REAL NOT NULL DEFAULT 0.5,
                state_emb    BLOB,
                state_vis_emb   BLOB,
                outcome_vis_emb BLOB
            );
            CREATE INDEX IF NOT EXISTS idx_wm_ts ON world_model(ts);
            CREATE INDEX IF NOT EXISTS idx_wm_source ON world_model(source);
        """)
        # Auto-migrate older single-modality DBs: add the visual columns if
        # the table predates them (DINOv2 state/outcome embeddings for the
        # action-conditioned visual world model).
        existing = {r["name"] for r in
                    self.conn.execute("PRAGMA table_info(world_model)").fetchall()}
        for col in ("state_vis_emb", "outcome_vis_emb"):
            if col not in existing:
                self.conn.execute(f"ALTER TABLE world_model ADD COLUMN {col} BLOB")
        self.conn.commit()

    # ── observe ─────────────────────────────────────────────────────────────
    def observe(self, state_text: str, action_text: str, outcome_text: str,
                ok: bool = True, source: str = "observed",
                salience: float = 0.5,
                state_vis: Optional[list] = None,
                outcome_vis: Optional[list] = None) -> int:
        """Record one (state, action, outcome) triple.

        Salience is bumped automatically for high-surprise observations; the
        caller can also pass it explicitly (e.g. dreams arrive with low
        salience, real observations with mid).

        `state_vis` / `outcome_vis` are optional DINOv2 screen embeddings — the
        substrate of the action-conditioned VISUAL world model. When present,
        the same row carries both the text triple and the visual transition, so
        the visual forward model trains on `f([state_vis;action]) -> outcome_vis`."""
        emb_blob: Optional[bytes] = None
        if self.backend.persistent:
            try:
                emb_blob = self.backend.encode_one(state_text)
            except Exception:
                emb_blob = None
        sv = _pack_floats(state_vis) if state_vis else None
        ov = _pack_floats(outcome_vis) if outcome_vis else None
        cur = self.conn.execute(
            "INSERT INTO world_model (ts, state_text, action_text, outcome_text, "
            "ok, source, salience, state_emb, state_vis_emb, outcome_vis_emb) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (time.time(), state_text[:600], action_text[:280],
             outcome_text[:600], int(bool(ok)), source,
             float(salience), emb_blob, sv, ov),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def visual_triples(self, limit: int = 4000) -> list[dict[str, Any]]:
        """Return recent rows that carry BOTH a state and outcome visual
        embedding — the training set for the visual forward model (V2).
        Embeddings are unpacked to float lists."""
        rows = self.conn.execute(
            "SELECT id, action_text, ok, state_vis_emb, outcome_vis_emb "
            "FROM world_model WHERE state_vis_emb IS NOT NULL "
            "AND outcome_vis_emb IS NOT NULL ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append({
                "id": int(r["id"]),
                "action_text": r["action_text"],
                "ok": int(r["ok"]),
                "state_vis": _unpack_floats(r["state_vis_emb"]),
                "outcome_vis": _unpack_floats(r["outcome_vis_emb"]),
            })
        return out

    def count_visual(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM world_model "
            "WHERE state_vis_emb IS NOT NULL AND outcome_vis_emb IS NOT NULL"
        ).fetchone()
        return int(row["n"])

    # ── predict ─────────────────────────────────────────────────────────────
    def predict(self, state_text: str, action_text: str, k: int = 3,
                min_score: float = 0.10,
                action_overlap_min: float = 0.5) -> list[dict[str, Any]]:
        """Return up to k past outcomes for the most similar (state, action)
        triples. Filter: action must overlap (same effector + ≥1 shared
        token in args) so we don't return outcomes from a different verb.
        Scores are cosine similarity over `state_text`."""
        rows = self._candidates()
        if not rows:
            return []
        # Pre-filter by action overlap to a smaller eligibility set
        eligible: list[sqlite3.Row] = [
            r for r in rows
            if _action_overlap(r["action_text"], action_text) >= action_overlap_min
        ]
        if not eligible:
            return []
        self._fit_backend(eligible)
        eligible_ids = {int(r["id"]) for r in eligible}
        ranked = self.backend.topk(state_text, k=k * 2, eligible=eligible_ids)
        if not ranked:
            return []
        rows_by_id = {int(r["id"]): r for r in eligible}
        out: list[dict[str, Any]] = []
        for did, score in ranked:
            if score < min_score:
                continue
            row = rows_by_id.get(did)
            if row is None:
                continue
            d = dict(row)
            d["score"] = round(float(score), 4)
            out.append(d)
            if len(out) >= k:
                break
        return out

    def counterfactuals(self, state_text: str,
                         current_action: str,
                         k: int = 3,
                         min_score: float = 0.10) -> list[dict[str, Any]]:
        """Return up to k past triples with similar STATE but DIFFERENT
        action. Used by the dreamer during REM for 'what if'
        recombination."""
        rows = self._candidates()
        if not rows:
            return []
        # Different-action: action_overlap below threshold
        eligible = [
            r for r in rows
            if _action_overlap(r["action_text"], current_action) < 0.3
        ]
        if not eligible:
            return []
        self._fit_backend(eligible)
        eligible_ids = {int(r["id"]) for r in eligible}
        ranked = self.backend.topk(state_text, k=k * 2, eligible=eligible_ids)
        if not ranked:
            return []
        rows_by_id = {int(r["id"]): r for r in eligible}
        out: list[dict[str, Any]] = []
        for did, score in ranked:
            if score < min_score:
                continue
            row = rows_by_id.get(did)
            if row is None:
                continue
            d = dict(row)
            d["score"] = round(float(score), 4)
            out.append(d)
            if len(out) >= k:
                break
        return out

    # ── helpers ─────────────────────────────────────────────────────────────
    def _candidates(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM world_model ORDER BY ts DESC LIMIT ?",
            (self.window,),
        ).fetchall()

    def _fit_backend(self, rows) -> None:
        """Populate the backend's hot index from `rows`. Loads any cached
        BLOBs first so persistent backends skip API calls for known rows."""
        bk = self.backend
        if bk.persistent and getattr(bk, "remember", None):
            for r in rows:
                blob = r["state_emb"]
                if not blob:
                    continue
                vec = bk.from_bytes(blob)
                if vec:
                    bk.remember(int(r["id"]), vec)
        if isinstance(bk, TfidfBackend):
            # TF-IDF is corpus-relative; rebuild over the current window
            bk.mark_stale()
        bk.fit((int(r["id"]), r["state_text"]) for r in rows)

        # Write back newly-computed embeddings for persistent backends
        if bk.persistent:
            new_vecs = getattr(bk, "_vecs", None) or {}
            updates: list[tuple[bytes, int]] = []
            for r in rows:
                rid = int(r["id"])
                if rid not in new_vecs:
                    continue
                if r["state_emb"]:
                    continue
                updates.append((_pack_floats(new_vecs[rid]), rid))
            if updates:
                self.conn.executemany(
                    "UPDATE world_model SET state_emb=? WHERE id=?", updates)
                self.conn.commit()

    def count(self, source: Optional[str] = None) -> int:
        if source:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM world_model WHERE source=?",
                (source,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM world_model"
            ).fetchone()
        return int(row["n"])

    def close(self) -> None:
        self.conn.close()
