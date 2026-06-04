"""SkillStore — compiled procedural skills (the System-1 substrate).

Whenever the brain successfully completes an action, the (percept_signature,
effector, args) tuple is consolidated here. After enough successful repetitions
the same signature can fire the cached skill *directly* from the basal ganglia,
bypassing the prefrontal LLM call entirely. This is the brain's analogue of
striatum-based habit / procedural-skill learning: System-2 reasoning gets
compiled into a System-1 reflex through practice.

Storage shares the same SQLite db as `Memory` so skills persist across runs.

Signature design: a coarse but stable hash from the current sensory percept
(goal + entities) plus whether an amygdala interrupt is active. Same task →
same signature; small wording changes still match because we tokenize +
lowercase + sort + cap.

The basal ganglia consults this on every cycle BEFORE the prefrontal. Fire
conditions (see `should_fire_habit` in regions/basal_ganglia.py):
  - confidence ≥ 0.55  AND uses ≥ 2  (need actual practice)
  - no amygdala interrupt
  - last cycle's prediction_error below a threshold (no recent surprise)
  - stress or fatigue elevated (cognitive-load regime) OR conscientiousness low
  - curiosity NOT dominant (exploration mode suppresses habit)

Reward/penalty:
  - on success → confidence ← min(0.97, 0.7·confidence + 0.3·1.0); successes++
  - on failure → confidence ← 0.85·confidence; successes unchanged
  - on stale (not used in T cycles) → small decay applied at read time
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_WORD = re.compile(r"[a-z0-9]+")


def signature_from_percept(percept_data: dict, interrupt: Optional[str]) -> str:
    """Coarse stable hash of a percept. Two near-identical tasks produce the
    same signature; this is intentionally lossy so habits generalize."""
    if not percept_data:
        return "<no-percept>"
    entities = sorted(
        {str(e).lower().strip() for e in (percept_data.get("entities") or [])}
    )[:6]
    goal = (percept_data.get("goal") or "").lower()
    goal_tokens = sorted(set(_WORD.findall(goal)))[:8]
    interrupt_flag = "INT" if interrupt else "N"
    return f"e={'|'.join(entities)};g={'|'.join(goal_tokens)};{interrupt_flag}"


@dataclass
class Skill:
    id: int
    signature: str
    effector: str
    args: dict
    confidence: float
    uses: int
    successes: int
    last_used: float
    last_outcome: str

    @property
    def success_rate(self) -> float:
        return self.successes / self.uses if self.uses else 0.0

    def is_fireable(self, min_uses: int = 2, min_conf: float = 0.55) -> bool:
        return self.uses >= min_uses and self.confidence >= min_conf


class SkillStore:
    """SQLite-backed compiled-skill cache. Schema is created lazily and
    additively so reusing an existing memory.sqlite3 db is safe."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS skills (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                signature   TEXT NOT NULL,
                effector    TEXT NOT NULL,
                args_json   TEXT NOT NULL,
                confidence  REAL NOT NULL DEFAULT 0.5,
                uses        INTEGER NOT NULL DEFAULT 0,
                successes   INTEGER NOT NULL DEFAULT 0,
                last_used   REAL NOT NULL DEFAULT 0,
                last_outcome TEXT NOT NULL DEFAULT '',
                UNIQUE(signature, effector, args_json)
            );
            CREATE INDEX IF NOT EXISTS idx_skills_sig ON skills(signature);
        """)
        self.conn.commit()

    # ── lookup ──────────────────────────────────────────────────────────────
    def best_match(self, signature: str,
                   min_uses: int = 2, min_conf: float = 0.55) -> Optional[Skill]:
        """Return the highest-confidence fireable skill for this signature,
        or None. Applies a small staleness decay at read time."""
        rows = self.conn.execute(
            "SELECT * FROM skills WHERE signature = ? ORDER BY confidence DESC, uses DESC LIMIT 8",
            (signature,),
        ).fetchall()
        now = time.time()
        best: Optional[Skill] = None
        for row in rows:
            # Staleness: lose ~0.5%/day since last_used
            age_days = max(0.0, (now - float(row["last_used"])) / 86400.0)
            stale_factor = max(0.0, 1.0 - 0.005 * age_days)
            adj_conf = float(row["confidence"]) * stale_factor
            s = Skill(
                id=int(row["id"]),
                signature=row["signature"],
                effector=row["effector"],
                args=json.loads(row["args_json"]),
                confidence=adj_conf,
                uses=int(row["uses"]),
                successes=int(row["successes"]),
                last_used=float(row["last_used"]),
                last_outcome=row["last_outcome"],
            )
            if s.is_fireable(min_uses=min_uses, min_conf=min_conf):
                if best is None or s.confidence > best.confidence:
                    best = s
        return best

    def list_for(self, signature: str) -> list[Skill]:
        rows = self.conn.execute(
            "SELECT * FROM skills WHERE signature = ?", (signature,)
        ).fetchall()
        return [self._row(r) for r in rows]

    # ── consolidation ───────────────────────────────────────────────────────
    def consolidate(self, signature: str, effector: str, args: dict,
                    ok: bool, outcome: str = "") -> Skill:
        """Record one execution of (sig, effector, args). Creates the skill
        if missing, otherwise updates confidence + counts with EMA so a single
        result never dominates history (matches the AffectState pattern)."""
        args_json = json.dumps(args, sort_keys=True, default=str)[:1000]
        row = self.conn.execute(
            "SELECT * FROM skills WHERE signature=? AND effector=? AND args_json=?",
            (signature, effector, args_json),
        ).fetchone()
        now = time.time()
        outcome_blob = outcome[:200]

        if row is None:
            # First time: tentative confidence so it can't fire yet.
            conf = 0.55 if ok else 0.30
            cur = self.conn.execute(
                "INSERT INTO skills (signature, effector, args_json, confidence, "
                "uses, successes, last_used, last_outcome) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (signature, effector, args_json, conf,
                 1, 1 if ok else 0, now, outcome_blob),
            )
            self.conn.commit()
            return Skill(
                id=int(cur.lastrowid), signature=signature, effector=effector,
                args=args, confidence=conf, uses=1,
                successes=1 if ok else 0, last_used=now,
                last_outcome=outcome_blob,
            )

        # Existing skill: EMA on confidence, increment counters.
        prev_conf = float(row["confidence"])
        target = 1.0 if ok else 0.0
        new_conf = 0.75 * prev_conf + 0.25 * target
        # Clip
        new_conf = max(0.05, min(0.97, new_conf))
        new_uses = int(row["uses"]) + 1
        new_succ = int(row["successes"]) + (1 if ok else 0)
        self.conn.execute(
            "UPDATE skills SET confidence=?, uses=?, successes=?, "
            "last_used=?, last_outcome=? WHERE id=?",
            (new_conf, new_uses, new_succ, now, outcome_blob, int(row["id"])),
        )
        self.conn.commit()
        return Skill(
            id=int(row["id"]), signature=signature, effector=effector,
            args=args, confidence=new_conf, uses=new_uses, successes=new_succ,
            last_used=now, last_outcome=outcome_blob,
        )

    def punish(self, skill_id: int, factor: float = 0.85) -> None:
        """Decay confidence when a habit-fire produced a poor outcome that
        wasn't already a normal action result (e.g. surprise/regret)."""
        row = self.conn.execute(
            "SELECT confidence FROM skills WHERE id=?", (skill_id,)
        ).fetchone()
        if row is None:
            return
        new_conf = max(0.05, float(row["confidence"]) * factor)
        self.conn.execute("UPDATE skills SET confidence=? WHERE id=?",
                          (new_conf, skill_id))
        self.conn.commit()

    # ── inspection ──────────────────────────────────────────────────────────
    def top(self, k: int = 10) -> list[Skill]:
        rows = self.conn.execute(
            "SELECT * FROM skills ORDER BY confidence DESC, uses DESC LIMIT ?",
            (k,),
        ).fetchall()
        return [self._row(r) for r in rows]

    def _row(self, row: sqlite3.Row) -> Skill:
        return Skill(
            id=int(row["id"]),
            signature=row["signature"],
            effector=row["effector"],
            args=json.loads(row["args_json"]),
            confidence=float(row["confidence"]),
            uses=int(row["uses"]),
            successes=int(row["successes"]),
            last_used=float(row["last_used"]),
            last_outcome=row["last_outcome"],
        )

    def close(self) -> None:
        self.conn.close()


# ── prediction-error helper ────────────────────────────────────────────────
def prediction_surprise(predicted: str, actual: str) -> float:
    """Trigram Jaccard distance, used by the orchestrator to compute surprise
    after an action whose expected_result was emitted by the prefrontal.
    Returns 0 (identical) .. 1 (no overlap)."""
    if not predicted:
        return 0.0  # no prediction → no error signal
    def tris(s: str) -> set:
        toks = _WORD.findall(s.lower())
        return set(zip(toks, toks[1:], toks[2:]))
    A, B = tris(predicted), tris(actual or "")
    if not A and not B:
        return 0.0
    union = len(A | B)
    if union == 0:
        return 0.0
    return 1.0 - len(A & B) / union
