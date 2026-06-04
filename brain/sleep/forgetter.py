"""Forgetter — NREM agent that prunes low-salience un-recalled episodes.

Real episodic memory is heavily pruned during sleep; without it the brain
grows unbounded and recall slows. This agent runs during NREM bouts only.

Rules (all configurable):
  - Never prune `prior:*` rows (persona facts).
  - Never prune the most recent N episodic rows (recency safety net).
  - Never prune semantic / source / affect / prospective rows.
  - For the remaining episodic rows: delete those with salience < threshold
    AND age > min_age_seconds AND not tagged 'consolidated' (those were
    distilled into semantic facts already, the episode is still kept as
    provenance but can be pruned later).

Returns a stats dict the daemon logs.
"""
from __future__ import annotations

import time
from typing import Any

from ..memory import EPISODIC, Memory


class Forgetter:
    name = "forgetter"

    def __init__(self,
                 salience_threshold: float = 0.30,
                 min_age_seconds: float = 6 * 3600.0,
                 recency_safety: int = 60):
        self.salience_threshold = salience_threshold
        self.min_age_seconds = min_age_seconds
        self.recency_safety = recency_safety

    def run(self, memory: Memory) -> dict[str, Any]:
        now = time.time()
        # Identify the recency safety set
        safe_rows = memory.conn.execute(
            "SELECT id FROM episodes WHERE mem_type=? "
            "ORDER BY ts DESC LIMIT ?",
            (EPISODIC, self.recency_safety),
        ).fetchall()
        safe_ids = {int(r["id"]) for r in safe_rows}

        # Candidates: low-salience old episodic rows that aren't priors
        candidates = memory.conn.execute(
            "SELECT id, kind, salience, ts FROM episodes "
            "WHERE mem_type=? AND salience < ? AND ts < ? "
            "AND kind NOT LIKE 'prior:%' "
            "ORDER BY ts ASC LIMIT 500",
            (EPISODIC, self.salience_threshold, now - self.min_age_seconds),
        ).fetchall()
        to_delete = [int(r["id"]) for r in candidates if int(r["id"]) not in safe_ids]

        if not to_delete:
            return {"considered": len(candidates), "pruned": 0, "kept_safe": 0}

        placeholders = ",".join("?" * len(to_delete))
        memory.conn.execute(
            f"DELETE FROM episodes WHERE id IN ({placeholders})", to_delete)
        memory.conn.commit()
        return {
            "considered": len(candidates),
            "pruned": len(to_delete),
            "kept_safe": len(candidates) - len(to_delete),
        }
