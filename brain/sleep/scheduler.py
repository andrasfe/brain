"""Scheduler — always-on agent that fires time-based prospective triggers.

The hippocampus checks `prospective_match` each cognitive cycle against the
current percept — that handles keyword and percept-substring triggers
naturally. But `kind='time'` triggers need an active clock check even when
no input has come in. That's the scheduler's job.

Runs in every daemon tick (wake or sleep) at very low cost (a single SQLite
query). When a time-triggered intention fires, it surfaces:
  - if the brain is asleep: it counts as a salient stimulus that pushes
    fatigue down and arousal up (a 'realization' rouses you)
  - if the brain is awake: it gets posted as a high-salience broadcast at
    the next cognitive cycle's start
"""
from __future__ import annotations

import time
from typing import Any, List

from ..memory import Memory


class Scheduler:
    name = "scheduler"

    def fire_due(self, memory: Memory, now: float | None = None) -> List[dict[str, Any]]:
        now = now if now is not None else time.time()
        rows = memory.conn.execute(
            "SELECT * FROM prospective WHERE done=0 AND trigger_kind='time' "
            "AND fires_after_ts IS NOT NULL AND fires_after_ts <= ? "
            "ORDER BY salience DESC",
            (now,),
        ).fetchall()
        if not rows:
            return []
        fired = []
        for r in rows:
            fired.append({
                "id": int(r["id"]),
                "content": r["content"],
                "trigger_kind": r["trigger_kind"],
                "salience": float(r["salience"]),
                "fires_after_ts": float(r["fires_after_ts"]),
            })
        # Time triggers are one-shot: mark done so they don't re-fire next tick.
        memory.prospective_mark_fired([f["id"] for f in fired], mark_done=True)
        return fired
