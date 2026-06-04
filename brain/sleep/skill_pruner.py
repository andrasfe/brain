"""SkillPruner — NREM agent that decays unused skills and drops the lost causes.

Skills (procedural habits in the SkillStore) decay if not used. Sleep is the
time to do this aggressively — when waking, the brain shouldn't have a
forest of low-confidence routines polluting the BG's habit-fire decisions.
"""
from __future__ import annotations

import time
from typing import Any

from ..skills import SkillStore


class SkillPruner:
    name = "skill_pruner"

    def __init__(self,
                 decay_factor: float = 0.92,
                 unused_age_seconds: float = 7 * 24 * 3600.0,
                 delete_below_conf: float = 0.20):
        self.decay_factor = decay_factor
        self.unused_age_seconds = unused_age_seconds
        self.delete_below_conf = delete_below_conf

    def run(self, skills: SkillStore) -> dict[str, Any]:
        now = time.time()
        cutoff = now - self.unused_age_seconds
        # Decay all skills that haven't been used since `cutoff`
        rows = skills.conn.execute(
            "SELECT id, confidence FROM skills WHERE last_used < ? OR last_used IS NULL",
            (cutoff,),
        ).fetchall()
        decayed = 0
        for r in rows:
            new_conf = max(0.0, float(r["confidence"]) * self.decay_factor)
            skills.conn.execute("UPDATE skills SET confidence=? WHERE id=?",
                                (new_conf, int(r["id"])))
            decayed += 1
        skills.conn.commit()

        # Delete anything below the floor — these will never fire again
        # and just take up space + slow up lookups.
        deleted_cur = skills.conn.execute(
            "DELETE FROM skills WHERE confidence < ?", (self.delete_below_conf,))
        deleted = deleted_cur.rowcount or 0
        skills.conn.commit()
        return {"decayed": decayed, "deleted": int(deleted)}
