"""Hippocampus — typed episodic + semantic retrieval, mood-congruent ranking,
and consolidation of new traces.

Each cycle it:
  1. Pulls relevant SEMANTIC facts via TF-IDF (the "I know that…" lane).
  2. Pulls relevant EPISODIC traces via keyword overlap, mood-congruent.
  3. Pulls high-salience PRIOR persona facts (identity / recent events).
  4. Checks PROSPECTIVE memory for trigger matches against the percept.

All four are deduped and posted as one `memory` broadcast so the workspace
spotlight competes on substance, not on which lane the row came from.

`consolidate()` tags every write with a `mem_type` (defaulting to episodic).
"""
from __future__ import annotations

from typing import Optional

from ..memory import AFFECT, EPISODIC, Memory, PROSPECTIVE, SEMANTIC, SOURCE
from ..region import Region
from ..workspace import Broadcast, Workspace


class Hippocampus(Region):
    name = "hippocampus"
    system_prompt = (
        "You are the HIPPOCAMPUS of a human-like brain. You bind events into "
        "episodic traces and surface relevant past experience plus distilled "
        "semantic knowledge."
    )

    def __init__(self, cfg, llm, memory: Memory):
        super().__init__(cfg, llm)
        self.memory = memory
        self.k = int(cfg.memory.get("retrieve_k", 5))

    def step(self, ws: Workspace) -> Broadcast | None:
        percept = ws.latest(kind="percept")
        query = (percept.content if percept else ws.task)
        af = ws.affect
        # Mood-congruent query expansion (cheap hack that biases the bag-of-words
        # ranker toward congruent priors).
        if af.valence < -0.25:
            query_kw = query + " worry stress regret tired failed lost"
        elif af.valence > 0.25:
            query_kw = query + " good happy proud success warm calm"
        else:
            query_kw = query

        # ── multi-lane retrieval ────────────────────────────────────────────
        semantic = self.memory.retrieve_semantic(
            query, k=max(2, self.k // 2),
            types=[SEMANTIC])
        episodic = self.memory.retrieve(
            query_kw, k=self.k, types=[EPISODIC, AFFECT, SOURCE])
        # Persona priors live as kind='prior:*' under mem_type=episodic with
        # the persona task. Keep the dedicated keyword pull so they always
        # surface even when not lexically matched.
        priors = self.memory.retrieve("prior identity recent",
                                       k=max(2, self.k // 2))

        # ── prospective triggers ────────────────────────────────────────────
        prospective_hits = self.memory.prospective_match(
            (percept.content if percept else ws.task))
        if prospective_hits:
            # Mark them as fired and post each as a HIGH-salience broadcast —
            # intentions returning to mind shouldn't compete on small margins.
            self.memory.prospective_mark_fired([h["id"] for h in prospective_hits])
            for h in prospective_hits:
                ws.post(Broadcast(
                    source="hippocampus", kind="prospective",
                    content=f"prospective fires ({h['trigger_kind']}): {h['content']}",
                    salience=min(0.95, 0.55 + 0.2 * h["salience"]),
                    data=h,
                ))

        # Merge and dedup by id
        merged: list[dict] = []
        seen: set[int] = set()
        for batch in (semantic, episodic, priors):
            for e in batch:
                eid = int(e["id"])
                if eid in seen:
                    continue
                seen.add(eid)
                merged.append(e)
                if len(merged) >= self.k + 2:
                    break
        if not merged and not prospective_hits:
            return None
        if not merged:
            return None

        recalled = "; ".join(
            f"[{e.get('mem_type', '?')}/{e['kind']}] {e['content'][:110]}"
            for e in merged
        )
        return ws.post(Broadcast(
            source=self.name, kind="memory",
            content=f"recalled {len(merged)} (sem={len(semantic)} "
                    f"epi={len(episodic)} prior={len(priors)}): {recalled}",
            salience=0.55, data={"episodes": merged},
        ))

    def consolidate(self, ws: Workspace, kind: str, content: str,
                    salience: float, mem_type: str = EPISODIC,
                    tags: Optional[list[str]] = None) -> None:
        """Write a new trace. Action/outcome events go in as 'episodic';
        the orchestrator's end-of-task consolidator promotes recurring
        patterns to 'semantic' via an LLM extraction pass."""
        affect_snapshot = {
            "valence": round(ws.affect.valence, 3),
            "arousal": round(ws.affect.arousal, 3),
            "mood": ws.affect.mood_label,
        }
        self.memory.store(
            task=ws.task, kind=kind, content=content, salience=salience,
            mem_type=mem_type, affect_at_encode=affect_snapshot,
            tags=tags,
        )
