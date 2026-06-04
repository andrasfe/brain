"""Hippocampus — episodic memory: retrieves relevant past episodes and stores new ones.

It wraps the SQLite Memory store. On each cycle it retrieves episodes relevant to
the current goal and posts them to the workspace; the orchestrator separately
asks it to store new action results.
"""
from __future__ import annotations

from ..memory import Memory
from ..region import Region
from ..workspace import Broadcast, Workspace


class Hippocampus(Region):
    name = "hippocampus"
    system_prompt = (
        "You are the HIPPOCAMPUS of an artificial brain, responsible for episodic "
        "memory. You summarize and surface relevant past experience."
    )

    def __init__(self, cfg, llm, memory: Memory):
        super().__init__(cfg, llm)
        self.memory = memory
        self.k = int(cfg.memory.get("retrieve_k", 5))

    def step(self, ws: Workspace) -> Broadcast | None:
        percept = ws.latest(kind="percept")
        query = (percept.content if percept else ws.task)
        # Mood-congruent recall: when mood is negative, weight the query toward
        # negatively-toned content; when positive, the opposite. Cheap hack:
        # append mood keywords to the retrieval query so the bag-of-words
        # ranker pulls congruent priors.
        af = ws.affect
        if af.valence < -0.25:
            query += " worry stress regret tired failed lost"
        elif af.valence > 0.25:
            query += " good happy proud success warm calm"
        # Always pull in identity / recent priors with a small extra weight
        # via a second retrieval keyed on 'prior'.
        episodes = self.memory.retrieve(query, k=self.k)
        priors = self.memory.retrieve("prior identity recent", k=max(2, self.k // 2))
        # de-dupe by id while keeping order: episodes first, then priors
        seen = {e["id"] for e in episodes}
        for p in priors:
            if p["id"] not in seen:
                episodes.append(p)
                seen.add(p["id"])
        if not episodes:
            return None
        recalled = "; ".join(f"[{e['kind']}] {e['content'][:120]}" for e in episodes)
        return ws.post(Broadcast(
            source=self.name, kind="memory",
            content=f"recalled {len(episodes)} episode(s): {recalled}",
            salience=0.55, data={"episodes": episodes},
        ))

    def consolidate(self, ws: Workspace, kind: str, content: str, salience: float) -> None:
        """Write a new episode to long-term store."""
        self.memory.store(ws.task, kind, content, salience)
