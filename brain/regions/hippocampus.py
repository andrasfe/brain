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
        episodes = self.memory.retrieve(query, k=self.k)
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
