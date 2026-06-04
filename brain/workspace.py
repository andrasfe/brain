"""The Global Workspace — a shared blackboard with an attention mechanism.

Every region reads the workspace and may post a Broadcast. Each cognitive cycle
the orchestrator decays existing salience and selects the most salient new
broadcast to mark as `broadcast` (globally visible / "conscious"). This is the
core of Global Workspace Theory: many specialists, one serial spotlight.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Broadcast:
    """A single contribution posted to the workspace by a region."""
    source: str                 # region name
    kind: str                   # e.g. "percept", "plan", "salience", "action", "result"
    content: str                # natural-language payload
    salience: float = 0.5       # 0..1 attention weight
    data: dict[str, Any] = field(default_factory=dict)  # structured payload
    cycle: int = 0
    ts: float = field(default_factory=time.time)
    broadcast: bool = False     # became globally visible this/earlier cycle


@dataclass
class ActionRecord:
    cycle: int
    effector: str
    args: dict[str, Any]
    result: str
    ok: bool


class Workspace:
    def __init__(self, task: str, decay: float = 0.85):
        self.task: str = task
        self.decay: float = decay
        self.cycle: int = 0
        self.items: list[Broadcast] = []
        self.history: list[ActionRecord] = []
        self.done: bool = False
        self.final_output: Optional[str] = None
        self.interrupt: Optional[str] = None  # set by amygdala for urgent salience

    # ── posting / reading ────────────────────────────────────────────────────
    def post(self, b: Broadcast) -> Broadcast:
        b.cycle = self.cycle
        self.items.append(b)
        return b

    def latest(self, kind: str | None = None, source: str | None = None) -> Optional[Broadcast]:
        for b in reversed(self.items):
            if (kind is None or b.kind == kind) and (source is None or b.source == source):
                return b
        return None

    def all_of(self, kind: str) -> list[Broadcast]:
        return [b for b in self.items if b.kind == kind]

    def broadcasts(self) -> list[Broadcast]:
        """Items currently in the conscious spotlight, most salient first."""
        return sorted(
            [b for b in self.items if b.broadcast],
            key=lambda b: b.salience,
            reverse=True,
        )

    # ── attention ────────────────────────────────────────────────────────────
    def tick_attention(self) -> Optional[Broadcast]:
        """Decay old salience, then promote the most salient un-broadcast item.

        Returns the newly broadcast item (the cycle's "conscious content"), if any.
        """
        for b in self.items:
            if b.broadcast:
                b.salience *= self.decay

        candidates = [b for b in self.items if not b.broadcast]
        if not candidates:
            return None
        winner = max(candidates, key=lambda b: b.salience)
        winner.broadcast = True
        return winner

    # ── rendering for prompts ────────────────────────────────────────────────
    def render_context(self, limit: int = 12) -> str:
        """A compact view of the conscious workspace for region prompts."""
        lines = [f"TASK: {self.task}", f"CYCLE: {self.cycle}"]
        if self.interrupt:
            lines.append(f"!! INTERRUPT (amygdala): {self.interrupt}")
        spot = self.broadcasts()[:limit]
        if spot:
            lines.append("\nCONSCIOUS WORKSPACE (most salient first):")
            for b in spot:
                lines.append(f"  [{b.source}/{b.kind} s={b.salience:.2f}] {b.content}")
        if self.history:
            lines.append("\nRECENT ACTIONS:")
            for a in self.history[-5:]:
                status = "ok" if a.ok else "ERR"
                lines.append(f"  ({status}) {a.effector}({_short(a.args)}) -> {_short_text(a.result)}")
        return "\n".join(lines)


def _short(d: dict[str, Any], n: int = 80) -> str:
    s = ", ".join(f"{k}={v!r}" for k, v in d.items())
    return s if len(s) <= n else s[: n - 1] + "…"


def _short_text(s: str, n: int = 160) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"
