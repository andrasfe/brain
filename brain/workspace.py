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

from .affect import AffectState


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


@dataclass
class ThoughtUnit:
    """One step of the autoregressive 'stream of thought'.

    A ThoughtUnit is the analogue of a token in next-token prediction: the
    prefrontal produces one at a time, conditioned on the full chain so far
    plus the current AffectState and workspace. Other regions tick *between*
    units and can perturb the chain — a DMN tangent inserts a unit from
    source='default_mode', an amygdala interrupt forces the next unit to
    address the threat, a body signal can pivot the chain mid-argument.
    """
    step: int                         # position in the chain (1-indexed)
    source: str                       # "prefrontal" | "default_mode" | "amygdala_intrusion" | "interoception_intrusion"
    content: str                      # the natural-language thought
    kind: str                         # reflect | recall | appraise | tentative_plan | action | finish | tangent
    args: dict[str, Any] = field(default_factory=dict)  # for action/finish
    affect_snapshot: str = ""         # mood at the moment of this thought
    interrupted: bool = False         # True if a prior region perturbed this step
    ts: float = field(default_factory=time.time)


class Workspace:
    def __init__(self, task: str, decay: float = 0.85,
                 affect: Optional[AffectState] = None):
        self.task: str = task
        self.decay: float = decay
        self.cycle: int = 0
        self.items: list[Broadcast] = []
        self.history: list[ActionRecord] = []
        self.done: bool = False
        self.final_output: Optional[str] = None
        self.interrupt: Optional[str] = None  # set by amygdala for urgent salience
        # Persistent affective state — read by all regions, mutated by
        # affect-producing regions (amygdala/interoception/VTA/LC).
        self.affect: AffectState = affect if affect is not None else AffectState()
        # Autoregressive stream of thought — each unit is the analogue of a
        # token in next-token prediction. The prefrontal extends this chain
        # one unit at a time; other regions can also append (DMN tangents,
        # amygdala intrusions) — that IS the interruption mechanism.
        self.thought_chain: list[ThoughtUnit] = []
        # Predictive coding: the prefrontal may emit an `expected_result`
        # for each action. After the action runs, the orchestrator computes
        # surprise (trigram distance vs actual) and stores it here. The
        # basal ganglia consults it to break habit-fire on the next cycle.
        self.last_prediction: Optional[str] = None
        self.last_surprise: float = 0.0
        # The signature used for habit lookup last cycle (for debugging /
        # tracing only).
        self.last_habit_signature: Optional[str] = None
        self.habit_fired: bool = False  # True if this cycle's action came from cache

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

        Affect modulates this: high arousal narrows attention (the spotlight is
        sharper — only the very top item gets in), low arousal widens it (a
        random-ish second-tier item can win when the brain is wandering).

        Returns the newly broadcast item (the cycle's "conscious content"), if any.
        """
        # Arousal-modulated decay: high arousal => sharper decay of stale items
        decay = self.decay * (0.85 + 0.25 * (1 - self.affect.arousal))
        for b in self.items:
            if b.broadcast:
                b.salience *= decay

        candidates = [b for b in self.items if not b.broadcast]
        if not candidates:
            return None
        candidates.sort(key=lambda b: b.salience, reverse=True)
        # Wide attention (low arousal) lets a less-salient runner-up sometimes
        # win — deterministic but state-dependent: when distractibility is high
        # AND there's a DMN/world item near the top, allow it to grab the spotlight.
        if (self.affect.distractibility > 0.55
                and len(candidates) > 1
                and candidates[1].source in {"default_mode", "world"}
                and candidates[1].salience > 0.5 * candidates[0].salience):
            winner = candidates[1]
        else:
            winner = candidates[0]
        winner.broadcast = True
        return winner

    # ── rendering for prompts ────────────────────────────────────────────────
    def render_context(self, limit: int = 12) -> str:
        """A compact view of the conscious workspace for region prompts.

        Includes the persistent AffectState so every region's decisions are
        colored by mood — this is the main mechanism that makes the brain's
        outputs diverge from a raw stateless LLM call.
        """
        lines = [f"TASK: {self.task}", f"CYCLE: {self.cycle}", self.affect.render()]
        if self.interrupt:
            lines.append(f"!! INTERRUPT (amygdala): {self.interrupt}")
        # Arousal narrows attention: under high arousal show fewer items
        eff_limit = max(3, int(limit * self.affect.attention_width))
        spot = self.broadcasts()[:eff_limit]
        # Render the recent thought chain — same logic as autoregressive
        # token-conditioning: the next thought sees the prior chain.
        if self.thought_chain:
            lines.append("\nTHOUGHT CHAIN (most recent last — your inner monologue):")
            for t in self.thought_chain[-10:]:
                tag = t.source if t.source != "prefrontal" else "you"
                interrupt_mark = "⟪!⟫ " if t.interrupted else ""
                lines.append(f"  {interrupt_mark}[{t.step:02d} {tag}/{t.kind}] {t.content}")
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
