"""Default Mode Network — mind-wandering, intrusive thoughts, daydreaming.

In humans the DMN is most active when external task demands are LOW: it spins
up memories, self-referential thoughts, hypothetical futures. This is a primary
source of off-task behavior — a humanized brain that never wanders is just an
LLM in a costume.

Activation rule: fires only when distractibility is high (boredom + fatigue +
low arousal), or randomly with small probability scaled by openness trait.

When it fires, it posts a tangent — a memory snippet, a worry, an irrelevant
hypothetical — with enough salience that under the right conditions it can win
the attention spotlight (see Workspace.tick_attention).

The DMN uses the reflex tier (cheap) — its content is supposed to feel
half-formed, not eloquent.
"""
from __future__ import annotations

import json
import random
from typing import Optional

from ..region import Region
from ..workspace import Broadcast, ThoughtUnit, Workspace


class DefaultMode(Region):
    name = "default_mode"
    system_prompt = (
        "You are the DEFAULT MODE NETWORK of a human-like brain — the daydreaming "
        "voice. When the task is dull or the mind is tired, you produce a brief, "
        "off-topic intrusive thought: a memory fragment, a worry about something "
        "unrelated, a hypothetical, a craving. Be human, short, slightly raw. "
        "DO NOT solve the task — the prefrontal cortex does that. Your job is to "
        "wander."
    )

    def __init__(self, cfg, llm, seed: Optional[int] = None):
        super().__init__(cfg, llm)
        self.rng = random.Random(seed)

    def _should_fire(self, ws: Workspace) -> bool:
        d = ws.affect.distractibility
        # Trait nudge: open people wander more
        d += 0.15 * (ws.affect.traits.openness - 0.5)
        # If the amygdala has an active interrupt, suppress DMN (focused threat)
        if ws.interrupt:
            d *= 0.3
        return self.rng.random() < d

    def step(self, ws: Workspace) -> Broadcast | None:
        if not self._should_fire(ws):
            return None

        # Bias the tangent by current affect — mood-congruent intrusive thought.
        a = ws.affect
        flavor = (
            "negative, ruminative" if a.valence < -0.2
            else "wistful or fond" if a.valence > 0.2
            else "neutral or random"
        )
        out = self._chat_json(
            ws.render_context(limit=4) + "\n\n"
            f"Produce ONE brief intrusive thought ({flavor}, mood={a.mood_label}). "
            "Up to ~18 words. Return JSON: "
            '{"thought": str, "kind": "memory|worry|craving|hypothetical|tangent", '
            '"affect_delta": {"valence": -0.1..0.1, "arousal": -0.1..0.1, '
            '"social_need": -0.1..0.1, "boredom": -0.2..0.05}}',
            temperature=0.95,
            max_tokens=180,
        )
        thought = (out.get("thought") or "").strip()
        if not thought:
            return None

        # Mind-wandering bleeds a little affect into the system
        affect_delta = {
            k: float(v) for k, v in (out.get("affect_delta") or {}).items()
            if isinstance(v, (int, float))
        }
        # Wandering itself relieves boredom slightly
        affect_delta.setdefault("boredom", -0.04)
        ws.affect.update(self.name, ws.cycle, affect_delta, smoothing=0.8)

        # Salience: enough to *sometimes* win the spotlight when the brain is bored
        sal = 0.35 + 0.4 * ws.affect.distractibility

        # CRITICAL: in the stream-of-thought model, DMN hijacks the chain by
        # appending its tangent AS the next thought-unit. The prefrontal's
        # next step will see it as the most recent thought and either pick up
        # where it left off (suppression) or let it pull the chain (drift).
        # This is what stream interruption *means*.
        ws.thought_chain.append(ThoughtUnit(
            step=len(ws.thought_chain) + 1,
            source=self.name,
            content=thought,
            kind="tangent",
            affect_snapshot=a.mood_label,
            interrupted=False,
        ))

        return ws.post(Broadcast(
            source=self.name, kind="tangent",
            content=f"[{out.get('kind', 'tangent')}] {thought}",
            salience=sal, data=out,
        ))
