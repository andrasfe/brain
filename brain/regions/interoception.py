"""Interoception (insula) — bottom-up bodily contribution to emotion.

Reads the latest `body` broadcast (posted by the world ticker) and converts
physiological state into AffectState updates. This is the "I feel a knot in my
stomach" channel: the body talks, mood listens.

Deterministic and cheap on purpose — no LLM call. Insular cortex is fast.
"""
from __future__ import annotations

from ..region import Region
from ..workspace import Broadcast, Workspace


class Interoception(Region):
    name = "interoception"
    system_prompt = "INSULA — bodily state to affect (deterministic, no LLM)."

    def step(self, ws: Workspace) -> Broadcast | None:
        # Pull the current body line from the world
        body = ws.latest(kind="body", source="world")
        a = ws.affect

        deltas: dict = {}
        # Hunger drains valence and concentration; raises arousal at extremes
        if a.hunger > 0.6:
            deltas["valence"] = -0.05 * (a.hunger - 0.6) / 0.4
            deltas["stress"] = +0.04 * (a.hunger - 0.6) / 0.4
        # Fatigue drops arousal and tilts mood negative
        if a.fatigue > 0.5:
            deltas["arousal"] = -0.06 * (a.fatigue - 0.5) / 0.5
            deltas["valence"] = deltas.get("valence", 0.0) - 0.03 * (a.fatigue - 0.5) / 0.5
        # Boredom rises with low arousal — handled by world drift; here we let
        # boredom tug valence down a little
        if a.boredom > 0.65:
            deltas["valence"] = deltas.get("valence", 0.0) - 0.02
            deltas["curiosity"] = +0.04

        # Apply (EMA inside)
        if deltas:
            ws.affect.update(self.name, ws.cycle, deltas, smoothing=0.75)

        note = (f"body: hunger={a.hunger:.2f} fatigue={a.fatigue:.2f} "
                f"boredom={a.boredom:.2f}")
        if body:
            note += f"; env={body.content[:50]}"
        return ws.post(Broadcast(
            source=self.name, kind="body_state",
            content=note,
            # Body items only steal the spotlight when they're insistent
            salience=min(0.55, 0.25 + 0.5 * max(a.hunger, a.fatigue, 0.0)),
            data={"deltas": deltas},
        ))
