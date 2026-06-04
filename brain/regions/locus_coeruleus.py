"""Locus Coeruleus — noradrenergic arousal / attention width.

The LC tracks how much *unexpected information* is hitting the workspace this
cycle and nudges arousal up or down. It is the brain's gain knob. High arousal
sharpens but narrows attention; low arousal lets the DMN in. This region does
NOT call other regions — it only writes to AffectState.

Deterministic, no LLM call.
"""
from __future__ import annotations

from ..region import Region
from ..workspace import Broadcast, Workspace


class LocusCoeruleus(Region):
    name = "locus_coeruleus"
    system_prompt = "LC — arousal / attention gain (deterministic, no LLM)."

    def step(self, ws: Workspace) -> Broadcast | None:
        # Look at the most recent action outcome and any high-salience new items
        last_action = ws.history[-1] if ws.history else None
        novel_items = [b for b in ws.items if b.cycle == ws.cycle and b.salience > 0.55]

        arousal_delta = 0.0
        # Surprise / failure / interrupt all raise arousal
        if last_action and not last_action.ok:
            arousal_delta += 0.06
        if ws.interrupt:
            arousal_delta += 0.05
        # Many novel items → engaged → mild arousal up
        if len(novel_items) >= 3:
            arousal_delta += 0.02
        elif len(novel_items) == 0:
            arousal_delta -= 0.02  # nothing happening → drowsy drift

        # Cap and apply
        if arousal_delta != 0:
            ws.affect.update(self.name, ws.cycle,
                             {"arousal": arousal_delta}, smoothing=0.8)

        return ws.post(Broadcast(
            source=self.name, kind="modulator",
            content=f"arousal={ws.affect.arousal:.2f} (Δ={arousal_delta:+.3f}) "
                    f"width={ws.affect.attention_width:.2f}",
            salience=0.20,  # modulators don't grab the spotlight
            data={"arousal_delta": arousal_delta},
        ))
