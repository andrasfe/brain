"""VTA — ventral tegmental area, the dopamine source.

VTA tracks reward prediction error (was the last action better/worse than
expected?) and feeds `reward_tone` into AffectState. High reward tone:
basal-ganglia gating loosens (impulsive go), prefrontal becomes a bit
risk-tolerant. Low reward tone (disappointment): motivation drops, more
'finish' / 'think' choices, less exploration.

Deterministic, no LLM call.
"""
from __future__ import annotations

from ..region import Region
from ..workspace import Broadcast, Workspace


class VTA(Region):
    name = "vta"
    system_prompt = "VTA — dopaminergic reward signal (deterministic, no LLM)."

    def step(self, ws: Workspace) -> Broadcast | None:
        last = ws.history[-1] if ws.history else None
        if last is None:
            return None

        # Reward prediction error: ok = mild positive, err = negative,
        # progress-toward-goal heuristic via spotlight kind would be nice but
        # we keep it simple here.
        if last.ok:
            rpe = +0.08
        else:
            rpe = -0.12

        # Repeated failures sting more
        recent_fails = sum(1 for a in ws.history[-3:] if not a.ok)
        if recent_fails >= 2:
            rpe -= 0.06

        ws.affect.update(self.name, ws.cycle, {
            "reward_tone": rpe,
            "valence": 0.5 * rpe,        # reward feels good
            "dominance": 0.4 * rpe,       # success feels in-control
        }, smoothing=0.7)

        return ws.post(Broadcast(
            source=self.name, kind="modulator",
            content=f"reward_tone={ws.affect.reward_tone:+.2f} (RPE={rpe:+.2f})",
            salience=0.22,
            data={"rpe": rpe},
        ))
