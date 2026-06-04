"""Basal ganglia — action gating / selection.

The prefrontal cortex *proposes*; the basal ganglia *disposes*. It approves,
modifies, or vetoes the proposed action — the brain's go/no-go gate. A veto on a
risky action (especially under an amygdala interrupt) forces a rethink.
"""
from __future__ import annotations

from ..region import Region
from ..workspace import Broadcast, Workspace


class BasalGanglia(Region):
    name = "basal_ganglia"
    system_prompt = (
        "You are the BASAL GANGLIA of an artificial brain. You gate actions: given a "
        "proposed action, decide go / no-go. Veto actions that are unsafe, off-goal, "
        "or redundant with recent failed attempts. If you veto, say why so the "
        "prefrontal cortex can replan. Prefer 'go' when the action is reasonable."
    )

    def step(self, ws: Workspace, proposal: dict) -> Broadcast:
        a = ws.affect
        # Disinhibition: high arousal + positive reward_tone + low
        # conscientiousness loosens vetoes (impulsivity). The reverse tightens
        # them. We tell the gate this explicitly so it acts in character.
        gate_mood = []
        if a.reward_tone > 0.25 and a.arousal > 0.55:
            gate_mood.append("dopamine high: you are inclined to APPROVE; give the "
                             "benefit of the doubt unless plainly unsafe.")
        if a.stress > 0.65:
            gate_mood.append("under stress: you may veto more on safety, but also "
                             "more easily APPROVE shortcuts that get to 'finish'.")
        if a.traits.conscientiousness < 0.35:
            gate_mood.append("low conscientiousness: rarely veto unless harmful.")
        if a.traits.conscientiousness > 0.65:
            gate_mood.append("high conscientiousness: prefer caution; veto sloppy or "
                             "redundant actions.")
        if a.traits.agreeableness > 0.65:
            gate_mood.append("agreeable: veto actions that might harm or annoy "
                             "people.")
        if a.fatigue > 0.7:
            gate_mood.append("tired: less patience to repair args — more likely to "
                             "go or veto on intuition.")

        out = self._chat_json(
            ws.render_context() + "\n\n"
            f"Proposed action: {proposal}\n"
            + (("Gate mood: " + " ".join(gate_mood) + "\n") if gate_mood else "")
            + "\nGate it. Return JSON: "
            '{"decision": "go|no_go", "reason": "brief, in first person", '
            '"effector": str, "args": {...}}  '
            "(echo/repair effector+args when decision is go).",
            temperature=0.2 + 0.25 * a.arousal,
        )
        go = out.get("decision") == "go"
        return ws.post(Broadcast(
            source=self.name, kind="gate",
            content=f"{out.get('decision', 'no_go')}: {out.get('reason', '')[:140]}",
            salience=0.6, data=out,
        ))
