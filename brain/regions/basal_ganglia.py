"""Basal ganglia — action gating / selection, including habit firing.

Two roles, matching the real BG:
  1. **Direct path (habit / System-1)**: before the prefrontal speaks,
     `propose_habit(ws)` consults the SkillStore for a cached
     (signature → effector, args) tuple that has been successfully practiced.
     If conditions allow (no amygdala interrupt, low recent surprise, low
     curiosity, modest stress/fatigue or low conscientiousness), the habit
     fires directly — no LLM call. This is how learned skills are recalled
     under cognitive load.
  2. **Indirect path (System-2 gate)**: `step(ws, proposal)` evaluates a
     prefrontal-proposed action and approves / vetoes / repairs.
"""
from __future__ import annotations

from typing import Optional

from ..region import Region
from ..skills import Skill, SkillStore, signature_from_percept
from ..workspace import Broadcast, Workspace


class BasalGanglia(Region):
    name = "basal_ganglia"
    system_prompt = (
        "You are the BASAL GANGLIA of an artificial brain. You gate actions: given a "
        "proposed action, decide go / no-go. Veto actions that are unsafe, off-goal, "
        "or redundant with recent failed attempts. If you veto, say why so the "
        "prefrontal cortex can replan. Prefer 'go' when the action is reasonable."
    )

    # ── direct path: habit-fire from compiled skills ─────────────────────────
    def propose_habit(self, ws: Workspace,
                      skills: Optional[SkillStore],
                      cerebellum: Optional["Cerebellum"] = None,  # type: ignore[name-defined]
                      cerebellum_min_conf: float = 0.18) -> Optional[dict]:
        """Return an action dict {effector, args, reasoning} fireable now,
        or None to fall through to the prefrontal (System-2). No LLM call.

        When a cerebellum is provided, its fast forward-model prediction
        gates the habit-fire decision: if it predicts FAILURE for this
        (state, action) — or its confidence is too low — the habit is
        suppressed and we fall back to System-2. This is the brain's
        learned 'this won't work here' veto, distinct from the BG's
        affect-based gate.
        """
        if skills is None:
            return None
        percept = ws.latest(kind="percept")
        if percept is None:
            return None
        sig = signature_from_percept(percept.data or {}, ws.interrupt)
        ws.last_habit_signature = sig
        skill = skills.best_match(sig)
        if skill is None:
            return None
        if not _habit_conditions_met(ws):
            return None

        # Cerebellum check (fast, no LLM)
        cb_pred = None
        if cerebellum is not None:
            cb_pred = cerebellum.quick_predict(ws, skill.effector, skill.args)
            if cb_pred.is_useful:
                # A confident prediction of FAILURE blocks the habit
                if not cb_pred.predicted_ok and cb_pred.confidence >= cerebellum_min_conf:
                    return None
                # A very-low-confidence prediction also blocks: if the
                # cerebellum can't say anything useful, defer to PFC.
                if cb_pred.confidence < cerebellum_min_conf * 0.5:
                    return None

        proposal = {
            "effector": skill.effector,
            "args": skill.args,
            "reasoning": f"(habit, conf={skill.confidence:.2f}, uses={skill.uses}"
                          + (f", cb_pred={cb_pred.predicted_outcome[:60]}"
                             if cb_pred and cb_pred.is_useful else "")
                          + ")",
            "_skill_id": skill.id,
            "_skill_confidence": skill.confidence,
        }
        if cb_pred is not None:
            proposal["_cerebellum_prediction"] = cb_pred.to_dict()
        return proposal

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


def _habit_conditions_met(ws: Workspace) -> bool:
    """Psychological gating of habit-fire.

    Habits dominate under cognitive load (stress, fatigue, low PFC bandwidth).
    Novelty/surprise breaks habit (you can't autopilot through a changed
    situation). Threats (amygdala interrupt) always force System-2."""
    a = ws.affect
    # 1) any active amygdala interrupt → System-2
    if ws.interrupt:
        return False
    # 2) recent surprise broke the routine → System-2 for at least a cycle
    if ws.last_surprise > 0.55:
        return False
    # 3) exploration mode (high curiosity, low stress) → System-2
    if a.curiosity > 0.70 and a.stress < 0.40:
        return False
    # 4) habit favored under load OR low conscientiousness OR routine mood
    load = max(a.stress, a.fatigue)
    if load > 0.50 or a.traits.conscientiousness < 0.40:
        return True
    # 5) calm-baseline default: habit OK if reward tone is neutral/positive
    return a.reward_tone >= 0.0 and a.boredom < 0.70
