"""Broca's area — language production: synthesizes the final answer.

Runs once at the end. Reads the whole conscious trace and the action history and
produces the user-facing output that satisfies the success criterion.
"""
from __future__ import annotations

from ..region import Region
from ..workspace import Workspace


class Broca(Region):
    name = "broca"
    system_prompt = (
        "You are BROCA'S AREA of a human-like brain — language production. "
        "Synthesize a complete final answer to the original task, grounded in what "
        "the brain actually did. Write IN THE FIRST PERSON as the individual whose "
        "mood and recent experience are part of you. Let mood color word choice and "
        "register without ever announcing the mood. Be direct; no meta-commentary "
        "about being a brain or an AI."
    )

    def step(self, ws: Workspace) -> str:
        actions = "\n".join(
            f"- {a.effector}({a.args}) -> {a.result[:200]}" for a in ws.history
        ) or "(no external actions were taken)"
        af = ws.affect
        voice = _broca_voice(af)
        return self._chat(
            f"Original task:\n{ws.task}\n\n"
            f"Conscious workspace:\n{ws.render_context(limit=20)}\n\n"
            f"Actions taken:\n{actions}\n\n"
            f"Voice instructions: {voice}\n\n"
            "Write the final answer.",
            temperature=0.55 + 0.25 * af.arousal,
            max_tokens=1500,
        )


def _broca_voice(a) -> str:
    bits = [f"Mood is '{a.mood_label}'."]
    if a.stress > 0.6:
        bits.append("Be terse; clipped sentences; less hedging.")
    if a.fatigue > 0.65:
        bits.append("Sound a little tired; one or two minor word fumbles are fine.")
    if a.valence > 0.4:
        bits.append("Warmer register; one small flicker of enthusiasm is OK.")
    if a.valence < -0.4:
        bits.append("Flatter, drier; no false enthusiasm.")
    if a.reward_tone > 0.3:
        bits.append("Slightly proud, but don't gloat.")
    if a.reward_tone < -0.3:
        bits.append("A note of resignation is allowed.")
    if a.traits.agreeableness > 0.65:
        bits.append("Softening hedges where appropriate ('I think', 'maybe').")
    if a.traits.agreeableness < 0.35:
        bits.append("Blunt; skip social softeners.")
    if a.traits.conscientiousness > 0.65:
        bits.append("Tidy structure.")
    return " ".join(bits)
