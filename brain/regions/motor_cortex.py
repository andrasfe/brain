"""Motor cortex — convert narrated action-intent into a concrete screen action.

Local models often *describe* doing a screen step ("I'm scrolling down now")
without ever emitting a `kind='action'`, so nothing happens. When the prefrontal
stalls in that rehearsal loop while embodied, this region is invoked: a focused,
executive-tier (qwen-35b on the Mac Studio) motor-planning pass that — given the
goal, the live screen, and the exact screen-effector schemas — returns the ONE
concrete effector call to take right now. It is forced to commit to a real
`screen_*` effector with real args (coordinates, scroll amount), or abstain.

This is deliberately the strong model, not the reflex one: grounding an action
in the actual screen (picking a click target, choosing a scroll amount) is the
hardest, most consequential step, and worth the compute. Config-gated
(`embodiment.motor_cortex`); off by default.
"""
from __future__ import annotations

import re
from typing import Optional

from ..region import Region
from ..workspace import Workspace
from ..world_model import render_state


# Verbs that signal the agent intends to manipulate the screen.
_MOTOR_INTENT = re.compile(
    r"\b(scroll(ing|ed)?|click(ing|ed)?|tap(ping|ped)?|type(ing|d)?|typing|"
    r"press(ing|ed)?|select(ing|ed)?|open(ing|ed)?|drag(ging|ged)?|swipe|"
    r"navigat(e|ing)|enter(ing|ed)?|hit (the )?(return|enter))\b",
    re.IGNORECASE,
)


def has_motor_intent(text: str) -> bool:
    """True when the text reads like the agent means to act on the screen."""
    return bool(text and _MOTOR_INTENT.search(text))


class MotorCortex(Region):
    name = "motor_cortex"   # executive tier (set in config regions:)
    system_prompt = (
        "You are the MOTOR CORTEX. You translate an intention to act on the "
        "computer SCREEN into one concrete, executable action. You never "
        "describe or plan — you commit to a single real effector call."
    )

    def plan_action(self, ws: Workspace, affordances: str, intent_text: str,
                    screen_text: str = "") -> Optional[dict]:
        """Return {effector, args, reasoning} for the screen action to take now,
        or None to abstain (no screen action fits)."""
        goal = ""
        percept = ws.latest(kind="percept")
        if percept and percept.data:
            goal = str(percept.data.get("goal", ""))
        if not screen_text:
            v = ws.latest(kind="vision")
            screen_text = (v.content if v else "") or render_state(ws)

        prompt = (
            f"Goal: {goal}\n"
            f"The agent just said it would: \"{intent_text[:200]}\"\n"
            "But narrating does nothing — you must emit the real action.\n\n"
            f"Current screen:\n{screen_text[:800]}\n\n"
            f"Available screen actions (use EXACT names + args):\n{affordances}\n\n"
            "Output JSON ONLY for the SINGLE concrete action to take right now:\n"
            '{"effector": "screen_scroll|screen_click|screen_type|screen_key|look",\n'
            ' "args": {<exact args — real coordinates 0..1, real scroll amount, etc.>},\n'
            ' "reason": "one short clause"}\n'
            "If no screen action fits the intent, output {\"effector\": \"none\"}.\n"
            "Do NOT use shell. Do NOT describe. Commit to one action."
        )
        out = self._chat_json(prompt, temperature=0.2, max_tokens=600)
        eff = str(out.get("effector") or "").strip()
        if not eff or eff.lower() in ("none", "noop", ""):
            return None
        args = out.get("args")
        if not isinstance(args, dict):
            args = {}
        return {"effector": eff, "args": args,
                "reasoning": f"(motor cortex) {out.get('reason', intent_text)[:80]}"}
