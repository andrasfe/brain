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
        "You are BROCA'S AREA of an artificial brain — language production. "
        "Synthesize a clear, complete final answer to the original task, grounded in "
        "what the brain actually did. Be direct; no meta-commentary about being a brain."
    )

    def step(self, ws: Workspace) -> str:
        actions = "\n".join(
            f"- {a.effector}({a.args}) -> {a.result[:200]}" for a in ws.history
        ) or "(no external actions were taken)"
        return self._chat(
            f"Original task:\n{ws.task}\n\n"
            f"Conscious workspace:\n{ws.render_context(limit=20)}\n\n"
            f"Actions taken:\n{actions}\n\n"
            "Write the final answer for the user.",
            temperature=0.6,
            max_tokens=1500,
        )
