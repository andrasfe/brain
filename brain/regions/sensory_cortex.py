"""Sensory cortex — parses the raw task into a structured percept.

Runs once at the start: turns the user's goal into entities, constraints, and an
initial framing the rest of the brain can reason over.
"""
from __future__ import annotations

from ..region import Region
from ..workspace import Broadcast, Workspace


class SensoryCortex(Region):
    name = "sensory_cortex"
    system_prompt = (
        "You are the SENSORY CORTEX of an artificial brain. Your job is perception: "
        "take a raw task and parse it into a clear structured representation. "
        "Identify the goal, key entities, explicit constraints, and the success "
        "criterion. Do not plan or act — only perceive and structure."
    )

    def step(self, ws: Workspace) -> Broadcast:
        out = self._chat_json(
            f"Raw task:\n{ws.task}\n\n"
            'Return JSON: {"goal": str, "entities": [str], "constraints": [str], '
            '"success_criterion": str}'
        )
        summary = (
            f"goal={out.get('goal', ws.task)}; "
            f"success={out.get('success_criterion', 'task completed')}"
        )
        if out.get("constraints"):
            summary += f"; constraints={out['constraints']}"
        return ws.post(Broadcast(
            source=self.name, kind="percept", content=summary,
            salience=0.9, data=out,
        ))
