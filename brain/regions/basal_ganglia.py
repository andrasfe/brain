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
        out = self._chat_json(
            ws.render_context() + "\n\n"
            f"Proposed action: {proposal}\n\n"
            "Gate it. Return JSON: "
            '{"decision": "go|no_go", "reason": "brief", '
            '"effector": str, "args": {...}}  '
            "(echo/repair effector+args when decision is go).",
            temperature=0.2,
        )
        go = out.get("decision") == "go"
        return ws.post(Broadcast(
            source=self.name, kind="gate",
            content=f"{out.get('decision', 'no_go')}: {out.get('reason', '')[:140]}",
            salience=0.6, data=out,
        ))
