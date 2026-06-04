"""Prefrontal cortex — executive control: planning and next-action proposal.

Reads the conscious workspace (percept, memory, salience, recent results) and
proposes one concrete next action drawn from the available effectors. Also
decides when the task is complete.
"""
from __future__ import annotations

from ..region import Region
from ..workspace import Broadcast, Workspace


class Prefrontal(Region):
    name = "prefrontal"
    system_prompt = (
        "You are the PREFRONTAL CORTEX of an artificial brain — the executive. "
        "You plan and choose the single best next action to advance the goal, using "
        "the available effectors. Think step by step but commit to ONE action. "
        "Heed amygdala interrupts. When the success criterion is met, choose 'finish'."
    )

    def step(self, ws: Workspace, effectors: list[str]) -> Broadcast:
        out = self._chat_json(
            ws.render_context() + "\n\n"
            f"Available effectors: {effectors}\n"
            "Effector arg schemas:\n"
            "  read_file{path}; write_file{path,content}; list_dir{path};\n"
            "  shell{command}; web_fetch{url}; think{note}; finish{answer}\n\n"
            "Propose the SINGLE next action. Return JSON: "
            '{"reasoning": "brief", "effector": str, "args": {...}, '
            '"confidence": 0.0-1.0}',
            temperature=0.5,
            max_tokens=1200,
        )
        eff = out.get("effector", "think")
        conf = float(out.get("confidence", 0.6) or 0.6)
        return ws.post(Broadcast(
            source=self.name, kind="plan",
            content=f"propose {eff}: {out.get('reasoning', '')[:160]}",
            salience=0.7 + 0.2 * conf, data=out,
        ))
