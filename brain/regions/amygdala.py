"""Amygdala — salience and emotional valence; can raise an interrupt.

Each cycle it scans the conscious workspace and the latest action result, tags
urgency/valence, and may set an interrupt that forces the prefrontal cortex to
attend to a risk (e.g. a destructive command, an error, going off-task).
"""
from __future__ import annotations

from ..region import Region
from ..workspace import Broadcast, Workspace


class Amygdala(Region):
    name = "amygdala"
    system_prompt = (
        "You are the AMYGDALA of an artificial brain. You assess salience and "
        "emotional valence — especially threat and urgency. Flag risks: data loss, "
        "destructive or irreversible actions, repeated failures, or drifting off the "
        "goal. You do not plan; you tag importance and sound alarms."
    )

    def step(self, ws: Workspace) -> Broadcast:
        out = self._chat_json(
            ws.render_context() + "\n\n"
            'Assess the current state. Return JSON: '
            '{"valence": "positive|neutral|negative", "urgency": 0.0-1.0, '
            '"interrupt": null or "short reason to halt/redirect", '
            '"note": "one-line salience assessment"}',
            temperature=0.4,
        )
        urgency = float(out.get("urgency", 0.3) or 0.3)
        interrupt = out.get("interrupt")
        if interrupt:
            ws.interrupt = str(interrupt)
        else:
            ws.interrupt = None
        return ws.post(Broadcast(
            source=self.name, kind="salience",
            content=out.get("note", f"valence={out.get('valence')} urgency={urgency:.2f}"),
            salience=min(1.0, 0.4 + urgency),  # urgent things grab the spotlight
            data=out,
        ))
