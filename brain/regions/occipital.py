"""Occipital (visual cortex) — turns what the body's eyes see into a percept.

When the brain is embodied (an afferent `Embodiment` is attached), this region
observes the screen and posts a `vision` broadcast to the workspace so the
prefrontal stream-of-thought can reason over it.

Two perception paths:
  1. If the backend already returns structured elements / OCR text (e.g. the
     FakeBackend in tests, or a future grounding backend), `Observation.
     render_text()` is rich on its own — we post that directly, no LLM call.
  2. If the backend returns only a screenshot frame (e.g. the macOS backend:
     screencapture gives pixels but no element detection), we optionally hand
     the frame to a vision-capable model (`llm.describe_image`) to produce a
     short textual description. This is the expensive path — rate-limited by
     the orchestrator and gated by `describe_with_vision` in config.

Eyes have no blast radius, so this region never gates and never acts.
"""
from __future__ import annotations

from typing import Optional

from ..region import Region
from ..workspace import Broadcast, Workspace


class Occipital(Region):
    name = "occipital"
    system_prompt = "OCCIPITAL — visual perception of the screen."

    def __init__(self, cfg, llm, embodiment, *,
                 describe_with_vision: bool = True,
                 vision_model: str = ""):
        super().__init__(cfg, llm)
        self.embodiment = embodiment
        self.describe_with_vision = describe_with_vision
        # Default to the reflex tier model unless overridden.
        self.vision_model = vision_model or cfg.models.get("reflex", "")

    def step(self, ws: Workspace) -> Broadcast | None:
        if self.embodiment is None:
            return None
        try:
            obs = self.embodiment.observe()
        except Exception as e:  # noqa: BLE001 — perception must never crash a cycle
            return ws.post(Broadcast(
                source=self.name, kind="vision",
                content=f"(vision unavailable: {type(e).__name__})",
                salience=0.2,
            ))

        rendered = obs.render_text(limit=20)
        described = ""
        # When the backend gave us pixels but no elements/text, and we're
        # allowed to, ask the vision model what's on screen.
        if (self.describe_with_vision and not obs.elements and not obs.ocr_text
                and obs.frame is not None and obs.frame.path and self.vision_model):
            described = self.llm.describe_image(
                self.vision_model,
                "Describe what is on this screen: the app, the main UI "
                "elements and their rough positions, and any salient text. "
                "Be concise (<60 words).",
                obs.frame.path,
            )

        content = rendered
        if described:
            content = f"{rendered}; vision: {described}"

        return ws.post(Broadcast(
            source=self.name, kind="vision",
            content=content,
            salience=0.5,
            data={"frontmost_app": obs.frontmost_app,
                  "frame": obs.frame.path if obs.frame else None,
                  "n_elements": len(obs.elements)},
        ))
