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
                 vision_model: str = "",
                 screen_model=None, memory=None, visual_embedder=None):
        super().__init__(cfg, llm)
        self.embodiment = embodiment
        self.describe_with_vision = describe_with_vision
        # Default to the reflex tier model unless overridden.
        self.vision_model = vision_model or cfg.models.get("reflex", "")
        # Learned screen-sequence model (next-screen predictor) + the memory it
        # embeds against, for novelty + anticipation signals. Optional.
        self.screen_model = screen_model
        self.memory = memory
        # When set, the current screen is embedded with the SAME encoder the
        # observation stream uses (DINOv2), keeping novelty/anticipation in the
        # same space as the trained model. Else we fall back to text-embedding.
        self.visual_embedder = visual_embedder
        self._last_prediction = None   # predicted embedding for THIS step

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

        # ── learned dynamics: novelty + anticipation (when a screen-sequence
        # model is loaded and we can embed the current screen) ─────────────
        novelty = None
        anticipated = None
        salience = 0.5
        if self.screen_model is not None and self.memory is not None:
            # Same encoder as the observation stream: DINOv2 image embed when
            # available, else text-embed of the description.
            cur = None
            if (self.visual_embedder is not None
                    and getattr(self.visual_embedder, "available", False)
                    and obs.frame is not None and obs.frame.path):
                cur = self.visual_embedder.embed(obs.frame.path)
            if cur is None and described:
                cur = self._embed(described)
            if cur is not None:
                from ..screen_model import cosine_distance, make_screen_model  # noqa: F401
                # novelty: did this screen match what we predicted last step?
                if self._last_prediction is not None:
                    novelty = round(cosine_distance(self._last_prediction, cur), 3)
                    # a surprising screen is more salient (grabs attention)
                    salience = min(0.85, 0.5 + 0.4 * max(0.0, novelty - 0.3))
                # anticipate the NEXT screen, render as "you usually do X next"
                try:
                    import time as _t
                    hour = _t.localtime().tm_hour + _t.localtime().tm_min / 60.0
                    pred = self.screen_model.predict_next(cur, hour)
                    self._last_prediction = pred
                    anticipated = self._nearest_observation(pred)
                except Exception:
                    self._last_prediction = None
        if novelty is not None:
            content += f"  (novelty={novelty})"
        if anticipated:
            content += f"  [likely next: {anticipated[:60]}]"

        # PIXEL-DROP: occipital captured its own frame; delete it once used.
        if obs.frame is not None and obs.frame.path:
            try:
                import os
                os.remove(obs.frame.path)
            except OSError:
                pass

        return ws.post(Broadcast(
            source=self.name, kind="vision",
            content=content,
            salience=salience,
            data={"frontmost_app": obs.frontmost_app,
                  "n_elements": len(obs.elements),
                  "novelty": novelty, "anticipated": anticipated},
        ))

    def _embed(self, text: str):
        bk = getattr(self.memory, "backend", None)
        if bk is None or not getattr(bk, "persistent", False):
            return None
        try:
            blob = bk.encode_one(text or "")
            return bk.from_bytes(blob) if blob else None
        except Exception:
            return None

    def _nearest_observation(self, pred_emb):
        """Match the predicted next-embedding to the closest past observation's
        text — a human-legible 'you usually do X next'."""
        if self.memory is None:
            return None
        try:
            from ..screen_model import cosine_distance
            from ..memory import OBSERVATION
            rows = self.memory.conn.execute(
                "SELECT content, embedding FROM episodes WHERE mem_type=? "
                "AND embedding IS NOT NULL ORDER BY ts DESC LIMIT 300",
                (OBSERVATION,)).fetchall()
            best, best_d = None, 1e9
            bk = self.memory.backend
            for r in rows:
                v = bk.from_bytes(r["embedding"]) if r["embedding"] else None
                if not v:
                    continue
                d = cosine_distance(pred_emb, v)
                if d < best_d:
                    best_d, best = d, r["content"]
            return best if best_d < 0.5 else None
        except Exception:
            return None
