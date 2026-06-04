"""Cerebellum — fast deterministic motor / action forward model.

Neuroscience: the cerebellum hosts the brain's fastest predictive models. It
predicts the sensory consequences of motor commands (Wolpert/Miall), and
this prediction is what makes smooth movement possible. In our architecture
it's the fast path: NO LLM call, just a k-NN lookup over the existing
WorldModelStore, returning an aggregated outcome prediction plus a
confidence score.

Three callers:
  - basal ganglia (`propose_habit`): consults `quick_predict` before
    committing to habit-fire. A low-confidence or predicted-failure result
    suppresses the habit and forces System-2.
  - prefrontal: uses the prediction as a default `expected_result` so the
    LLM doesn't have to invent one from text.
  - daemon (eventually): can pre-classify ambient input items by predicted
    salience to decide whether to surface them at all.

This is the load-bearing fast path for streaming + local-LLM deployment:
when every LLM call is 1-30s on the M4 box, ANY decision the cerebellum
can answer is a decision the brain doesn't have to wake the LLM for.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..region import Region
from ..workspace import Broadcast, Workspace
from ..world_model import WorldModelStore, render_action, render_state


@dataclass
class CerebellumPrediction:
    predicted_outcome: str           # short text, aggregated from k-NN matches
    confidence: float                 # 0..1, mean similarity of matches × count factor
    predicted_ok: bool                # majority-ok across matches
    n_matches: int                    # how many world-model rows contributed
    top_similarity: float             # similarity of the single closest match

    @property
    def is_useful(self) -> bool:
        return self.n_matches >= 1 and self.top_similarity >= 0.10

    def to_dict(self) -> dict:
        return {
            "predicted_outcome": self.predicted_outcome,
            "confidence": round(self.confidence, 3),
            "predicted_ok": self.predicted_ok,
            "n_matches": self.n_matches,
            "top_similarity": round(self.top_similarity, 3),
        }


class Cerebellum(Region):
    name = "cerebellum"
    system_prompt = "CEREBELLUM — fast deterministic forward model (no LLM)."

    def __init__(self, cfg, llm, world_model: Optional[WorldModelStore] = None,
                 k: int = 3, min_top_sim: float = 0.10,
                 confidence_floor: float = 0.30):
        super().__init__(cfg, llm)  # llm unused — this region never calls it
        self.world_model = world_model
        self.k = k
        self.min_top_sim = min_top_sim
        self.confidence_floor = confidence_floor

    # ── public: fast prediction for an action ──────────────────────────────
    def quick_predict(self, ws: Workspace, effector: str,
                       args: dict) -> CerebellumPrediction:
        """Predict the outcome of `effector(args)` in the current workspace
        state via k-NN over the WorldModelStore. No LLM call."""
        if self.world_model is None:
            return CerebellumPrediction("", 0.0, True, 0, 0.0)
        state_text = render_state(ws)
        action_text = render_action(effector, args or {})
        try:
            hits = self.world_model.predict(
                state_text, action_text, k=self.k,
                min_score=self.min_top_sim,
            )
        except Exception:
            hits = []
        if not hits:
            return CerebellumPrediction("", 0.0, True, 0, 0.0)
        top_sim = float(hits[0]["score"])
        # Aggregate confidence: similarity × (count factor capped at k)
        sims = [float(h["score"]) for h in hits]
        mean_sim = sum(sims) / len(sims)
        count_factor = min(1.0, len(hits) / float(self.k))
        confidence = max(self.confidence_floor * 0.0,
                          mean_sim * (0.5 + 0.5 * count_factor))
        # Majority-vote on ok
        oks = sum(1 for h in hits if int(h.get("ok", 1)))
        predicted_ok = oks > (len(hits) - oks)
        # Pick the strongest match's outcome as the predicted result text;
        # fall back to a concatenation when the top is short
        top_outcome = (hits[0].get("outcome_text") or "").strip()
        if len(top_outcome) < 8 and len(hits) > 1:
            top_outcome = " / ".join(
                (h.get("outcome_text") or "")[:80] for h in hits[:2]
            )
        return CerebellumPrediction(
            predicted_outcome=top_outcome[:240],
            confidence=round(confidence, 4),
            predicted_ok=predicted_ok,
            n_matches=len(hits),
            top_similarity=top_sim,
        )

    # ── optional cycle step: post a status broadcast for inspection ────────
    def step(self, ws: Workspace) -> Broadcast | None:
        """Low-salience cycle broadcast so the trace shows the cerebellum
        is alive. Doesn't call quick_predict here — that happens on demand
        from BG / PFC."""
        return ws.post(Broadcast(
            source=self.name, kind="modulator",
            content=f"cerebellum online (wm_rows="
                    f"{self.world_model.count() if self.world_model else 0})",
            salience=0.18,
        ))
