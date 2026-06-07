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
                 confidence_floor: float = 0.30,
                 forward_model=None, embedding_backend=None,
                 visual_forward_model=None):
        super().__init__(cfg, llm)  # llm unused — this region never calls it
        self.world_model = world_model
        self.k = k
        self.min_top_sim = min_top_sim
        self.confidence_floor = confidence_floor
        # Optional learned forward model (numpy MLP trained during sleep). When
        # present (and the embedding backend is dense), its learned success
        # probability augments the k-NN majority vote. Set/refreshed by the
        # ForwardModelTrainer; loaded from a checkpoint at construction.
        self.forward_model = forward_model
        self.embedding_backend = embedding_backend
        # Optional VISUAL forward model (DINOv2 latent space), trained by the
        # VisualForwardModelTrainer during sleep. Consumed by visual planning
        # (V3); refreshed in-place after each sleep training pass.
        self.visual_forward_model = visual_forward_model

    def _embed(self, text: str):
        bk = self.embedding_backend
        if bk is None or not getattr(bk, "persistent", False):
            return None
        try:
            blob = bk.encode_one(text or "")
            return bk.from_bytes(blob) if blob else None
        except Exception:
            return None

    # ── public: fast prediction for an action ──────────────────────────────
    def quick_predict(self, ws: Workspace, effector: str,
                       args: dict) -> CerebellumPrediction:
        """Predict the outcome of `effector(args)` in the current workspace
        state via k-NN over the WorldModelStore. No LLM call."""
        if self.world_model is None and self.forward_model is None:
            return CerebellumPrediction("", 0.0, True, 0, 0.0)
        state_text = render_state(ws)
        action_text = render_action(effector, args or {})
        hits = []
        if self.world_model is not None:
            try:
                hits = self.world_model.predict(
                    state_text, action_text, k=self.k,
                    min_score=self.min_top_sim,
                )
            except Exception:
                hits = []

        # Learned forward model — consulted independently of k-NN, because its
        # whole value is generalizing where k-NN has no neighbour. Returns the
        # learned success probability when a trained model + dense embeddings
        # are available.
        learned_ok = None
        learned_conf = 0.0
        if self.forward_model is not None:
            s = self._embed(state_text)
            a = self._embed(action_text)
            if s is not None and a is not None:
                try:
                    _, ok_prob = self.forward_model.predict(s, a)
                    learned_ok = ok_prob >= 0.5
                    learned_conf = round(abs(ok_prob - 0.5) * 2.0, 4)
                except Exception:
                    learned_ok = None

        if not hits:
            # No retrieval neighbour. If the learned model spoke, trust it;
            # otherwise we genuinely know nothing.
            if learned_ok is None:
                return CerebellumPrediction("", 0.0, True, 0, 0.0)
            return CerebellumPrediction(
                predicted_outcome="", confidence=learned_conf,
                predicted_ok=bool(learned_ok), n_matches=0, top_similarity=0.0)

        top_sim = float(hits[0]["score"])
        # Aggregate confidence: similarity × (count factor capped at k)
        sims = [float(h["score"]) for h in hits]
        mean_sim = sum(sims) / len(sims)
        count_factor = min(1.0, len(hits) / float(self.k))
        confidence = max(self.confidence_floor * 0.0,
                          mean_sim * (0.5 + 0.5 * count_factor))
        # Majority-vote on ok (k-NN baseline)
        oks = sum(1 for h in hits if int(h.get("ok", 1)))
        predicted_ok = oks > (len(hits) - oks)

        # Learned override: generalizes between observed triples instead of
        # voting over the nearest few; blends confidence toward its certainty.
        if learned_ok is not None:
            predicted_ok = bool(learned_ok)
            confidence = max(confidence, learned_conf)

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

    # ── visual prediction (DINOv2 latent space) ────────────────────────────
    def visual_predict(self, state_vis, effector: str, args: dict,
                       goal_vis=None) -> Optional[CerebellumPrediction]:
        """Predict the outcome of a SCREEN action from a DINOv2 screen
        embedding via the learned visual forward model. Returns None when no
        visual model / no visual state / the action can't be embedded — callers
        fall back to the text k-NN path. The predicted next-screen embedding is
        scored against `goal_vis` (a target screen) when provided, threading
        goal-distance into confidence — the substrate of visual MPC (Phase 1b)."""
        if self.visual_forward_model is None or state_vis is None:
            return None
        bk = self.embedding_backend
        if bk is None or not getattr(bk, "persistent", False):
            return None
        action_text = render_action(effector, args or {})
        try:
            blob = bk.encode_one(action_text)
            a = bk.from_bytes(blob) if blob else None
        except Exception:
            a = None
        if a is None:
            return None
        try:
            next_vis, ok_prob = self.visual_forward_model.predict(state_vis, a)
        except Exception:
            return None
        conf = round(abs(float(ok_prob) - 0.5) * 2.0, 4)
        # Optional goal-distance: how close is the imagined next screen to the
        # target screen? Cosine in (unit-normalized) DINO space.
        if goal_vis is not None:
            from ..imagination import cosine
            sim = cosine(next_vis, goal_vis)
            conf = round(max(0.0, min(1.0, 0.5 * conf + 0.5 * max(0.0, sim))), 4)
        # top_similarity floored so is_useful is True whenever the model spoke.
        return CerebellumPrediction(
            predicted_outcome="", confidence=conf,
            predicted_ok=bool(ok_prob >= 0.5), n_matches=1,
            top_similarity=max(self.min_top_sim, conf))

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
