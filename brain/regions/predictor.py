"""Predictor — the learned Monitor (imagination-based planning, single-step).

MAP's headline Hanoi result (0% invalid moves) comes from a Monitor that
rejects rule-violating moves before they commit. We do the same thing, but the
validity check is **learned, not coded**: before a deliberate (System-2) action
is gated, the Predictor asks the brain's forward model "what happens if I do
this here?" and vetoes actions it confidently predicts will fail in the current
state. Nobody writes `is_valid_move()` — the rejection emerges from experience
(real observations + the sleep-trained forward model).

Deterministic, NO LLM call. It reuses the cerebellum's `quick_predict`, which
already blends the k-NN `WorldModelStore` with the learned forward model's
success probability — so the Monitor works from k-NN alone at cold start and
sharpens as the forward model trains during sleep.

Single-step (depth=1) today: it evaluates the proposed action. The scaffolding
(candidate ranking, `imagination.Rollout`, goal-distance helpers) is in place
for the multi-step latent rollout (Phase 1b) and generative replay (Phase 2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..region import Region
from ..workspace import Broadcast, Workspace


@dataclass
class PlanEvaluation:
    chosen_effector: str       # best candidate after ranking (== proposed when single)
    chosen_args: dict
    vetoed: bool               # learned Monitor predicts confident failure
    reason: str                # human-readable veto reason (empty when not vetoed)
    predicted_ok: bool         # the model's go/no-go on the chosen action
    confidence: float          # 0..1 confidence behind predicted_ok
    n_candidates: int
    n_matches: int             # world-model neighbours that informed the call

    def to_dict(self) -> dict:
        return {
            "chosen_effector": self.chosen_effector,
            "vetoed": self.vetoed,
            "reason": self.reason,
            "predicted_ok": self.predicted_ok,
            "confidence": round(self.confidence, 3),
            "n_candidates": self.n_candidates,
            "n_matches": self.n_matches,
        }


class Predictor(Region):
    name = "predictor"
    system_prompt = "PREDICTOR — deterministic learned Monitor (no LLM)."

    def __init__(self, cfg, llm, cerebellum=None, world_model=None, *,
                 min_rows: int = 40, veto_floor: float = 0.30,
                 goal_weight: float = 0.5):
        super().__init__(cfg, llm)  # llm unused — this region never calls it
        self.cerebellum = cerebellum
        self.world_model = world_model
        self.min_rows = int(min_rows)
        self.veto_floor = float(veto_floor)
        self.goal_weight = float(goal_weight)

    @property
    def is_active(self) -> bool:
        """Active only once there's something to predict FROM: enough
        world-model rows for k-NN, or a trained forward model. Below that it
        no-ops (returns None), so cold-start behavior is unchanged."""
        if self.cerebellum is None:
            return False
        wm = self.world_model or getattr(self.cerebellum, "world_model", None)
        has_data = wm is not None and wm.count() >= self.min_rows
        has_model = getattr(self.cerebellum, "forward_model", None) is not None
        return bool(has_data or has_model)

    def evaluate(self, ws: Workspace,
                 candidates: List[Tuple[str, dict]],
                 depth: int = 1) -> Optional[PlanEvaluation]:
        """Score each candidate action by its predicted outcome in the current
        state and return the best, flagged `vetoed` when the top candidate is a
        confident predicted failure. depth>1 (multi-step latent rollout) is
        Phase 1b; today we evaluate one step."""
        if not self.is_active or not candidates:
            return None
        scored = []
        for eff, args in candidates:
            pred = self.cerebellum.quick_predict(ws, eff, args or {})
            scored.append((eff, args or {}, pred))
        # Prefer predicted-ok candidates, then higher confidence.
        scored.sort(key=lambda t: (1 if t[2].predicted_ok else 0, t[2].confidence),
                    reverse=True)
        eff, args, pred = scored[0]
        vetoed = bool(pred.is_useful and not pred.predicted_ok
                      and pred.confidence >= self.veto_floor)
        reason = ""
        if vetoed:
            outcome = (pred.predicted_outcome or "").strip()
            reason = (f"predicted failure (conf {pred.confidence:.2f}, "
                      f"{pred.n_matches} similar)"
                      + (f": {outcome[:80]}" if outcome else ""))
        return PlanEvaluation(
            chosen_effector=eff, chosen_args=args, vetoed=vetoed, reason=reason,
            predicted_ok=pred.predicted_ok, confidence=pred.confidence,
            n_candidates=len(candidates), n_matches=pred.n_matches,
        )

    def step(self, ws: Workspace) -> Optional[Broadcast]:
        """No autonomous cycle role — the orchestrator calls evaluate() on
        demand, like the cerebellum."""
        return None
