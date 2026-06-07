"""Imagination — shared substrate for model-based planning and replay.

Both the waking Predictor (imagination-based planning, `regions/predictor.py`)
and the sleeping GenerativeReplay agent (Phase 2) need the same primitives:

  - embed arbitrary text into the world model's latent space,
  - measure how close a predicted outcome is to the goal,
  - turn a trajectory of (signature, effector, args, reward) into discounted
    Monte-Carlo returns for credit assignment.

These were duplicated across `cerebellum._embed`, `forward_model_trainer`, and
`orchestrator._assign_credit`. Centralizing them here keeps the planning and
replay code thin and guarantees both halves of the flywheel reason in the
*same* latent space. Pure-Python (no numpy import at module load) so it stays
importable on machines without the optional ML deps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple


# A vector is whatever the embedding backend returns (list[float] or ndarray);
# we only ever iterate it, so we stay duck-typed and numpy-free here.
Vec = Sequence[float]


def embed_text(backend, text: str) -> Optional[Vec]:
    """Embed `text` via a dense persistent backend's encode_one/from_bytes
    round-trip. Returns None for sparse/absent backends (TF-IDF) or on any
    failure — callers degrade gracefully, exactly like the cerebellum does."""
    if backend is None or not getattr(backend, "persistent", False):
        return None
    try:
        blob = backend.encode_one(text or "")
        return backend.from_bytes(blob) if blob else None
    except Exception:
        return None


def cosine(a: Optional[Vec], b: Optional[Vec]) -> float:
    """Cosine similarity in [-1, 1]; 0.0 when either side is missing/zero.
    Works for python lists and numpy arrays alike (iteration only)."""
    if a is None or b is None:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        x = float(x)
        y = float(y)
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def goal_text(ws) -> str:
    """The task goal as plain text, pulled from the sensory percept."""
    percept = ws.latest(kind="percept")
    if percept and percept.data:
        return str(percept.data.get("goal", "") or "")
    return ""


def goal_vec(ws, backend) -> Optional[Vec]:
    """Embed the goal once. Cached on the workspace when it allows attribute
    assignment (Workspace is a plain dataclass, so it does); otherwise recomputed."""
    cached = getattr(ws, "_goal_vec_cache", None)
    if cached is not None:
        return cached
    vec = embed_text(backend, goal_text(ws))
    try:
        ws._goal_vec_cache = vec
    except Exception:
        pass
    return vec


def discounted_returns(
    trajectory: Sequence[Tuple[Any, str, dict, float]],
    gamma: float = 0.9,
) -> List[Tuple[Any, str, dict, float]]:
    """Backward discounted Monte-Carlo returns: G_t = r_t + gamma·G_{t+1}.
    Input rows are (signature, effector, args, reward); output rows are
    (signature, effector, args, G_t) in the SAME forward order. Used by both
    waking credit assignment and sleep-time replay so an action that set up a
    later success gets credit for it."""
    out: List[Tuple[Any, str, dict, float]] = []
    g = 0.0
    for sig, eff, args, r in reversed(list(trajectory)):
        g = float(r) + gamma * g
        out.append((sig, eff, args, g))
    out.reverse()
    return out


@dataclass
class Rollout:
    """One imagined trajectory through the learned forward model. Populated by
    multi-step planning (Phase 1b) and replay (Phase 2); the single-step
    Monitor only needs the head of it."""
    actions: List[str] = field(default_factory=list)
    pred_states: List[Vec] = field(default_factory=list)
    success_probs: List[float] = field(default_factory=list)
    score: float = 0.0
