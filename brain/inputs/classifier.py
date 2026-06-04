"""SalienceClassifier — deterministic, NO LLM.

The load-shedding gate for streaming. Every StreamItem from every adapter
goes through `classify(item)` BEFORE the brain can be woken up to think
about it. The classifier returns a salience in [0, 1]:

  < ambient_threshold      → drop (or store as low-salience episodic for
                              morning recall via the hippocampus).
  ambient_threshold..direct_threshold → post as workspace broadcast; the
                              attention mechanism can promote it.
  >= direct_threshold      → push into the daemon's coalesced input
                              buffer; will be processed as a real task.

Signal blend:
  - source channel prior (direct > ambient default)
  - keyword rules (config-driven word/phrase weight lists)
  - embedding similarity to past HIGH-salience items in the WorldModelStore
    or Memory (when an embedding backend is available)
  - sender priors (known senders are higher than unknown)
  - affect modulation: stress raises baseline (you notice everything),
    fatigue lowers it (you tune out)

Cost: O(len(keywords)) + at most one k-NN call against the active embedding
backend. Stays in the millisecond range on a local box.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..affect import AffectState

_WORD = re.compile(r"[a-zA-Z']+")


@dataclass
class ClassifierRules:
    """Keyword + sender + source rules. All optional; sensible defaults."""
    keywords_high: list[str] = field(default_factory=list)
    keywords_low: list[str] = field(default_factory=list)
    senders_high: list[str] = field(default_factory=list)
    senders_low: list[str] = field(default_factory=list)
    # Per-source-name baseline (e.g. {"webhook": 0.4, "rss": 0.1})
    source_baseline: dict[str, float] = field(default_factory=dict)


class SalienceClassifier:
    def __init__(self,
                 rules: ClassifierRules | None = None,
                 ambient_threshold: float = 0.25,
                 direct_threshold: float = 0.55,
                 affect: Optional[AffectState] = None,
                 memory=None,
                 embedding_backend=None,
                 max_classify_per_tick: int = 200):
        self.rules = rules or ClassifierRules()
        self.ambient_threshold = ambient_threshold
        self.direct_threshold = direct_threshold
        self.affect = affect
        self.memory = memory                      # for embedding-similarity lookups
        self.embedding_backend = embedding_backend
        self.max_classify_per_tick = max_classify_per_tick
        # Lower-cased rule lookups for hot path
        self._kw_high = {k.lower() for k in self.rules.keywords_high}
        self._kw_low = {k.lower() for k in self.rules.keywords_low}
        self._sender_high = {s.lower() for s in self.rules.senders_high}
        self._sender_low = {s.lower() for s in self.rules.senders_low}

    # ── main entry ──────────────────────────────────────────────────────────
    def classify(self, item) -> float:
        """Return salience for one item. Pure function (apart from affect)."""
        # 1. Channel + source prior
        score = self._channel_prior(item)

        # 2. Sender prior
        score += self._sender_score(item)

        # 3. Keyword rules
        score += self._keyword_score(item.content)

        # 4. Embedding similarity to past high-salience items (optional)
        score += self._embedding_score(item.content)

        # 5. Affect modulation
        score = self._apply_affect(score)

        # Final clip
        item.salience = max(0.0, min(1.0, score))
        return item.salience

    def classify_batch(self, items: Sequence) -> list:
        """Classify a list, capped at max_classify_per_tick so a flood
        doesn't burn cycles. Returns the same list (mutates salience)."""
        for it in items[: self.max_classify_per_tick]:
            self.classify(it)
        return list(items[: self.max_classify_per_tick])

    # ── signals ─────────────────────────────────────────────────────────────
    def _channel_prior(self, item) -> float:
        # Adapter's intent: direct items start at 0.55, ambient at 0.15.
        base = 0.55 if item.channel == "direct" else 0.15
        bump = self.rules.source_baseline.get(item.source, 0.0)
        return base + bump

    def _sender_score(self, item) -> float:
        sender = (item.sender or "").lower().strip()
        if not sender:
            return 0.0
        if sender in self._sender_high:
            return +0.20
        if sender in self._sender_low:
            return -0.15
        return 0.0

    def _keyword_score(self, content: str) -> float:
        toks = {t.lower() for t in _WORD.findall(content or "")}
        hi = len(self._kw_high & toks)
        lo = len(self._kw_low & toks)
        return min(0.30, 0.10 * hi) - min(0.20, 0.08 * lo)

    def _embedding_score(self, content: str) -> float:
        """Similarity to the top-K past HIGH-salience items, if a backend
        and memory are available. Capped so it can't dominate. Costs at
        most one k-NN call against the current Memory index."""
        if self.memory is None:
            return 0.0
        try:
            # Hippocampal echo: high-salience past episodes shape attention.
            hits = self.memory.retrieve_semantic(
                content, k=3, min_score=0.10)
        except Exception:
            return 0.0
        if not hits:
            return 0.0
        # Weight by stored salience × similarity
        best = 0.0
        for h in hits:
            score = float(h.get("score", 0.0)) * float(h.get("salience", 0.5))
            if score > best:
                best = score
        # Cap influence
        return min(0.20, 0.45 * best)

    def _apply_affect(self, score: float) -> float:
        if self.affect is None:
            return score
        a = self.affect
        # High stress raises salience floor — you notice everything when wired
        floor_lift = 0.18 * max(0.0, a.stress - 0.40)
        score = max(score, score + floor_lift) if score > 0 else score + floor_lift
        # Fatigue lowers responsiveness
        if a.fatigue > 0.6:
            score *= (1.0 - 0.4 * (a.fatigue - 0.6) / 0.4)
        # Curiosity modestly lifts novel-feeling items
        if a.curiosity > 0.6:
            score += 0.05 * (a.curiosity - 0.6)
        return score

    # ── routing ─────────────────────────────────────────────────────────────
    def route(self, item) -> str:
        """Bucket the item: 'drop' | 'ambient' | 'direct'.

        Adapters that already declared channel='direct' bypass the ambient
        bucket even if their classified score sits just below
        direct_threshold — the intent matters."""
        if item.salience < self.ambient_threshold:
            return "drop"
        if item.channel == "direct" or item.salience >= self.direct_threshold:
            return "direct"
        return "ambient"
