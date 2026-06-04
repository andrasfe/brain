"""Lightweight TF-IDF retrieval — pure Python, no extra deps.

Used by `Memory.retrieve_semantic` to rank rows by semantic similarity beyond
keyword overlap. Surprisingly effective for the corpus sizes a single brain
accumulates (hundreds to low thousands of episodes). When the embedding
question becomes load-bearing — say tens of thousands of episodes, or
cross-language reuse — swap in a real embedding backend at the same interface
(`fit / score / topk`).

Design choices:
  * In-memory inverted index: term → list[(doc_id, tf)]. Cheap to rebuild
    on `fit()`; called lazily by Memory after batches of inserts.
  * Document-length normalization (cosine, with tf-idf weights).
  * Vocabulary cap so a long-running brain doesn't blow up: keep the top
    `vocab_cap` most frequent terms (df-ranked) — the long tail of
    one-off proper nouns is noise.
  * Stop-list small and English; the goal is to discount the very most
    common closed-class words, not to do full NLP.
"""
from __future__ import annotations

import math
import re
from typing import Iterable, List, Optional, Sequence, Tuple

_WORD = re.compile(r"[a-zA-Z][a-zA-Z']+")
_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "by", "for", "with", "as", "is", "are", "was", "were", "be", "been",
    "being", "this", "that", "these", "those", "it", "its", "i", "you", "he",
    "she", "we", "they", "them", "his", "her", "their", "my", "your", "our",
    "me", "him", "us", "do", "does", "did", "have", "has", "had", "not",
    "no", "yes", "so", "than", "then", "from", "into", "about", "out", "up",
    "down", "over", "under", "again", "more", "most", "very", "just",
}


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(text or "") if t.lower() not in _STOP]


class TfidfIndex:
    """Sparse TF-IDF index over a list of (doc_id, text) pairs."""

    def __init__(self, vocab_cap: int = 4000):
        self.vocab_cap = vocab_cap
        self.df: dict[str, int] = {}           # term → document frequency
        self.idf: dict[str, float] = {}        # term → idf weight
        self.doc_vecs: dict[int, dict[str, float]] = {}  # doc_id → {term: tfidf}
        self.doc_norms: dict[int, float] = {}  # ‖doc_vec‖₂
        self.n_docs: int = 0
        self.fitted: bool = False

    # ── fit ─────────────────────────────────────────────────────────────────
    def fit(self, docs: Iterable[Tuple[int, str]]) -> None:
        docs = list(docs)
        if not docs:
            self.df, self.idf, self.doc_vecs, self.doc_norms = {}, {}, {}, {}
            self.n_docs = 0
            self.fitted = True
            return

        # First pass: term frequencies per doc + document frequency
        tfs: dict[int, dict[str, int]] = {}
        df: dict[str, int] = {}
        for doc_id, text in docs:
            tf: dict[str, int] = {}
            for tok in _tokens(text):
                tf[tok] = tf.get(tok, 0) + 1
            tfs[doc_id] = tf
            for term in tf:
                df[term] = df.get(term, 0) + 1

        # Vocabulary cap by document frequency: keep the most informative,
        # not the absolute rarest (df=1 is often a typo/proper noun nobody
        # will ever query). Keep top-N by df, with a floor to retain rare
        # but recurring terms above df>=2.
        if len(df) > self.vocab_cap:
            kept = sorted(df.items(), key=lambda x: x[1], reverse=True)[: self.vocab_cap]
            df = dict(kept)
            for did in tfs:
                tfs[did] = {t: c for t, c in tfs[did].items() if t in df}

        # IDF (smoothed, classic formulation)
        n = len(tfs)
        self.idf = {term: math.log((n + 1) / (cnt + 1)) + 1.0
                    for term, cnt in df.items()}
        self.df = df
        self.n_docs = n

        # Doc vectors (tfidf) + L2 norms
        self.doc_vecs.clear()
        self.doc_norms.clear()
        for doc_id, tf in tfs.items():
            vec: dict[str, float] = {}
            for term, c in tf.items():
                idf = self.idf.get(term)
                if idf is None:
                    continue
                # 1 + log(tf) damping
                vec[term] = (1.0 + math.log(c)) * idf
            self.doc_vecs[doc_id] = vec
            self.doc_norms[doc_id] = math.sqrt(sum(v * v for v in vec.values()))
        self.fitted = True

    # ── query ──────────────────────────────────────────────────────────────
    def _query_vec(self, query: str) -> Tuple[dict[str, float], float]:
        toks = _tokens(query)
        tf: dict[str, int] = {}
        for tok in toks:
            tf[tok] = tf.get(tok, 0) + 1
        vec: dict[str, float] = {}
        for term, c in tf.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            vec[term] = (1.0 + math.log(c)) * idf
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return vec, norm

    def topk(self, query: str, k: int = 5,
             eligible: Optional[Sequence[int]] = None) -> List[Tuple[int, float]]:
        """Return [(doc_id, cosine_sim)] sorted desc. `eligible` restricts the
        candidate set (use for typed retrieval)."""
        if not self.fitted or not self.doc_vecs:
            return []
        qvec, qnorm = self._query_vec(query)
        if qnorm == 0:
            return []
        eligible_set = set(eligible) if eligible is not None else None

        scored: list[Tuple[int, float]] = []
        for doc_id, dvec in self.doc_vecs.items():
            if eligible_set is not None and doc_id not in eligible_set:
                continue
            # Sparse dot product: iterate the smaller vector
            if len(qvec) < len(dvec):
                dot = sum(v * dvec.get(t, 0.0) for t, v in qvec.items())
            else:
                dot = sum(v * qvec.get(t, 0.0) for t, v in dvec.items())
            if dot == 0:
                continue
            dnorm = self.doc_norms.get(doc_id, 0.0)
            if dnorm == 0:
                continue
            scored.append((doc_id, dot / (qnorm * dnorm)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
