"""Embedding backends — semantic similarity for `Memory.retrieve_semantic`.

LeCun's JEPA-style world models live in a *learned latent space*. Our brain
can move in that direction without an ML stack by swapping the cheap TF-IDF
default for real text embeddings. Three backends, all behind one interface
so callers (Memory, world_model later, dreamer later) don't care which:

  TfidfBackend         — pure Python, no deps. The cheap default. What we
                          already had; just lifted behind the interface.
  OpenRouterBackend    — HTTP /api/v1/embeddings against OpenRouter using
                          the configured embedding model id. Pluggable: any
                          OpenAI-compatible endpoint works.
  SentenceTfBackend    — optional, only if `sentence-transformers` is
                          installed locally. Free, fast, deterministic; the
                          recommended choice for offline / cost-sensitive
                          setups.

Persistence: embeddings are cached as BLOB on the `episodes.embedding` column
(added by Memory's migration). The TF-IDF backend never persists — it just
rebuilds the in-memory index on demand, which is cheap. The neural backends
embed lazily on read (first time a row is needed for ranking) and write back.

Selection logic (`make_backend(cfg, llm=None)`):
  1. cfg.memory.embedding_backend explicitly named → that one (errors if
     unavailable — fail loud rather than silently fall back to TF-IDF).
  2. None / "auto" → try sentence-transformers, then openrouter (if a
     model is configured), then tfidf.

The interface is sync. Async batching is a nice-to-have we can add later;
end-of-task consolidation and lazy-read fill are both rare enough that the
HTTP latency is fine on the executive timeline (~hundreds of ms).
"""
from __future__ import annotations

import math
import struct
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Sequence, Tuple

from .tfidf import TfidfIndex


# ── interface ──────────────────────────────────────────────────────────────
class EmbeddingBackend(ABC):
    name: str = "abstract"
    dim: int = 0          # 0 means "variable" (e.g. TF-IDF sparse)
    persistent: bool = False  # True = embeddings worth caching on disk

    @abstractmethod
    def fit(self, docs: Iterable[Tuple[int, str]]) -> None:
        """Index `docs` (TF-IDF rebuilds; neural backends embed each)."""

    @abstractmethod
    def topk(self, query: str, k: int,
             eligible: Optional[Sequence[int]] = None) -> List[Tuple[int, float]]:
        """Return [(doc_id, score)] sorted desc. Score is cosine in [0,1]."""

    # Optional persistence helpers — neural backends override.
    def encode_one(self, text: str) -> Optional[bytes]:
        """Return a portable byte representation of one document's embedding,
        or None if the backend doesn't persist (e.g. TF-IDF)."""
        return None

    def from_bytes(self, blob: bytes) -> Optional[List[float]]:
        """Decode an embedding cached on disk. None when not persistable."""
        return None

    def close(self) -> None:
        pass


# ── TF-IDF backend (default, no deps) ───────────────────────────────────────
class TfidfBackend(EmbeddingBackend):
    name = "tfidf"
    persistent = False  # we just rebuild the index on demand

    def __init__(self, vocab_cap: int = 4000):
        self._idx = TfidfIndex(vocab_cap=vocab_cap)
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def mark_stale(self) -> None:
        self._fitted = False
        self._idx.fitted = False

    def fit(self, docs: Iterable[Tuple[int, str]]) -> None:
        self._idx.fit(docs)
        self._fitted = True

    def topk(self, query, k, eligible=None):
        if not self._fitted:
            return []
        return self._idx.topk(query, k=k, eligible=eligible)


# ── OpenRouter / OpenAI-compatible embedding backend ────────────────────────
class OpenRouterBackend(EmbeddingBackend):
    """HTTP `/embeddings` against any OpenAI-compatible endpoint.

    Caches embeddings in-memory keyed by doc_id; the caller (Memory) persists
    them as BLOBs on the episodes table via `encode_one` / `from_bytes`.
    """
    name = "openrouter"
    persistent = True

    def __init__(self, llm, model: str, dim: int = 1536, batch_size: int = 64):
        # We reuse the LLM's httpx client + auth — same base_url, same token.
        # Subclassing or a separate HTTP client is overkill.
        self._llm = llm
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self._vecs: dict[int, List[float]] = {}
        self._norms: dict[int, float] = {}
        self._fitted = False

    # ── HTTP ────────────────────────────────────────────────────────────────
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Call /embeddings; return one vector per input."""
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            payload = {"model": self.model, "input": chunk}
            r = self._llm._client.post("/embeddings", json=payload)
            r.raise_for_status()
            data = r.json()
            # OpenAI-compatible shape: {"data": [{"embedding": [...]}, ...]}
            for item in data["data"]:
                out.append([float(x) for x in item["embedding"]])
        if out:
            self.dim = len(out[0])
        return out

    # ── interface ───────────────────────────────────────────────────────────
    def fit(self, docs: Iterable[Tuple[int, str]]) -> None:
        docs = list(docs)
        # Embed any docs not already cached
        missing_ids = [did for did, _ in docs if did not in self._vecs]
        missing_texts = [text for (did, text) in docs if did not in self._vecs]
        if missing_ids:
            vecs = self._embed_texts(missing_texts)
            for did, v in zip(missing_ids, vecs):
                self._vecs[did] = v
                self._norms[did] = math.sqrt(sum(x * x for x in v))
        # Drop any cached vecs no longer in the eligible set (cheap GC)
        keep = {did for did, _ in docs}
        for did in list(self._vecs):
            if did not in keep:
                self._vecs.pop(did, None)
                self._norms.pop(did, None)
        self._fitted = True

    def topk(self, query, k, eligible=None):
        if not self._fitted or not self._vecs:
            return []
        # Embed the query (no caching by query string — typically unique)
        qvec = self._embed_texts([query])[0]
        qnorm = math.sqrt(sum(x * x for x in qvec))
        if qnorm == 0:
            return []
        eligible_set = set(eligible) if eligible is not None else None
        scored: list[Tuple[int, float]] = []
        for did, dvec in self._vecs.items():
            if eligible_set is not None and did not in eligible_set:
                continue
            dnorm = self._norms.get(did, 0.0)
            if dnorm == 0:
                continue
            dot = sum(a * b for a, b in zip(qvec, dvec))
            scored.append((did, dot / (qnorm * dnorm)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # ── persistence helpers ────────────────────────────────────────────────
    def encode_one(self, text: str) -> Optional[bytes]:
        v = self._embed_texts([text])[0]
        return _pack_floats(v)

    def from_bytes(self, blob: bytes) -> Optional[List[float]]:
        return _unpack_floats(blob)

    def remember(self, doc_id: int, vec: List[float]) -> None:
        """Load a previously-cached embedding into the in-memory index
        without re-calling the network."""
        self._vecs[doc_id] = vec
        self._norms[doc_id] = math.sqrt(sum(x * x for x in vec))


# ── optional local sentence-transformers backend ───────────────────────────
class SentenceTfBackend(EmbeddingBackend):
    """Local embeddings via `sentence-transformers`. Imported lazily so the
    project keeps no hard dependency on it."""
    name = "sentence_transformers"
    persistent = True

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:  # pragma: no cover — only fires when chosen
            raise RuntimeError(
                "sentence-transformers not installed; "
                "`pip install sentence-transformers` or pick another backend"
            ) from e
        self._model = SentenceTransformer(model)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self._vecs: dict[int, list[float]] = {}
        self._norms: dict[int, float] = {}
        self._fitted = False

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        arr = self._model.encode(texts, convert_to_numpy=True,
                                  normalize_embeddings=False)
        return [list(map(float, row)) for row in arr]

    def fit(self, docs):
        docs = list(docs)
        missing_ids = [did for did, _ in docs if did not in self._vecs]
        missing_texts = [text for (did, text) in docs if did not in self._vecs]
        if missing_ids:
            vecs = self._embed_texts(missing_texts)
            for did, v in zip(missing_ids, vecs):
                self._vecs[did] = v
                self._norms[did] = math.sqrt(sum(x * x for x in v))
        keep = {did for did, _ in docs}
        for did in list(self._vecs):
            if did not in keep:
                self._vecs.pop(did, None)
                self._norms.pop(did, None)
        self._fitted = True

    def topk(self, query, k, eligible=None):
        if not self._fitted or not self._vecs:
            return []
        qvec = self._embed_texts([query])[0]
        qnorm = math.sqrt(sum(x * x for x in qvec))
        if qnorm == 0:
            return []
        eligible_set = set(eligible) if eligible is not None else None
        scored: list[Tuple[int, float]] = []
        for did, dvec in self._vecs.items():
            if eligible_set is not None and did not in eligible_set:
                continue
            dnorm = self._norms.get(did, 0.0)
            if dnorm == 0:
                continue
            dot = sum(a * b for a, b in zip(qvec, dvec))
            scored.append((did, dot / (qnorm * dnorm)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def encode_one(self, text: str) -> bytes:
        return _pack_floats(self._embed_texts([text])[0])

    def from_bytes(self, blob: bytes) -> List[float]:
        return _unpack_floats(blob)

    def remember(self, doc_id: int, vec: List[float]) -> None:
        self._vecs[doc_id] = vec
        self._norms[doc_id] = math.sqrt(sum(x * x for x in vec))


# ── float packing (compact persistence) ─────────────────────────────────────
def _pack_floats(v: Sequence[float]) -> bytes:
    """4 bytes per dim. Header = dim (uint32, little-endian)."""
    n = len(v)
    return struct.pack(f"<I{n}f", n, *v)


def _unpack_floats(blob: bytes) -> List[float]:
    if not blob or len(blob) < 4:
        return []
    (n,) = struct.unpack("<I", blob[:4])
    if 4 + 4 * n > len(blob):
        return []
    return list(struct.unpack(f"<{n}f", blob[4:4 + 4 * n]))


# ── factory ────────────────────────────────────────────────────────────────
def make_backend(cfg, llm=None) -> EmbeddingBackend:
    """Build the embedding backend named in cfg.memory.embedding_backend.

    Values: "tfidf" | "openrouter" | "sentence_transformers" | "auto" | unset.
    "auto" / unset = try sentence-transformers, then openrouter (if model
    configured), then tfidf. Explicit names FAIL LOUD if unavailable so a
    misconfiguration is visible rather than silently downgraded.
    """
    mem_cfg = cfg.memory or {}
    name = (mem_cfg.get("embedding_backend") or "auto").lower()
    model = mem_cfg.get("embedding_model")  # used by neural backends
    vocab_cap = int(mem_cfg.get("tfidf_vocab_cap", 4000))

    if name == "tfidf":
        return TfidfBackend(vocab_cap=vocab_cap)
    if name == "openrouter":
        if not llm or not model:
            raise RuntimeError(
                "embedding_backend=openrouter requires an LLM client and "
                "memory.embedding_model in config.yaml")
        return OpenRouterBackend(llm, model=model)
    if name == "sentence_transformers":
        return SentenceTfBackend(model=model or "sentence-transformers/all-MiniLM-L6-v2")

    # auto: try local → openrouter → tfidf
    try:
        return SentenceTfBackend(
            model=model or "sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        pass
    if llm and model:
        try:
            return OpenRouterBackend(llm, model=model)
        except Exception:
            pass
    return TfidfBackend(vocab_cap=vocab_cap)
