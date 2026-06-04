"""Dreamer — REM agent for creative recombination.

Real dreams aren't random; they look like the brain *sampling distant
memory pairs* and trying to find or fabricate a connection. That's
exactly what this agent does:

  1. Pull a window of recent + salient rows from episodic+semantic memory.
  2. Sample pairs that are LOW in TF-IDF cosine (distant — the unobvious
     associations are where novelty lives).
  3. For each pair, ask the LLM (reflex tier — fast, sloppy on purpose)
     for ONE connecting hypothesis, metaphor, or "what if" — ≤ 25 words.
  4. Write good ones back as `mem_type=semantic` with LOW confidence
     (a dream is a hunch, not a fact) and tags=['dream'].

These dream-facts become candidates for waking corroboration: if a future
real experience matches the dream's claim, it's confirmed; if not, the
forgetter prunes it over time. Same selection pressure real semantic
memory faces.
"""
from __future__ import annotations

import math
import random
from typing import Any, List, Optional, Tuple

from ..memory import EPISODIC, Memory, SEMANTIC
from ..tfidf import TfidfIndex


_DREAM_SYSTEM = (
    "You are a REM-sleep DREAMER. Given two unrelated memory traces, "
    "produce ONE brief connecting hypothesis, metaphor, or 'what if' "
    "in first person (≤ 25 words). Be a little loose; dreams are not "
    "literal. If the pair is truly random and produces nothing, return "
    "the empty string."
)


class Dreamer:
    name = "dreamer"

    def __init__(self, dreams_per_bout: int = 3,
                 sample_window: int = 80,
                 max_pair_cosine: float = 0.20,
                 seed: Optional[int] = None):
        self.dreams_per_bout = dreams_per_bout
        self.sample_window = sample_window
        self.max_pair_cosine = max_pair_cosine
        self.rng = random.Random(seed)

    def run(self, memory: Memory, llm, model: str,
            log=None) -> dict[str, Any]:
        log = log or (lambda _m: None)
        # Sample window: recent or high-salience rows of either type
        rows = memory.conn.execute(
            "SELECT id, content, task, kind, mem_type, salience FROM episodes "
            "WHERE mem_type IN (?,?) ORDER BY ts DESC LIMIT ?",
            (EPISODIC, SEMANTIC, self.sample_window),
        ).fetchall()
        candidates: List[dict] = [memory._row_to_dict(r) for r in rows]
        if len(candidates) < 4:
            return {"dreamed": 0, "written": 0, "candidates": len(candidates)}

        # Build a TF-IDF index over the candidates for distance sampling
        idx = TfidfIndex(vocab_cap=1500)
        idx.fit((int(c["id"]), c["content"] + " " + (c.get("task") or ""))
                for c in candidates)
        if not idx.fitted:
            return {"dreamed": 0, "written": 0, "candidates": len(candidates)}

        # Pre-compute norms; we'll cosine-by-id ad hoc
        by_id = {int(c["id"]): c for c in candidates}

        def cos(a_id: int, b_id: int) -> float:
            av, bv = idx.doc_vecs.get(a_id, {}), idx.doc_vecs.get(b_id, {})
            if not av or not bv:
                return 0.0
            if len(av) > len(bv):
                av, bv = bv, av
            dot = sum(v * bv.get(t, 0.0) for t, v in av.items())
            an = idx.doc_norms.get(a_id, 0.0)
            bn = idx.doc_norms.get(b_id, 0.0)
            if an == 0 or bn == 0:
                return 0.0
            return dot / (an * bn)

        pairs: List[Tuple[int, int]] = []
        attempts = 0
        ids = list(by_id.keys())
        while len(pairs) < self.dreams_per_bout and attempts < 30:
            attempts += 1
            a, b = self.rng.sample(ids, 2)
            if cos(a, b) > self.max_pair_cosine:
                continue
            pairs.append((a, b))

        if not pairs:
            return {"dreamed": 0, "written": 0, "candidates": len(candidates)}

        written = 0
        for a_id, b_id in pairs:
            a, b = by_id[a_id], by_id[b_id]
            prompt = (
                f"Memory A ({a['mem_type']}/{a['kind']}): {a['content'][:200]}\n"
                f"Memory B ({b['mem_type']}/{b['kind']}): {b['content'][:200]}\n\n"
                "Return ONLY the dream-hypothesis (one line, ≤ 25 words, "
                "first person). Empty string if nothing."
            )
            try:
                dream = llm.chat(model, _DREAM_SYSTEM, prompt,
                                  temperature=0.9, max_tokens=80).strip()
            except Exception:
                continue
            dream = dream.strip('"').strip("'").strip()
            if len(dream) < 6:
                continue
            log(f"  💤 dream: {dream[:120]}")
            memory.store(
                task="rem_sleep", kind="dream", content=dream[:280],
                salience=0.45,           # dream hunches are not strong
                mem_type=SEMANTIC,
                tags=["dream"],
            )
            written += 1

        return {"dreamed": len(pairs), "written": written,
                "candidates": len(candidates)}
