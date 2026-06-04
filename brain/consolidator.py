"""Consolidator — the brain's offline 'sleep' pass.

Episodic traces are cheap and lossy: they record what happened, blow-by-blow.
Semantic memory is what you actually USE day-to-day: distilled facts and
generalizations ("I always struggle with shell quoting", "Maria calls on
Sundays"). Real brains do this consolidation primarily during sleep, with
the hippocampus replaying episodes to neocortex which extracts the durable
gist.

This module does the same thing in miniature:

  1. Pull the last N episodic rows that haven't been consolidated yet.
  2. Cluster them by TF-IDF cosine similarity (cheap; reuses Memory's index).
  3. For each cluster of size ≥ min_cluster_size, ask the LLM to produce ONE
     short semantic fact summarizing the recurring pattern.
  4. Write that fact back as `mem_type=semantic` with high salience and a
     `tags=consolidated` marker so we don't re-consolidate it next pass.

Runs in two contexts:
  - end-of-task: orchestrator calls `consolidate(memory, llm, model, …)` after
    `run()` finishes (cheap; default ≤ 3 facts per pass).
  - standalone CLI: `python -m brain.consolidator` for a deeper pass over all
    accumulated episodes (the offline 'sleep'). Use `--n 200 --max-facts 10`
    for a full sweep at the end of a session.

The consolidator owns the LLM call; Memory itself stays LLM-free.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from brain.config import load_config  # noqa: E402
from brain.llm import LLM  # noqa: E402
from brain.memory import EPISODIC, SEMANTIC, Memory  # noqa: E402
from brain.tfidf import TfidfIndex, _tokens  # noqa: E402


_CONSOLIDATED_TAG = "consolidated"


def _already_consolidated(row: dict) -> bool:
    tags = (row.get("tags") or "").split(",")
    return _CONSOLIDATED_TAG in tags


def _fetch_candidates(memory: Memory, n: int) -> list[dict]:
    """Most recent episodic rows that don't carry the consolidated tag."""
    # Pull a generous window then filter; the tag column is sparse and not
    # worth a dedicated index for now.
    raw = memory.conn.execute(
        "SELECT * FROM episodes WHERE mem_type=? "
        "ORDER BY ts DESC LIMIT ?",
        (EPISODIC, n * 3),
    ).fetchall()
    rows: list[dict] = []
    for r in raw:
        d = memory._row_to_dict(r)
        if _already_consolidated(d):
            continue
        rows.append(d)
        if len(rows) >= n:
            break
    return rows


def _cluster(rows: list[dict], threshold: float = 0.18,
             vocab_cap: int = 1500) -> list[list[dict]]:
    """Greedy single-link clustering over TF-IDF cosine.

    Cheap; we don't need k-means or hierarchical for batches of <100. Each
    row joins the highest-similarity existing cluster if any cluster's
    centroid (mean vec) scores above `threshold`; otherwise it starts its
    own. Returns clusters sorted by size desc."""
    if not rows:
        return []
    idx = TfidfIndex(vocab_cap=vocab_cap)
    docs: list[Tuple[int, str]] = [
        (int(r["id"]), (r["content"] + " " + (r.get("task") or "")))
        for r in rows
    ]
    idx.fit(docs)
    by_id = {int(r["id"]): r for r in rows}

    clusters: list[dict] = []  # each: {"ids": set[int], "centroid": dict[str,float]}

    def cosine(a: dict, b: dict) -> float:
        if not a or not b:
            return 0.0
        if len(a) > len(b):
            a, b = b, a
        dot = sum(v * b.get(t, 0.0) for t, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def add_to_centroid(centroid: dict, vec: dict, n: int) -> dict:
        # Online mean
        for t, v in vec.items():
            centroid[t] = centroid.get(t, 0.0) + (v - centroid.get(t, 0.0)) / n
        return centroid

    for did, _ in docs:
        vec = idx.doc_vecs.get(did, {})
        if not vec:
            continue
        # Find best existing cluster
        best_i, best_sim = -1, 0.0
        for i, c in enumerate(clusters):
            sim = cosine(vec, c["centroid"])
            if sim > best_sim:
                best_i, best_sim = i, sim
        if best_i >= 0 and best_sim >= threshold:
            c = clusters[best_i]
            c["ids"].add(did)
            c["centroid"] = add_to_centroid(dict(c["centroid"]), vec, len(c["ids"]))
        else:
            clusters.append({"ids": {did}, "centroid": dict(vec)})

    out: list[list[dict]] = []
    for c in clusters:
        out.append([by_id[i] for i in c["ids"] if i in by_id])
    out.sort(key=lambda g: len(g), reverse=True)
    return out


_EXTRACT_SYSTEM = (
    "You are a memory-consolidation module. Given a cluster of episodic memory "
    "traces (raw event logs), distill ONE durable semantic fact that captures "
    "the recurring pattern across them — a piece of generalizable knowledge "
    "this brain should carry forward. Be terse (≤ 20 words), first person, "
    "and skip filler like 'I noticed that'. If the cluster is incoherent or "
    "trivial, return the empty string."
)


def _extract_semantic_fact(llm: LLM, model: str, cluster: list[dict]) -> Optional[str]:
    if not cluster:
        return None
    lines = "\n".join(
        f"- ({c.get('mem_type','epi')}/{c['kind']}) {c['content'][:240]}"
        for c in cluster[:10]
    )
    prompt = (
        f"Episodic cluster (size={len(cluster)}):\n{lines}\n\n"
        "Return ONLY the distilled semantic fact (one line, ≤20 words, first "
        "person). If nothing durable, return an empty string."
    )
    try:
        text = llm.chat(model, _EXTRACT_SYSTEM, prompt,
                        temperature=0.3, max_tokens=80).strip()
    except Exception as e:  # noqa: BLE001 — failure here is non-fatal
        return None
    text = text.strip().strip('"').strip("'")
    if len(text) < 4:
        return None
    return text[:240]


def consolidate(memory: Memory, llm: Optional[LLM], model: Optional[str],
                n_episodes: int = 60,
                min_cluster_size: int = 3,
                max_new_facts: int = 3,
                cluster_threshold: float = 0.18,
                log=None) -> dict[str, Any]:
    """One consolidation pass. Returns stats."""
    log = log or (lambda _m: None)
    rows = _fetch_candidates(memory, n_episodes)
    if len(rows) < min_cluster_size:
        log(f"consolidator: only {len(rows)} candidate(s); skipping")
        return {"candidates": len(rows), "clusters": 0,
                "facts_written": 0, "rows_tagged": 0}

    clusters = _cluster(rows, threshold=cluster_threshold)
    eligible = [c for c in clusters if len(c) >= min_cluster_size][:max_new_facts]
    log(f"consolidator: {len(rows)} candidate(s), "
        f"{len(clusters)} cluster(s), {len(eligible)} eligible")

    facts_written = 0
    rows_tagged = 0
    for cluster in eligible:
        if llm is None or model is None:
            # offline / dry-run mode: just tag the cluster as consolidated
            # so it isn't reprocessed every pass
            fact = None
        else:
            fact = _extract_semantic_fact(llm, model, cluster)
        if fact:
            # The semantic fact carries an aggregate salience; tag with the
            # cluster size so downstream callers can weigh it.
            memory.store(
                task="consolidation",
                kind="distilled",
                content=fact,
                salience=0.78,
                mem_type=SEMANTIC,
                tags=["consolidated", f"cluster:{len(cluster)}"],
            )
            facts_written += 1
            log(f"  + semantic: {fact[:120]}")
        # Always tag the cluster's rows so we don't reprocess them
        ids = [int(r["id"]) for r in cluster]
        _tag_rows_consolidated(memory, ids)
        rows_tagged += len(ids)

    return {"candidates": len(rows), "clusters": len(clusters),
            "facts_written": facts_written, "rows_tagged": rows_tagged}


def _tag_rows_consolidated(memory: Memory, ids: list[int]) -> None:
    """Append the `consolidated` tag without clobbering existing tags."""
    if not ids:
        return
    placeholders = ",".join("?" * len(ids))
    rows = memory.conn.execute(
        f"SELECT id, tags FROM episodes WHERE id IN ({placeholders})", ids
    ).fetchall()
    for r in rows:
        tags = [t for t in (r["tags"] or "").split(",") if t]
        if _CONSOLIDATED_TAG in tags:
            continue
        tags.append(_CONSOLIDATED_TAG)
        memory.conn.execute(
            "UPDATE episodes SET tags=? WHERE id=?",
            (",".join(sorted(set(tags))), int(r["id"])),
        )
    memory.conn.commit()


# ── CLI: standalone 'sleep' pass over all accumulated runs ─────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run an offline memory consolidation 'sleep' pass.")
    ap.add_argument("--n", type=int, default=120,
                    help="episode window to scan (default 120)")
    ap.add_argument("--min-cluster", type=int, default=3,
                    help="min rows per cluster to consolidate (default 3)")
    ap.add_argument("--max-facts", type=int, default=8,
                    help="max semantic facts to write this pass (default 8)")
    ap.add_argument("--threshold", type=float, default=0.18,
                    help="cosine similarity threshold for clustering")
    ap.add_argument("--dry-run", action="store_true",
                    help="cluster + tag but skip LLM extraction (free)")
    ap.add_argument("--model", default=None,
                    help="LLM model id (default: cfg.models.reflex)")
    args = ap.parse_args()

    cfg = load_config()
    memory = Memory(cfg.db_path)
    if args.dry_run:
        llm, model = None, None
    else:
        llm = LLM(cfg)
        model = args.model or cfg.models.get("reflex")
    try:
        stats = consolidate(
            memory, llm, model,
            n_episodes=args.n,
            min_cluster_size=args.min_cluster,
            max_new_facts=args.max_facts,
            cluster_threshold=args.threshold,
            log=lambda m: print(m, flush=True),
        )
    finally:
        memory.close()
        if llm is not None:
            llm.close()
    print(f"\nstats: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
