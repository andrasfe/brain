"""Recall — ask the brain about your own activity.

The observation stream (what you read, did, and what ran unattended) is a
private record nothing ever queried. Recall closes the loop:

    python3 -m brain.recall "what was that Reddit thread about guardrails?"
    python3 -m brain.recall --days 1 "what did I work on today?"
    python3 -m brain.recall --app slack --no-llm ""        # raw matches only

Search is deliberately self-contained (keyword/topic overlap × salience with a
recency boost, plus app/agency/time filters) rather than reusing
`retrieve_semantic`: observation rows cache DINOv2 *image* vectors in the
`embedding` column when visual capture is on, which live in a different space
than any text query — comparing them would be wrong. Text ranking over the
filtered candidate set is dependable and fast at this scale.

Synthesis (optional): the executive model reads the matched, timestamped
observations and answers the question in 2-5 sentences, citing times. Local
only; read-only over the existing memory DB.
"""
from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any, Optional

from .memory import OBSERVATION

_WORD = re.compile(r"[a-z0-9#@]+")

# Half-life of the recency boost: a week-old observation scores half the
# recency bonus of a fresh one. Relevance still dominates (recency is a bonus).
_RECENCY_HALF_LIFE_S = 7 * 86400.0


def _terms(text: str) -> set:
    return set(_WORD.findall((text or "").lower()))


def search_observations(memory, query: str, *,
                        since_ts: Optional[float] = None,
                        until_ts: Optional[float] = None,
                        app: Optional[str] = None,
                        agency: Optional[str] = None,
                        k: int = 25,
                        now: Optional[float] = None) -> list[dict[str, Any]]:
    """Rank observation rows for `query` with hard filters. Empty query →
    pure time-ordered browse of the filtered window (newest first)."""
    now = time.time() if now is None else now
    sql = "SELECT * FROM episodes WHERE mem_type=?"
    params: list[Any] = [OBSERVATION]
    if since_ts is not None:
        sql += " AND ts >= ?"
        params.append(float(since_ts))
    if until_ts is not None:
        sql += " AND ts <= ?"
        params.append(float(until_ts))
    if app:
        sql += " AND tags LIKE ?"
        params.append(f"%app:{app.strip().lower()}%")
    if agency:
        sql += " AND tags LIKE ?"
        params.append(f"%agency:{agency.strip().lower()}%")
    sql += " ORDER BY ts DESC LIMIT 2000"
    rows = memory.conn.execute(sql, params).fetchall()

    q = _terms(query)
    scored: list[tuple[float, dict[str, Any]]] = []
    for r in rows:
        d = {"id": int(r["id"]), "ts": float(r["ts"]),
             "content": r["content"], "tags": r["tags"] or "",
             "salience": float(r["salience"])}
        if not q:
            scored.append((d["ts"], d))      # browse mode: newest first
            continue
        overlap = len(q & _terms(d["content"] + " " + d["tags"]))
        if overlap == 0:
            continue
        age = max(0.0, now - d["ts"])
        recency = 0.5 ** (age / _RECENCY_HALF_LIFE_S)
        score = overlap * (0.5 + d["salience"]) + recency
        d["score"] = round(score, 3)
        scored.append((score, d))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [d for _, d in scored[:k]]


def render_matches(matches: list[dict[str, Any]]) -> str:
    lines = []
    for m in matches:
        when = datetime.fromtimestamp(m["ts"]).strftime("%a %Y-%m-%d %H:%M")
        agency = "auto" if "agency:passive" in (m.get("tags") or "") else "you"
        lines.append(f"- [{when}] ({agency}) {m['content'][:220]}")
    return "\n".join(lines)


def synthesize(llm, model: str, question: str,
               matches: list[dict[str, Any]]) -> str:
    """One executive-model pass over the matched observations → a direct,
    timestamped answer. chat_json keeps reasoning models from leaking CoT."""
    if not matches:
        return "I have no matching observations for that."
    prompt = (
        "You are answering a question about the USER'S OWN recorded computer "
        "activity. Below are timestamped screen observations ((you) = the user "
        "acted; (auto) = the screen changed on its own, e.g. a script).\n\n"
        f"OBSERVATIONS:\n{render_matches(matches)}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer in 2-5 sentences, first person to the user ('you read…'), "
        "citing day/time when it helps. If the observations don't actually "
        "answer the question, say so plainly. Return JSON exactly like: "
        '{"answer": "..."}'
    )
    out = llm.chat_json(model, "You are a precise personal-memory assistant.",
                        prompt, temperature=0.2, max_tokens=900)
    ans = (out.get("answer") or "").strip()
    return ans or "(no answer produced)"


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> int:
    import argparse
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    from brain.config import load_config
    from brain.embeddings import make_backend
    from brain.llm import LLM
    from brain.memory import Memory

    ap = argparse.ArgumentParser(
        description="Ask the brain about your own activity.")
    ap.add_argument("question", nargs="?", default="",
                    help="what to recall (empty = browse the window)")
    ap.add_argument("--days", type=float, default=7.0,
                    help="look-back window in days (default 7)")
    ap.add_argument("--app", default=None, help="filter by app substring tag")
    ap.add_argument("--agency", default=None, choices=[None, "active", "passive"],
                    help="filter: active (you) / passive (autonomous)")
    ap.add_argument("--k", type=int, default=25, help="max matches")
    ap.add_argument("--no-llm", action="store_true",
                    help="print raw matches, skip synthesis")
    args = ap.parse_args()

    cfg = load_config()
    llm = LLM(cfg)
    memory = Memory(cfg.db_path, backend=make_backend(cfg, llm=llm))
    try:
        since = time.time() - args.days * 86400.0
        matches = search_observations(
            memory, args.question, since_ts=since, app=args.app,
            agency=args.agency, k=args.k)
        if not matches:
            print("No matching observations in the window.")
            return 0
        print(f"── {len(matches)} matching observation(s) "
              f"(last {args.days:g} days) " + "─" * 20)
        print(render_matches(matches))
        if args.question and not args.no_llm:
            print("\n── recall " + "─" * 44)
            print(synthesize(llm, cfg.models.get("executive", ""),
                             args.question, matches))
    finally:
        memory.close()
        llm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
