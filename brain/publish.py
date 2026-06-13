"""KnowledgePublisher — push the brain's human-readable memory to a note app.

Backend-agnostic: it builds one rich markdown page per DAY (journal + activity
stats + mood/wellness + windows-open) and upserts it through a pluggable
publisher backend. The Joplin backend is the first; another note app is a new
backend, not a rewrite (the brain has changed targets a few times).

Non-destructive: reads the SQLite stores read-only; SQLite remains the
operational store. The daily page is keyed `daily:YYYY-MM-DD`, so re-syncing
updates the same page (it fills in through the day, finalizes when the nightly
journal lands).
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Optional

from . import webui  # reuse the dashboard aggregates (DRY)


def _date_of(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def build_daily_markdown(conn, date_str: str, *, now: Optional[float] = None
                         ) -> Optional[str]:
    """One day's page. None when the day has no signal at all."""
    now = time.time() if now is None else now
    d = datetime.strptime(date_str, "%Y-%m-%d")
    lo = d.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    hi = lo + 86400.0
    span = max(0.0, min(hi, now) - lo)

    act = webui.activity_summary(conn, lo, now=min(hi, now))
    wl = [w for w in webui.wellness_series(conn, lo) if w["ts"] < hi]
    journal = conn.execute(
        "SELECT content FROM episodes WHERE mem_type='semantic' "
        "AND tags LIKE ? ORDER BY ts DESC LIMIT 1",
        (f"%journal:{date_str}%",)).fetchone()
    surveys = conn.execute(
        "SELECT ts, content FROM episodes WHERE mem_type='observation' "
        "AND tags LIKE '%app:survey%' AND ts >= ? AND ts < ? "
        "ORDER BY ts DESC LIMIT 3", (lo, hi)).fetchall()

    if not (journal or act["active_obs"] or wl or surveys):
        return None

    out = [f"# {date_str}\n"]
    if journal:
        text = journal["content"]
        text = text.split(":", 1)[1].strip() if text.startswith("Journal ") else text
        out.append("## Journal\n\n" + text + "\n")
    out.append("## Activity\n")
    out.append(f"- ~**{act['hours_active']}h** actively at the computer")
    out.append(f"- {act['active_obs']} your actions · {act['passive_obs']} autonomous")
    if act["top_apps"]:
        out.append("- top apps: " + ", ".join(
            f"{a['app']} ({a['n']})" for a in act["top_apps"]))
    out.append("")
    if wl:
        moods = [w["mood"] for w in wl if w.get("mood")]
        fat = [w["fatigue"] for w in wl if w.get("fatigue") is not None]
        out.append("## Wellness\n")
        out.append(f"- {len(wl)} self-checks; moods: {', '.join(moods) or 'n/a'}")
        if fat:
            out.append(f"- avg fatigue {round(sum(fat)/len(fat), 2)} "
                       f"(latest {fat[-1]})")
        if wl[-1].get("summary"):
            out.append(f"- latest: {wl[-1]['summary']}")
        out.append("")
    if surveys:
        out.append("## Windows open (sampled)\n")
        for s in surveys:
            when = datetime.fromtimestamp(float(s["ts"])).strftime("%H:%M")
            body = (s["content"] or "").replace("[survey]", "").strip()
            out.append(f"- **{when}** {body[:300]}")
        out.append("")
    out.append(f"\n*— published by brain · {round(span/3600.0, 1)}h of signal*")
    return "\n".join(out)


class KnowledgePublisher:
    """Maps the brain's memory → daily notes via a backend with
    `ensure_notebook(title, parent_id) -> id` and
    `upsert_note(parent_id, title, body, key) -> id`."""

    def __init__(self, backend, *, root_notebook: str = "Brain",
                 daily_notebook: str = "Daily"):
        self.backend = backend
        self.root_notebook = root_notebook
        self.daily_notebook = daily_notebook
        self._daily_parent: Optional[str] = None

    def _daily_folder(self) -> Optional[str]:
        if self._daily_parent is not None:
            return self._daily_parent
        root = self.backend.ensure_notebook(self.root_notebook)
        if root is None:
            return None
        self._daily_parent = self.backend.ensure_notebook(self.daily_notebook, root)
        return self._daily_parent

    def sync(self, memory, *, days_back: int = 2,
             now: Optional[float] = None) -> dict[str, Any]:
        """Upsert the daily pages for the last `days_back` days (inclusive of
        today). Returns counts. Best-effort; never raises."""
        now = time.time() if now is None else now
        parent = self._daily_folder()
        if parent is None:
            return {"published": 0, "reason": "no notebook"}
        published = 0
        for back in range(days_back):
            date_str = _date_of(now - back * 86400.0)
            try:
                md = build_daily_markdown(memory.conn, date_str, now=now)
            except Exception:
                md = None
            if not md:
                continue
            nid = self.backend.upsert_note(parent, f"Brain — {date_str}", md,
                                           f"daily:{date_str}")
            if nid:
                published += 1
        return {"published": published}


# ── CLI: one-shot publish ────────────────────────────────────────────────────
def main() -> int:
    import argparse
    import os
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    from brain.config import load_config
    from brain.embeddings import make_backend
    from brain.joplin import JoplinClient
    from brain.llm import LLM
    from brain.memory import Memory

    ap = argparse.ArgumentParser(description="Publish brain knowledge to Joplin.")
    ap.add_argument("--days", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config()
    jcfg = cfg.raw.get("joplin") or {}
    token = os.environ.get("JOPLIN_TOKEN", "") or str(jcfg.get("token", ""))
    client = JoplinClient(str(jcfg.get("base_url", "http://localhost:41184")),
                          token)
    if not client.ping():
        print("Joplin Data API not reachable on " + client.base_url +
              "\n  → open Joplin desktop → Settings → Web Clipper → enable, copy "
              "the token into JOPLIN_TOKEN (or config joplin.token).")
        return 1
    llm = LLM(cfg)
    memory = Memory(cfg.db_path, backend=make_backend(cfg, llm=llm))
    try:
        out = KnowledgePublisher(client).sync(memory, days_back=args.days)
        print(f"published {out['published']} daily page(s) to Joplin")
    finally:
        memory.close()
        llm.close()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
