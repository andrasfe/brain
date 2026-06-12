"""Journalist (NREM) — the nightly digest of your day.

While you sleep, the brain turns the day's observation stream into a short
journal entry: what you actually did, what you read about, and what ran
unattended (the agency tags pay off here). The digest is stored as a semantic
memory (so Recall can answer "what did I do Tuesday?") and appended to a local
markdown journal you can read over coffee:

    ~/brain/journal/YYYY-MM-DD.md          (git-ignored, local only)

Policy: digest COMPLETED days (yesterday and earlier) — a day digested mid-way
would be misleadingly partial. Idempotent via a `journal:YYYY-MM-DD` tag, so
interrupted sleep retries safely and later bouts skip. One LLM call per digest;
the daemon runs at most one digest per NREM bout. Manual/partial runs:

    python3 -m brain.sleep.journalist                # most recent undigested day
    python3 -m brain.sleep.journalist --date today --force
"""
from __future__ import annotations

import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from ..memory import OBSERVATION, SEMANTIC

_MAX_OBS_LINES = 40      # prompt budget: sampled observation lines
_MIN_OBS = 5             # don't journal a day with almost nothing in it


def _day_bounds(date_str: str) -> tuple[float, float]:
    d = datetime.strptime(date_str, "%Y-%m-%d")
    start = d.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.timestamp(), (start + timedelta(days=1)).timestamp()


def _date_of(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


class Journalist:
    name = "journalist"

    def __init__(self, journal_dir: Optional[Path] = None,
                 lookback_days: int = 3):
        self.journal_dir = Path(journal_dir) if journal_dir else None
        self.lookback_days = int(lookback_days)

    # ── selection ───────────────────────────────────────────────────────────
    def already_digested(self, memory, date_str: str) -> bool:
        row = memory.conn.execute(
            "SELECT id FROM episodes WHERE tags LIKE ? LIMIT 1",
            (f"%journal:{date_str}%",)).fetchone()
        return row is not None

    def next_undigested_day(self, memory, now: Optional[float] = None
                            ) -> Optional[str]:
        """Most recent COMPLETED day with observations and no journal yet."""
        now = time.time() if now is None else now
        for back in range(1, self.lookback_days + 1):
            date_str = _date_of(now - back * 86400.0)
            if self.already_digested(memory, date_str):
                continue
            lo, hi = _day_bounds(date_str)
            row = memory.conn.execute(
                "SELECT COUNT(*) AS n FROM episodes WHERE mem_type=? "
                "AND ts >= ? AND ts < ?", (OBSERVATION, lo, hi)).fetchone()
            if int(row["n"]) >= _MIN_OBS:
                return date_str
        return None

    # ── digest one day ──────────────────────────────────────────────────────
    def digest_day(self, memory, llm, model: str, date_str: str, *,
                   force: bool = False, log=None) -> dict[str, Any]:
        log = log or (lambda _m: None)
        if not force and self.already_digested(memory, date_str):
            return {"written": False, "reason": "already digested",
                    "date": date_str}
        lo, hi = _day_bounds(date_str)
        rows = memory.conn.execute(
            "SELECT ts, content, tags, salience FROM episodes "
            "WHERE mem_type=? AND ts >= ? AND ts < ? ORDER BY ts",
            (OBSERVATION, lo, hi)).fetchall()
        if len(rows) < (_MIN_OBS if not force else 1):
            return {"written": False, "reason": "too few observations",
                    "date": date_str, "rows": len(rows)}

        stats = self._aggregate(rows)
        sampled = self._sample(rows, _MAX_OBS_LINES)
        obs_lines = []
        for r in sampled:
            when = datetime.fromtimestamp(float(r["ts"])).strftime("%H:%M")
            who = "auto" if "agency:passive" in (r["tags"] or "") else "you"
            obs_lines.append(f"- {when} ({who}) {r['content'][:120]}")

        prompt = (
            f"Write the user's private journal entry for {date_str} from their "
            "screen-activity record. ((you) = the user acted; (auto) = the "
            "screen changed on its own, e.g. a script or feed).\n\n"
            f"DAY STATS: {stats['n']} observations, "
            f"{stats['pct_active']}% user-driven; "
            f"top apps: {', '.join(stats['top_apps']) or 'n/a'}; "
            f"top topics: {', '.join(stats['top_topics']) or 'n/a'}; "
            f"active span {stats['first']}–{stats['last']}.\n\n"
            "OBSERVATIONS (sampled, chronological):\n"
            + "\n".join(obs_lines) + "\n\n"
            "Write 4-8 sentences, second person ('You spent the morning…'): "
            "what they worked on, what they read about (name specific topics/"
            "threads), and anything that ran unattended. Honest, concrete, no "
            "filler, no bullet points. Return JSON with one key: "
            '{"digest": "<the full 4-8 sentence journal entry text>"}'
        )
        out = llm.chat_json(model, "You write precise, warm daily journals.",
                            prompt, temperature=0.4, max_tokens=2200)
        digest = (out.get("digest") or "").strip()
        # Guard against template-echo / degenerate outputs ("...", "<the ...>").
        if len(digest) < 40 or digest.startswith("<"):
            return {"written": False, "reason": "degenerate digest",
                    "date": date_str, "digest": digest}

        memory.store(
            task="journal", kind="digest",
            content=f"Journal {date_str}: {digest}"[:1500],
            salience=0.85, mem_type=SEMANTIC,
            tags=["journal", f"journal:{date_str}"])
        path = self._write_markdown(date_str, digest, stats)
        log(f"  📔 journal {date_str}: {digest[:100]}")
        return {"written": True, "date": date_str, "rows": len(rows),
                "path": str(path) if path else None, "digest": digest}

    # ── helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _aggregate(rows) -> dict[str, Any]:
        apps: Counter = Counter()
        topics: Counter = Counter()
        active = 0
        for r in rows:
            tags = (r["tags"] or "")
            for t in tags.split(","):
                t = t.strip()
                if t.startswith("app:"):
                    apps[t[4:]] += 1
                elif t.startswith("topic:"):
                    topics[t[6:]] += 1
            if "agency:passive" not in tags:
                active += 1
        n = len(rows)
        return {
            "n": n,
            "pct_active": int(round(100.0 * active / n)) if n else 0,
            "top_apps": [a for a, _ in apps.most_common(5)],
            "top_topics": [t for t, _ in topics.most_common(8)],
            "first": datetime.fromtimestamp(float(rows[0]["ts"])).strftime("%H:%M"),
            "last": datetime.fromtimestamp(float(rows[-1]["ts"])).strftime("%H:%M"),
        }

    @staticmethod
    def _sample(rows, limit: int):
        """Keep chronology while honoring the prompt budget: an even stride
        across the day, then the highest-salience rows fill remaining slots."""
        if len(rows) <= limit:
            return list(rows)
        stride = len(rows) / float(limit)
        picked = {int(i * stride) for i in range(limit)}
        return [rows[i] for i in sorted(picked)]

    def _write_markdown(self, date_str: str, digest: str,
                        stats: dict[str, Any]) -> Optional[Path]:
        if self.journal_dir is None:
            return None
        try:
            self.journal_dir.mkdir(parents=True, exist_ok=True)
            path = self.journal_dir / f"{date_str}.md"
            body = (
                f"# {date_str}\n\n{digest}\n\n---\n"
                f"*{stats['n']} observations · {stats['pct_active']}% you · "
                f"apps: {', '.join(stats['top_apps']) or 'n/a'} · "
                f"topics: {', '.join(stats['top_topics']) or 'n/a'} · "
                f"{stats['first']}–{stats['last']}*\n")
            path.write_text(body)
            return path
        except OSError:
            return None

    # ── daemon entry: at most ONE digest per call ───────────────────────────
    def run(self, memory, llm, model: str, *, log=None) -> dict[str, Any]:
        date_str = self.next_undigested_day(memory)
        if date_str is None:
            return {"written": False, "reason": "nothing to digest"}
        return self.digest_day(memory, llm, model, date_str, log=log)


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> int:
    import argparse
    import sys
    repo = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(repo))
    from brain.config import load_config
    from brain.embeddings import make_backend
    from brain.llm import LLM
    from brain.memory import Memory

    ap = argparse.ArgumentParser(description="Write a daily activity digest.")
    ap.add_argument("--date", default=None,
                    help="YYYY-MM-DD | today | yesterday (default: most "
                         "recent undigested completed day)")
    ap.add_argument("--force", action="store_true",
                    help="re-digest even if a journal exists (or partial day)")
    args = ap.parse_args()

    cfg = load_config()
    llm = LLM(cfg)
    memory = Memory(cfg.db_path, backend=make_backend(cfg, llm=llm))
    journal_dir = Path(cfg.db_path).parent / "journal"
    j = Journalist(journal_dir=journal_dir)
    try:
        model = cfg.models.get("executive", "")
        if args.date:
            date_str = args.date.lower()
            if date_str == "today":
                date_str = _date_of(time.time())
            elif date_str == "yesterday":
                date_str = _date_of(time.time() - 86400.0)
            out = j.digest_day(memory, llm, model, date_str,
                               force=args.force, log=print)
        else:
            out = j.run(memory, llm, model, log=print)
        print(f"\nresult: {out.get('written')} "
              f"({out.get('reason') or out.get('path')})")
        if out.get("digest"):
            print("\n" + out["digest"])
    finally:
        memory.close()
        llm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
