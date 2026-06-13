"""Window survey — the brain's quiet 'rounds'.

The per-frame observer deep-reads the FOCUSED window; this complements it by
taking stock of EVERYTHING that's open. During a lull (you've stepped back —
idle past a threshold but the screensaver isn't on yet), ~every 10 minutes the
brain enumerates every open application and its window titles and records one
"survey" observation. So Recall / the journal / the dashboard can answer
"what did I have open this afternoon?".

Non-disruptive by design: it does NOT tab/cmd-tab or change window focus, and
it takes no screenshots — it reads the window list via `osascript` (System
Events). Window TITLES are the content for most apps ("Reddit – r/claudeopus",
"brain — daemon.py", "Gmail – Inbox (3)"), which is exactly the note we want
without raising or rearranging anything. (Per-window pixel reads would need
Quartz/pyobjc — a future tier.)

Needs Accessibility permission for window titles (already granted for hands);
without it the app list still comes through, just title-less.
"""
from __future__ import annotations

import subprocess
import time
from typing import Any, Optional

from .memory import OBSERVATION

# AppleScript: dump "AppName\t::\ttitle1\ttitle2\t..." per non-background app.
_LIST_SCRIPT = r'''
tell application "System Events"
  set out to ""
  repeat with p in (every process whose background only is false)
    set pn to name of p
    set titles to ""
    try
      repeat with w in (windows of p)
        set titles to titles & (name of w) & tab
      end repeat
    end try
    set out to out & pn & tab & "::" & tab & titles & linefeed
  end repeat
  return out
end tell
'''

_MAX_TITLES_PER_APP = 5
_SKIP_APPS = {"finder", "loginwindow", "dock", "controlcenter",
              "notificationcenter", "systemuiserver", "wallpaper"}


def list_windows(*, run=None, timeout: float = 10.0) -> list[dict[str, Any]]:
    """[{app, titles:[...], n}] for every open, non-background app. `run`
    overrides the osascript call for tests. Never raises → [] on failure."""
    run = run or _osascript
    try:
        raw = run(_LIST_SCRIPT, timeout)
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for line in (raw or "").splitlines():
        if "\t::\t" not in line:
            continue
        app, rest = line.split("\t::\t", 1)
        app = app.strip()
        if not app or app.lower() in _SKIP_APPS:
            continue
        titles = [t.strip() for t in rest.split("\t") if t.strip()]
        # de-dup while preserving order
        seen: set = set()
        uniq = [t for t in titles if not (t in seen or seen.add(t))]
        out.append({"app": app, "titles": uniq[:_MAX_TITLES_PER_APP],
                    "n": len(uniq)})
    return out


def _osascript(script: str, timeout: float) -> str:
    r = subprocess.run(["osascript", "-"], input=script, text=True,
                       capture_output=True, timeout=timeout)
    return r.stdout or ""


def render_survey(windows: list[dict[str, Any]]) -> str:
    """Compact one-line note: 'Chrome[3]: Reddit; Gmail; Docs · Code[1]: daemon.py'."""
    parts = []
    for w in windows:
        head = f"{w['app']}[{w['n']}]"
        if w["titles"]:
            head += ": " + "; ".join(t[:60] for t in w["titles"])
        parts.append(head)
    return " · ".join(parts)


def topic_tags_from_windows(windows: list[dict[str, Any]], limit: int = 8) -> list[str]:
    """App-name tags so Recall can find 'when did I have Slack open'."""
    tags = []
    for w in windows:
        a = w["app"].lower().strip()
        if a:
            tags.append(f"topic:{a}")
    return tags[:limit]


def record_survey(memory, windows: list[dict[str, Any]]) -> Optional[str]:
    """Store one survey observation. Tagged app:survey so it stays OUT of the
    per-app usage counts (a survey isn't usage) while remaining queryable;
    agency=passive because the brain did the rounds, not the user."""
    if not windows:
        return None
    content = "[survey] " + render_survey(windows)
    tags = ["app:survey", "agency:passive"] + topic_tags_from_windows(windows)
    memory.store(task="window_survey", kind="activity",
                 content=content[:500], salience=0.5,
                 mem_type=OBSERVATION, tags=tags)
    return content


class WindowSurveyor:
    """Cadence gate + enumerate + record. No LLM, no screenshots, no focus
    changes — safe to run inline on the daemon tick (osascript is fast)."""

    def __init__(self, memory, *, interval_seconds: float = 600.0,
                 lull_seconds: float = 45.0, away_seconds: float = 300.0,
                 time_fn=time.monotonic, list_fn=None):
        self.memory = memory
        self.interval_seconds = float(interval_seconds)
        self.lull_seconds = float(lull_seconds)
        self.away_seconds = float(away_seconds)
        self._time_fn = time_fn
        self._list = list_fn or list_windows
        self._next_due = self._time_fn()   # eligible immediately on first lull
        self.surveys = 0
        self.last: list[dict[str, Any]] = []

    def maybe_survey(self, *, idle: Optional[float],
                     present: Optional[bool]) -> dict[str, Any]:
        """Do the rounds when: present, in a LULL (idle past lull_seconds but
        not away), and the interval has elapsed. Never raises."""
        if present is False:
            return {"surveyed": False, "reason": "away"}
        if idle is None or idle < self.lull_seconds or idle >= self.away_seconds:
            return {"surveyed": False, "reason": "active or away"}
        now = self._time_fn()
        if now < self._next_due:
            return {"surveyed": False, "reason": "not due"}
        self._next_due = now + self.interval_seconds
        try:
            windows = self._list()
        except Exception as e:  # noqa: BLE001
            return {"surveyed": False, "reason": f"{type(e).__name__}: {e}"}
        if not windows:
            return {"surveyed": False, "reason": "no windows"}
        record_survey(self.memory, windows)
        self.surveys += 1
        self.last = windows
        return {"surveyed": True, "n_apps": len(windows),
                "summary": render_survey(windows)[:200]}


# ── CLI: one-shot survey now ─────────────────────────────────────────────────
def main() -> int:
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    from brain.config import load_config
    from brain.embeddings import make_backend
    from brain.llm import LLM
    from brain.memory import Memory

    cfg = load_config()
    llm = LLM(cfg)
    memory = Memory(cfg.db_path, backend=make_backend(cfg, llm=llm))
    try:
        windows = list_windows()
        if not windows:
            print("no open windows found (grant Accessibility for titles)")
            return 0
        record_survey(memory, windows)
        print(f"🗂  {len(windows)} apps open:\n  " + render_survey(windows))
        print("\n(recorded as a survey observation)")
    finally:
        memory.close()
        llm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
