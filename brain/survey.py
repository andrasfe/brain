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

import os
import shutil
import subprocess
import time
from pathlib import Path
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


# ── Tier-2: per-window PIXEL reads (Quartz) ──────────────────────────────────
def quartz_available() -> bool:
    try:
        import Quartz  # noqa: F401
        return True
    except Exception:
        return False


def list_windows_quartz(*, min_w: int = 200, min_h: int = 150) -> list[dict[str, Any]]:
    """On-screen normal windows in front-to-back order, each with a CGWindowID
    we can screenshot WITHOUT raising it: [{id, app, title, w, h}]. Skips
    menubar/dock (layer != 0) and tiny popovers. [] when Quartz is absent."""
    try:
        import Quartz
        wl = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID)
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for w in wl or []:
        try:
            if int(w.get("kCGWindowLayer", 1)) != 0:
                continue
            b = w.get("kCGWindowBounds", {}) or {}
            wd, ht = int(b.get("Width", 0)), int(b.get("Height", 0))
            if wd < min_w or ht < min_h:
                continue
            out.append({"id": int(w.get("kCGWindowNumber", 0)),
                        "app": str(w.get("kCGWindowOwnerName", "") or ""),
                        "title": str(w.get("kCGWindowName", "") or ""),
                        "w": wd, "h": ht})
        except Exception:
            continue
    return out


def capture_window(window_id: int, dest: str, *, max_width: int = 1280,
                   timeout: float = 10.0) -> bool:
    """Screenshot ONE window by id (occluded content included; no raise, no
    shadow), downscaled. Returns success."""
    sc = shutil.which("screencapture")
    if not sc:
        return False
    try:
        subprocess.run([sc, "-l", str(window_id), "-x", "-o", "-t", "jpg", dest],
                       capture_output=True, timeout=timeout)
        if not os.path.exists(dest):
            return False
        sips = shutil.which("sips")
        if sips:
            subprocess.run([sips, "--resampleWidth", str(max_width), dest],
                           capture_output=True, timeout=timeout)
        return True
    except Exception:
        return False


def window_read_instruction(app: str, title: str) -> str:
    return (
        f"This is a screenshot of a SINGLE application window — app '{app or '?'}'"
        + (f", window title '{title[:120]}'" if title else "") + ".\n"
        "In ONE sentence, describe WHAT CONTENT this window shows — the topic/"
        "subject, key headlines, threads, or what the user has in it. Do not "
        "describe other windows. Never transcribe passwords, secrets, tokens, "
        "or full private messages. No preamble.")


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
                 time_fn=time.monotonic, list_fn=None,
                 job_queue=None, pixel_reads: bool = False,
                 max_window_reads: int = 4, exclude_apps=None,
                 spool_dir=None, ttl_seconds: float = 600.0,
                 quartz_list_fn=None, capture_fn=None,
                 require_absence: bool = False, presence_fn=None):
        self.memory = memory
        self.interval_seconds = float(interval_seconds)
        self.lull_seconds = float(lull_seconds)
        self.away_seconds = float(away_seconds)
        self._time_fn = time_fn
        self._list = list_fn or list_windows
        # Tier-2 (Quartz) per-window pixel reads — deferred via the queue.
        self.job_queue = job_queue
        self.pixel_reads = bool(pixel_reads)
        self.max_window_reads = int(max_window_reads)
        self._exclude = {a.lower() for a in (exclude_apps or [])}
        self._spool_dir = Path(spool_dir) if spool_dir else None
        self.ttl_seconds = float(ttl_seconds)
        self._qlist = quartz_list_fn or list_windows_quartz
        self._capture = capture_fn or capture_window
        # When set, the sweep runs only when the WEBCAM confirms nobody is
        # sitting — the empty-chair signal, stronger than keyboard idle.
        self.require_absence = bool(require_absence)
        self._presence_fn = presence_fn
        self._next_due = self._time_fn()   # eligible immediately on first lull
        self.surveys = 0
        self.window_reads_queued = 0
        self.last: list[dict[str, Any]] = []

    def _maybe_enqueue_window_reads(self) -> int:
        """Capture the largest BACKGROUND windows (the foreground one is already
        deep-read by the observer) and enqueue per-window VLM reads. Skips
        excluded/sensitive apps. Returns how many were queued."""
        if not (self.pixel_reads and self.job_queue is not None
                and self._spool_dir is not None and quartz_available()):
            return 0
        wins = self._qlist()
        if len(wins) <= 1:
            return 0
        # wins[0] is frontmost (observer covers it); rank the rest by area.
        bg = [w for w in wins[1:] if w["app"].lower() not in self._exclude]
        bg.sort(key=lambda w: w["w"] * w["h"], reverse=True)
        queued = 0
        try:
            self._spool_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return 0
        for w in bg[:self.max_window_reads]:
            dest = str(self._spool_dir / f"win_{w['id']}_{int(time.time())}.spool.jpg")
            if not self._capture(w["id"], dest):
                continue
            try:
                self.job_queue.enqueue(
                    "window_read",
                    {"spool": dest, "app": w["app"], "title": w["title"]},
                    priority=3, dedup_key=f"win:{w['app'].lower()}",
                    not_after_ts=time.time() + self.ttl_seconds)
                queued += 1
            except Exception:
                try:
                    os.remove(dest)
                except OSError:
                    pass
        self.window_reads_queued += queued
        return queued

    def maybe_survey(self, *, idle: Optional[float],
                     present: Optional[bool]) -> dict[str, Any]:
        """Do the rounds when: present, in a LULL (idle past lull_seconds but
        not away), and the interval has elapsed. Never raises."""
        # Cheap pre-filter: only consider sweeping after a keyboard lull (don't
        # check the camera while you're actively typing). When require_absence,
        # we DON'T require daemon-presence (the webcam decides); without it, the
        # old behavior (present + lull, below away) applies.
        if idle is None or idle < self.lull_seconds:
            return {"surveyed": False, "reason": "active"}
        if not self.require_absence:
            if present is False or idle >= self.away_seconds:
                return {"surveyed": False, "reason": "away/absent (no camera gate)"}
        now = self._time_fn()
        if now < self._next_due:
            return {"surveyed": False, "reason": "not due"}
        # Set next_due NOW so the (costly) presence check happens at most once
        # per interval regardless of outcome — no per-tick camera blinking.
        self._next_due = now + self.interval_seconds

        # Empty-chair gate: sweep only when nobody's sitting. Webcam is the
        # primary signal; fall back to the keyboard away-threshold when the
        # camera can't decide.
        if self.require_absence:
            pres = "unknown"
            if self._presence_fn is not None:
                try:
                    pres = self._presence_fn()
                except Exception:
                    pres = "unknown"
            if pres == "present":
                return {"surveyed": False, "reason": "person present (webcam)"}
            if pres == "unknown" and not (present is False or idle >= self.away_seconds):
                return {"surveyed": False, "reason": "presence unknown, user likely here"}
        try:
            windows = self._list()
        except Exception as e:  # noqa: BLE001
            return {"surveyed": False, "reason": f"{type(e).__name__}: {e}"}
        if not windows:
            return {"surveyed": False, "reason": "no windows"}
        record_survey(self.memory, windows)
        self.surveys += 1
        self.last = windows
        queued = self._maybe_enqueue_window_reads()
        return {"surveyed": True, "n_apps": len(windows),
                "window_reads": queued,
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
