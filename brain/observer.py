"""ScreenObserver — the brain passively watches the user's screen to learn.

Each capture: screenshot (afferent eyes, read-only) → local vision-model
description → text embedding → stored as an `observation` memory row →
**raw pixels deleted immediately**. The activity stream is therefore made of
embeddings + short descriptions, never a hoard of PNGs (that's the main answer
to "a bazillion screenshots": we don't keep them).

PRIVACY IS THE POINT — this stream is the most sensitive data the brain
touches, so the design is privacy-first and refuses to run when it can't keep
the data local:

  * **Local-only enforcement.** Both the screenshot description (vision LLM)
    and the embedding go through the brain's configured endpoint. If that
    endpoint is REMOTE (not localhost), capture is refused — screen contents
    must never leave the machine. A local embedding backend
    (sentence_transformers / tfidf) also counts as local.
  * **App exclusion list.** If the frontmost app is on the exclusion list
    (password managers, banking, private chat …), nothing is captured at all.
  * **Pixel-drop.** The PNG is deleted right after it's described/embedded.
  * **Toggle.** `capture.enabled` (config) + `pause()/resume()` at runtime.

Everything stays on disk in the user's home; nothing is uploaded.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import time
from typing import Any, Optional
from urllib.parse import urlparse

from .memory import OBSERVATION

# Apps whose contents are sensitive by default — never captured.
_DEFAULT_EXCLUDE = [
    "1password", "bitwarden", "keychain access", "keychain",
    "banking", "bank", "messages", "signal", "whatsapp", "telegram",
    "wallet", "authy", "lastpass", "dashlane",
]

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}

# Reasoning-model preamble lines we never want in an activity label.
_PREAMBLE_RE = re.compile(
    r"^\s*(\d+[.)]\s|[-*]\s|the user (wants|is asking|is trying)|"
    r"i (need|should|will|see)\b|let me\b|okay,?\b|first,?\b|"
    r"based on (the|this)|to (answer|describe)|thinking|analysis|"
    r"here('s| is)\b|step \d)",
    re.IGNORECASE,
)


_NICETY_RE = re.compile(
    r"^(the user (is |appears to be |seems to be )|the screen shows |"
    r"this (is |screen )|it (looks like|appears) )",
    re.IGNORECASE,
)


# Strip a leading label like "**Final Polish:**", "Answer:", "Final:" that
# reasoning models prepend to their conclusion line.
_LABEL_RE = re.compile(
    r"^(final polish|final answer|final|answer|conclusion|output|response|"
    r"result|tl;?dr|summary)\s*:?\s*",
    re.IGNORECASE,
)


def _strip_markup(line: str) -> str:
    """Remove markdown emphasis + a leading label prefix + surrounding quotes."""
    line = line.replace("**", "").replace("__", "").strip().strip('"').strip("'")
    line = _LABEL_RE.sub("", line).strip().strip('"').strip("'")
    return line


def clean_vision_text(text: str) -> str:
    """Reduce a (possibly reasoning-laden) vision-model reply to one clean
    activity label. Reasoning models put the actual answer LAST (after their
    step-by-step), so we take the last non-preamble line, strip list markers
    and 'the user is…' niceties."""
    if not text:
        return ""
    text = text.strip().strip('"').strip("'")
    lines = []
    for raw in text.split("\n"):
        line = re.sub(r"^\s*(\d+[.)]|[-*+])\s+", "", raw).strip().strip('"').strip("'")
        line = _strip_markup(line)
        if line:
            lines.append(line)
    # Prefer the LAST substantive non-preamble line (the conclusion).
    for line in reversed(lines):
        if len(line) >= 8 and not _PREAMBLE_RE.match(line):
            line = _NICETY_RE.sub("", _strip_markup(line)).strip()
            return (line[0].upper() + line[1:])[:200] if line else line
    # Fallback: sentence-split the whole blob, last non-preamble sentence.
    for c in reversed(re.split(r"(?<=[.!?])\s+", " ".join(lines))):
        c = _strip_markup(c.strip())
        if len(c) >= 8 and not _PREAMBLE_RE.match(c):
            return c[:200]
    return (" ".join(lines))[:200]


def endpoint_is_local(base_url: str) -> bool:
    try:
        host = urlparse(base_url).hostname or ""
    except Exception:
        return False
    return host.lower() in _LOCAL_HOSTS


def privacy_ok(cfg) -> "tuple[bool, str]":
    """Return (ok, reason). Capture is only allowed when screen content can
    stay on this machine: a local LLM/embedding endpoint, or a fully-local
    embedding backend."""
    backend_name = str((cfg.memory or {}).get("embedding_backend", "")).lower()
    if backend_name in ("sentence_transformers", "tfidf"):
        # embeddings computed in-process; but the vision *description* still
        # uses the LLM endpoint — that must be local too.
        if endpoint_is_local(cfg.base_url):
            return (True, "")
        return (False, f"vision LLM endpoint is remote ({cfg.base_url}); "
                       "screen capture refuses to send pixels off-machine")
    # openrouter-style backend: data goes to cfg.base_url for BOTH embed + describe
    if endpoint_is_local(cfg.base_url):
        return (True, "")
    return (False, f"endpoint is remote ({cfg.base_url}); screen capture "
                   "refuses to send pixels/embeddings off-machine. Use a local "
                   "provider (lmstudio/ollama) or a local embedding backend.")


# Boilerplate words the VLM emits that aren't useful topic tags.
_TOPIC_STOP = {
    "reading", "browsing", "viewing", "looking", "scrolling", "using", "doing",
    "user", "screen", "page", "window", "app", "application", "website", "site",
    "content", "currently", "appears", "shows", "showing", "display", "displays",
    "with", "that", "this", "what", "they", "their", "there", "from", "into",
    "some", "post", "posts", "feed", "view", "open", "click", "while", "about",
    "main", "text", "list", "menu", "left", "right", "top", "bottom",
}


def topic_tags(description: str, limit: int = 6) -> list[str]:
    """Extract retrievable topic tags from a content description — #hashtags,
    @handles, and salient words — so memory can be searched by SUBJECT, not just
    by which app was focused. Deterministic, no LLM."""
    if not description:
        return []
    out: list[str] = []
    seen: set[str] = set()

    def _add(key: str) -> None:
        k = key.lower().strip()
        if len(k) < 3 or k in seen or k in _TOPIC_STOP:
            return
        seen.add(k)
        out.append(f"topic:{k}")

    for h in re.findall(r"[#@](\w{2,})", description):
        _add(h)
    for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", description):
        _add(w)
    return out[:limit]


def content_changed(vec, last_vec, app_switched: bool, threshold: float) -> bool:
    """Did the SCREEN CONTENT meaningfully change since the last described
    frame? Uses DINOv2 cosine distance — so scrolling to a new post / opening a
    new thread counts as a change even within the same app (the case app-switch
    detection misses entirely). Falls back to the app-switch signal when no
    visual embedding is available."""
    if app_switched:
        return True
    if vec is None:
        return False
    if last_vec is None:
        return True
    from .screen_model import cosine_distance
    return cosine_distance(vec, last_vec) >= threshold


class ScreenObserver:
    def __init__(self, cfg, llm, memory, embodiment, *,
                 interval_seconds: float = 60.0,
                 min_interval_seconds: float = 8.0,
                 activity_window_seconds: float = 8.0,
                 exclude_apps: Optional[list] = None,
                 vision_model: str = "",
                 vision_model_strong: str = "",
                 change_detect: bool = True,
                 deep_read_on_change: bool = True,
                 content_change_threshold: float = 0.12,
                 visual_embedder=None,
                 time_fn=time.monotonic):
        self.visual_embedder = visual_embedder
        # Strong (executive) vision model: spent on frames where the screen
        # CONTENT changed meaningfully (new post / page / thread), not just on
        # app-switches — so sustained single-app browsing (X, Reddit) still gets
        # deep reads. The fast reflex model handles minor changes. Blank strong
        # falls back to the executive tier (mirrors vision_model → reflex).
        self.vision_model_strong = vision_model_strong or cfg.models.get("executive", "")
        self.deep_read_on_change = deep_read_on_change
        self.content_change_threshold = float(content_change_threshold)
        self._last_described_vec = None   # DINOv2 vec of the last frame we read
        self.cfg = cfg
        self.cfg = cfg
        self.llm = llm
        self.memory = memory
        self.embodiment = embodiment
        self.interval_seconds = interval_seconds          # fallback max interval
        self.min_interval_seconds = min_interval_seconds  # debounce floor
        self.activity_window_seconds = activity_window_seconds
        excludes = list(exclude_apps if exclude_apps is not None else _DEFAULT_EXCLUDE)
        self._exclude = {a.lower() for a in excludes}
        self.vision_model = vision_model or cfg.models.get("reflex", "")
        self.change_detect = change_detect
        self._sips = shutil.which("sips")
        self._time_fn = time_fn
        self._last_capture = -1e9
        self._last_fingerprint: Optional[str] = None
        self._last_app: Optional[str] = None
        self._paused = False
        # Resolve privacy once; refuse if not local.
        self.ok, self.reason = privacy_ok(cfg)
        self.captured = 0
        self.skipped = 0
        self.deduped = 0

    def _frontmost(self) -> Optional[str]:
        """Cheap frontmost-app read (no screenshot) for app-switch detection."""
        be = getattr(self.embodiment, "backend", None)
        if be is None or not hasattr(be, "frontmost_app"):
            return None
        try:
            return be.frontmost_app()
        except Exception:
            return None

    def _fingerprint(self, path: Optional[str]) -> Optional[str]:
        """Perceptual hash of a screenshot: an 8×8 grayscale thumbnail (via the
        built-in `sips`, no deps) hashed. Two ~static screens → same hash. None
        when it can't be computed (so callers don't dedup blindly)."""
        if not path or not self._sips or not os.path.exists(path):
            return None
        thumb = path + ".thumb.png"
        try:
            subprocess.run(
                [self._sips, "-z", "8", "8", "-s", "format", "png",
                 path, "--out", thumb],
                capture_output=True, timeout=5, check=False)
            if not os.path.exists(thumb):
                return None
            data = open(thumb, "rb").read()
            return hashlib.sha1(data).hexdigest()
        except Exception:
            return None
        finally:
            try:
                os.remove(thumb)
            except OSError:
                pass

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def _excluded(self, app: Optional[str]) -> bool:
        if not app:
            return False
        a = app.lower()
        return any(x in a for x in self._exclude)

    def maybe_capture(self, *, force: bool = False,
                      idle: Optional[float] = None) -> dict[str, Any]:
        """Capture one observation if warranted + allowed. Adaptive cadence,
        multiple triggers (any fires, all gated by dedup + privacy):
          - app-switch: frontmost app changed (immediate; bypasses debounce)
          - activity-settle: input happened then just stopped (idle within
            [1s, activity_window]) — lands the shot on the *result* of an action
          - fallback timer: capture at least every `interval_seconds` even idle
        A `min_interval_seconds` debounce floors the rate for the
        activity/fallback triggers (app-switch and force bypass it). `idle` is
        seconds-since-last-input (from the daemon's presence read); None ⇒ the
        activity trigger is inert and we rely on app-switch + fallback.
        Never raises."""
        if self._paused:
            return {"captured": False, "reason": "paused"}
        if not self.ok:
            return {"captured": False, "reason": f"privacy: {self.reason}"}
        if self.embodiment is None:
            return {"captured": False, "reason": "no embodiment"}

        now = self._time_fn()
        app_now = self._frontmost()
        app_switched = (app_now is not None and app_now != self._last_app)
        elapsed = now - self._last_capture
        if force or app_switched:
            pass  # high-value triggers bypass the debounce
        else:
            if elapsed < self.min_interval_seconds:
                return {"captured": False, "reason": "debounce"}
            settled = (idle is not None
                       and 1.0 <= idle <= self.activity_window_seconds)
            fallback = elapsed >= self.interval_seconds
            if not (settled or fallback):
                return {"captured": False, "reason": "not due"}

        # Exclusion is checked on the cheap app read first — never even
        # screenshot a sensitive app.
        if self._excluded(app_now):
            self._last_app = app_now
            self.skipped += 1
            return {"captured": False, "reason": f"excluded app: {app_now}"}

        try:
            obs = self.embodiment.observe()
        except Exception as e:  # noqa: BLE001
            return {"captured": False, "reason": f"observe failed: {e}"}

        app = obs.frontmost_app or app_now
        if self._excluded(app):
            self._last_app = app
            self.skipped += 1
            return {"captured": False, "reason": f"excluded app: {app}"}

        frame = obs.frame
        path = frame.path if frame else None

        # Change-dedup: if the screen is ~unchanged AND the app didn't switch,
        # skip the VLM+embed entirely (but still drop the pixels). This is what
        # makes the effective capture rate track real screen activity.
        fp = self._fingerprint(path) if self.change_detect else None
        unchanged = (fp is not None and fp == self._last_fingerprint
                     and not app_switched)
        self._last_capture = now
        self._last_app = app
        if fp is not None:
            self._last_fingerprint = fp
        if unchanged and not force:
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass
            self.deduped += 1
            return {"captured": False, "reason": "unchanged"}

        # DINOv2 image embedding (the screen-state vector) BEFORE pixel-drop,
        # when a visual embedder is configured + available. Kept raw (vis_vec)
        # for the content-change comparison, and packed (img_blob) for storage.
        img_blob = None
        vis_vec = None
        if self.visual_embedder is not None and path:
            try:
                if getattr(self.visual_embedder, "available", False):
                    vis_vec = self.visual_embedder.embed(path)
                    if vis_vec:
                        from .embeddings import _pack_floats
                        img_blob = _pack_floats(vis_vec)
            except Exception:
                vis_vec = None
                img_blob = None

        # Did the screen CONTENT change meaningfully (not just the app)? This is
        # what decides whether to spend the strong model — so reading a feed for
        # hours still gets a deep read each time the content actually changes.
        changed = content_changed(vis_vec, self._last_described_vec,
                                   app_switched, self.content_change_threshold)
        use_strong = (changed and self.deep_read_on_change
                      and bool(self.vision_model_strong))

        description = ""
        model = ""
        try:
            if path and self.vision_model:
                from .knowledge import render_rules
                rules = render_rules(self.cfg.db_path)
                app_known = app or "an unknown app"
                model = self.vision_model_strong if use_strong else self.vision_model
                if use_strong:
                    # Deep, content-focused read: WHAT is being read/done, not
                    # just which app. The substance — topics, titles, threads.
                    instruction = (
                        f"The macOS frontmost app is '{app_known}' (GROUND TRUTH; "
                        "do not rename it).\n"
                        f"Where to look on screen:\n{rules}\n\n"
                        "Describe WHAT THE USER IS READING OR DOING — the actual "
                        "CONTENT, not just the app. Name the topic/subject, the "
                        "key headlines, post titles, usernames or threads visible, "
                        "and the gist. 1-2 sentences. For a feed (X, Reddit, news) "
                        "name the specific topics and the most notable posts. "
                        "Never transcribe passwords, secrets, tokens, or full "
                        "private messages.")
                else:
                    instruction = (
                        f"The macOS frontmost app is '{app_known}' (GROUND TRUTH).\n"
                        f"Where to look on screen:\n{rules}\n\n"
                        "Reply with ONE short sentence: what is on screen / what "
                        "is the user doing? No preamble, no list.")
                raw = self.llm.describe_image(model, instruction, path)
                description = clean_vision_text(raw)
        finally:
            # PIXEL-DROP: delete the screenshot no matter what happened above.
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass

        # Remember the embedding of the frame we actually read, so the NEXT
        # content-change comparison is against the last *described* screen.
        if description and vis_vec is not None:
            self._last_described_vec = vis_vec

        rendered = obs.render_text(limit=12)
        content = description or rendered or f"(screen: {app or 'unknown'})"
        content = f"[{app or 'unknown'}] {content}"[:500]

        tags = [f"app:{(app or 'unknown').lower()}"] + topic_tags(description)
        self.memory.store(
            task="screen_observation", kind="activity",
            content=content, salience=0.4 + (0.15 if use_strong else 0.0),
            mem_type=OBSERVATION,
            tags=tags,
            embedding=img_blob,   # DINOv2 image vector when enabled, else None
        )
        self.captured += 1
        return {"captured": True, "app": app, "content": content[:120],
                "trigger": "app_switch" if app_switched else
                           ("content_change" if changed else "timer"),
                "model": model,
                "model_tier": "strong" if use_strong else "fast",
                "tags": tags}

    def stats(self) -> dict[str, Any]:
        return {"captured": self.captured, "skipped": self.skipped,
                "deduped": self.deduped, "paused": self._paused,
                "privacy_ok": self.ok, "reason": self.reason}
