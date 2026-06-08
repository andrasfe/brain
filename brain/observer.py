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
from pathlib import Path
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
    r"i (need|should|will|see)\b|we (need|should|can|have|could)\b|let me\b|"
    r"okay,?\b|first,?\b|based on (the|this)|given (the|that|ground)|"
    r"synthesize|patterns? include|to (answer|describe)|"
    r"thinking|analysis|here('s| is)\b|step \d)",
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
    raw_full = text.strip()                      # keep quotes for span extraction
    text = raw_full.strip('"').strip("'")
    lines = []
    for raw in text.split("\n"):
        line = re.sub(r"^\s*(\d+[.)]|[-*+])\s+", "", raw).strip().strip('"').strip("'")
        line = _strip_markup(line)
        if line:
            lines.append(line)
    # 1) Prefer the LAST substantive non-preamble line (a clean answer from the
    #    strong model lands here, keeping the full sentence incl. any title).
    for line in reversed(lines):
        if len(line) >= 8 and not _PREAMBLE_RE.match(line):
            line = _NICETY_RE.sub("", _strip_markup(line)).strip()
            return (line[0].upper() + line[1:])[:200] if line else line
    # 2) The line is all rationalization (weak model): reasoning models bury the
    #    real answer in quotes — "We need concise: \"The user is reading X.\"".
    #    Take the longest quoted span that isn't itself preamble.
    quoted = [q.strip() for q in re.findall(r'"([^"]{15,200})"', raw_full)
              if not _PREAMBLE_RE.match(q.strip())]
    if quoted:
        best = _NICETY_RE.sub("", _strip_markup(max(quoted, key=len))).strip()
        if len(best) >= 8:
            return (best[0].upper() + best[1:])[:200]
    # 3) Fallback: sentence-split the whole blob, last non-preamble sentence.
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


def crop_box(win, img_px, disp_pts, min_side: int = 40):
    """Map a focused-window rect (in screen POINTS) to a pixel crop box on the
    screenshot, accounting for Retina scaling (points → pixels). Pure +
    testable. Returns (left, top, right, bottom) clamped to the image, or None
    when the result is degenerate / the window is off the main display."""
    try:
        x, y, w, h = (float(v) for v in win)
        iw, ih = (float(v) for v in img_px)
        dw, dh = (float(v) for v in disp_pts)
    except (TypeError, ValueError):
        return None
    if dw <= 0 or dh <= 0 or iw <= 0 or ih <= 0:
        return None
    sx, sy = iw / dw, ih / dh           # pixels per point (≈2.0 on Retina)
    left = max(0, min(int(x * sx), int(iw)))
    top = max(0, min(int(y * sy), int(ih)))
    right = max(0, min(int((x + w) * sx), int(iw)))
    bottom = max(0, min(int((y + h) * sy), int(ih)))
    if right - left < min_side or bottom - top < min_side:
        return None                     # off-screen / tiny → caller uses full frame
    return (left, top, right, bottom)


def build_read_instruction(rules: str, app: str, deep: bool) -> str:
    """The VLM prompt for a screen read. `deep` (strong model) captures the
    actual CONTENT; the shallow form is a quick one-liner. Shared by the inline
    fast path and the deferred deep_read worker so they stay in lock-step."""
    app_known = app or "an unknown app"
    if deep:
        return (
            f"The macOS frontmost app is '{app_known}' (GROUND TRUTH; do not "
            "rename it).\n"
            f"Where to look on screen:\n{rules}\n\n"
            "Describe WHAT THE USER IS READING OR DOING — the actual CONTENT, "
            "not just the app. Name the topic/subject, the key headlines, post "
            "titles, usernames or threads visible, and the gist. 1-2 sentences. "
            "For a feed (X, Reddit, news) name the specific topics and the most "
            "notable posts. Never transcribe passwords, secrets, tokens, or "
            "full private messages.")
    return (
        f"The macOS frontmost app is '{app_known}' (GROUND TRUTH).\n"
        f"Where to look on screen:\n{rules}\n\n"
        "Reply with ONE short sentence: what is on screen / what is the user "
        "doing? No preamble, no list.")


def store_screen_observation(memory, app: str, description: str,
                             embedding=None, salience: float = 0.55) -> str:
    """Persist one screen observation: [app]-prefixed content + topic tags +
    optional DINOv2 embedding. Used by both the inline path and the worker."""
    content = (description or f"(screen: {app or 'unknown'})")
    content = f"[{app or 'unknown'}] {content}"[:500]
    tags = [f"app:{(app or 'unknown').lower()}"] + topic_tags(description)
    memory.store(task="screen_observation", kind="activity", content=content,
                 salience=salience, mem_type=OBSERVATION, tags=tags,
                 embedding=embedding)
    return content


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
                 content_change_threshold: float = 0.05,
                 content_focus_window: bool = True,
                 job_queue=None,
                 defer_strong: bool = True,
                 deep_read_ttl_seconds: float = 600.0,
                 max_queued_reads: int = 200,
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
        # Content grabbing crops to the FRONTMOST WINDOW (not the full screen):
        # sharper change signal, cleaner reads, background windows excluded.
        self.content_focus_window = content_focus_window
        # Async deferral: when a JobQueue is attached, slow STRONG reads are
        # enqueued (with a spooled crop) instead of run inline — so the 27b
        # never blocks the daemon tick. Fast reads stay inline (they're quick).
        self.job_queue = job_queue
        self.defer_strong = defer_strong
        self.deep_read_ttl_seconds = float(deep_read_ttl_seconds)
        self.max_queued_reads = int(max_queued_reads)
        self._spool_dir = Path(cfg.sandbox_dir) / "spool"
        self._last_described_vec = None   # DINOv2 vec of the last frame we read
        self.queued = 0
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

    def _focused_window_bounds(self):
        """(x, y, w, h) of the frontmost window in screen POINTS, or None.
        Via System Events (needs Accessibility, already granted for hands)."""
        script = ('tell application "System Events" to tell '
                  '(first application process whose frontmost is true) to '
                  'get {position, size} of front window')
        try:
            r = subprocess.run(["osascript", "-e", script],
                               capture_output=True, text=True, timeout=4)
            nums = [int(float(n)) for n in re.findall(r"-?\d+\.?\d*", r.stdout)]
            return tuple(nums[:4]) if len(nums) >= 4 else None
        except Exception:
            return None

    def _display_points(self):
        """Main display size in POINTS (logical), for the point→pixel scale."""
        try:
            r = subprocess.run(
                ["osascript", "-e",
                 'tell application "Finder" to get bounds of window of desktop'],
                capture_output=True, text=True, timeout=4)
            nums = [int(float(n)) for n in re.findall(r"-?\d+\.?\d*", r.stdout)]
            # bounds = {x1, y1, x2, y2} → width/height
            if len(nums) >= 4:
                return (nums[2] - nums[0], nums[3] - nums[1])
        except Exception:
            pass
        return None

    def _image_pixels(self, path):
        """(width, height) of the screenshot in PIXELS, via sips."""
        if not self._sips or not path:
            return None
        try:
            r = subprocess.run(
                [self._sips, "-g", "pixelWidth", "-g", "pixelHeight", path],
                capture_output=True, text=True, timeout=5)
            w = re.search(r"pixelWidth:\s*(\d+)", r.stdout)
            h = re.search(r"pixelHeight:\s*(\d+)", r.stdout)
            return (int(w.group(1)), int(h.group(1))) if w and h else None
        except Exception:
            return None

    def _focus_crop(self, path):
        """Crop the screenshot to the frontmost window and return the new path,
        or None to fall back to the full frame. The caller drops both files."""
        if not self.content_focus_window or not path:
            return None
        win = self._focused_window_bounds()
        img = self._image_pixels(path)
        disp = self._display_points()
        if not (win and img and disp):
            return None
        box = crop_box(win, img, disp)
        if not box:
            return None
        try:
            from PIL import Image
            out = path + ".focus.png"
            Image.open(path).crop(box).save(out)
            return out
        except Exception:
            return None

    def _spool(self, path):
        """Move a screenshot into the durable spool dir for a deferred read.
        Returns the new path, or None on failure (caller falls back to inline)."""
        if not path or not os.path.exists(path):
            return None
        try:
            self._spool_dir.mkdir(parents=True, exist_ok=True)
            dest = str(self._spool_dir / (os.path.basename(path) + ".spool.png"))
            os.replace(path, dest)
            return dest
        except OSError:
            return None

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

        # Content grabbing uses a crop of the FRONTMOST WINDOW (full-screen
        # capture stays for the world model / privacy spine). The crop sharpens
        # the content-change signal (no static chrome) and excludes background
        # windows from what gets embedded/described. Falls back to the full
        # frame when window bounds aren't available.
        focus_path = self._focus_crop(path)
        read_path = focus_path or path

        # DINOv2 image embedding (the screen-state vector) of the focused window
        # BEFORE pixel-drop. Kept raw (vis_vec) for the content-change
        # comparison, and packed (img_blob) for storage.
        img_blob = None
        vis_vec = None
        if self.visual_embedder is not None and read_path:
            try:
                if getattr(self.visual_embedder, "available", False):
                    vis_vec = self.visual_embedder.embed(read_path)
                    if vis_vec:
                        from .embeddings import _pack_floats
                        img_blob = _pack_floats(vis_vec)
            except Exception:
                vis_vec = None
                img_blob = None

        # Did the screen CONTENT change meaningfully (not just the app)? This is
        # what decides whether to spend the strong model — so reading a feed for
        # hours still gets a deep read each time the content actually changes.
        # We compute the raw DINOv2 distance explicitly so it can be logged and
        # the threshold calibrated against real screenshots (which cluster tight).
        content_dist = None
        if vis_vec is not None and self._last_described_vec is not None:
            from .screen_model import cosine_distance
            content_dist = round(float(cosine_distance(
                vis_vec, self._last_described_vec)), 4)
        changed = bool(
            app_switched
            or (vis_vec is not None and self._last_described_vec is None)
            or (content_dist is not None
                and content_dist >= self.content_change_threshold))
        use_strong = (changed and self.deep_read_on_change
                      and bool(self.vision_model_strong))

        trigger = ("app_switch" if app_switched
                   else ("content_change" if changed else "timer"))

        # ── DEFERRED deep read: enqueue the slow strong read off the tick ──
        # Spool the focused-window crop, enqueue a deep_read job, drop the full
        # frame, and return immediately. A worker comprehends it later (during
        # sleep). We advance the content-change baseline NOW (we've committed to
        # reading this frame) so we don't re-enqueue every tick.
        if (use_strong and self.job_queue is not None and self.defer_strong
                and read_path):
            spool = self._spool(read_path)
            other = focus_path if read_path == path else path  # the un-spooled one
            if other:
                try:
                    os.remove(other)
                except OSError:
                    pass
            if spool:
                try:
                    self.job_queue.enqueue(
                        "deep_read",
                        {"spool": spool, "app": app, "vec": vis_vec},
                        priority=10 if app_switched else 5,
                        dedup_key=f"win:{(app or 'unknown').lower()}",
                        not_after_ts=now + self.deep_read_ttl_seconds)
                    self.job_queue.trim("deep_read", self.max_queued_reads)
                except Exception:
                    pass
                if vis_vec is not None:
                    self._last_described_vec = vis_vec
                self.captured += 1
                self.queued += 1
                return {"captured": True, "queued": True, "app": app,
                        "trigger": trigger, "model": self.vision_model_strong,
                        "model_tier": "strong(queued)", "content_dist": content_dist,
                        "spool": spool}
            # spool failed → fall through to an inline read on read_path

        # ── INLINE read (fast frames, or strong when no queue) ──
        description = ""
        model = ""
        try:
            if read_path and self.vision_model:
                from .knowledge import render_rules
                model = self.vision_model_strong if use_strong else self.vision_model
                instruction = build_read_instruction(
                    render_rules(self.cfg.db_path), app, deep=use_strong)
                raw = self.llm.describe_image(model, instruction, read_path)
                description = clean_vision_text(raw)
        finally:
            # PIXEL-DROP: delete the full screenshot AND the focused-window crop.
            for _p in (path, focus_path):
                if _p:
                    try:
                        os.remove(_p)
                    except OSError:
                        pass

        if description and vis_vec is not None:
            self._last_described_vec = vis_vec

        rendered = obs.render_text(limit=12)
        content = store_screen_observation(
            self.memory, app, description or rendered, embedding=img_blob,
            salience=0.4 + (0.15 if use_strong else 0.0))
        self.captured += 1
        return {"captured": True, "app": app, "content": content[:120],
                "trigger": trigger, "model": model,
                "model_tier": "strong" if use_strong else "fast",
                "content_dist": content_dist}

    def stats(self) -> dict[str, Any]:
        return {"captured": self.captured, "skipped": self.skipped,
                "deduped": self.deduped, "queued": self.queued,
                "paused": self._paused,
                "privacy_ok": self.ok, "reason": self.reason}
