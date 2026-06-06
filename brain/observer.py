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

import os
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


class ScreenObserver:
    def __init__(self, cfg, llm, memory, embodiment, *,
                 interval_seconds: float = 30.0,
                 exclude_apps: Optional[list] = None,
                 vision_model: str = "",
                 time_fn=time.monotonic):
        self.cfg = cfg
        self.llm = llm
        self.memory = memory
        self.embodiment = embodiment
        self.interval_seconds = interval_seconds
        excludes = list(exclude_apps if exclude_apps is not None else _DEFAULT_EXCLUDE)
        self._exclude = {a.lower() for a in excludes}
        self.vision_model = vision_model or cfg.models.get("reflex", "")
        self._time_fn = time_fn
        self._last_capture = -1e9
        self._paused = False
        # Resolve privacy once; refuse if not local.
        self.ok, self.reason = privacy_ok(cfg)
        self.captured = 0
        self.skipped = 0

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def _excluded(self, app: Optional[str]) -> bool:
        if not app:
            return False
        a = app.lower()
        return any(x in a for x in self._exclude)

    def maybe_capture(self, *, force: bool = False) -> dict[str, Any]:
        """Capture one observation if due + allowed. Returns a small status
        dict (captured / skipped + reason). Never raises."""
        if self._paused:
            return {"captured": False, "reason": "paused"}
        if not self.ok:
            return {"captured": False, "reason": f"privacy: {self.reason}"}
        if self.embodiment is None:
            return {"captured": False, "reason": "no embodiment"}
        now = self._time_fn()
        if not force and (now - self._last_capture) < self.interval_seconds:
            return {"captured": False, "reason": "not due"}
        self._last_capture = now

        try:
            obs = self.embodiment.observe()
        except Exception as e:  # noqa: BLE001
            return {"captured": False, "reason": f"observe failed: {e}"}

        app = obs.frontmost_app
        if self._excluded(app):
            self.skipped += 1
            return {"captured": False, "reason": f"excluded app: {app}"}

        frame = obs.frame
        path = frame.path if frame else None
        description = ""
        try:
            if path and self.vision_model:
                description = self.llm.describe_image(
                    self.vision_model,
                    "In one sentence, what is the user doing on this screen? "
                    "Name the app and the activity. Do NOT transcribe any "
                    "passwords, secrets, or full personal messages.",
                    path,
                )
        finally:
            # PIXEL-DROP: delete the screenshot no matter what happened above.
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass

        # If the backend already gave a textual render (FakeBackend in tests,
        # or future structured backends), fold it in.
        rendered = obs.render_text(limit=12)
        content = description or rendered or f"(screen: {app or 'unknown'})"
        content = f"[{app or 'unknown'}] {content}"[:500]

        self.memory.store(
            task="screen_observation", kind="activity",
            content=content, salience=0.4,
            mem_type=OBSERVATION,
            tags=[f"app:{(app or 'unknown').lower()}"],
        )
        self.captured += 1
        return {"captured": True, "app": app, "content": content[:120]}

    def stats(self) -> dict[str, Any]:
        return {"captured": self.captured, "skipped": self.skipped,
                "paused": self._paused, "privacy_ok": self.ok,
                "reason": self.reason}
