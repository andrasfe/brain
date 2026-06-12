"""Workers — drain the JobQueue off the daemon tick.

A `Worker` is a background thread that claims jobs and runs a registered
handler per kind. The strong-model screen reads are the prime tenant: they're
slow (tens of seconds on a 27b), so doing them inline blocked the daemon. Now
the capture loop enqueues a `deep_read` and a worker processes it on its own
schedule — gated by the daemon's sleep state so it drains aggressively while
you're away and only opportunistically while you're active.

Thread-safety: the JobQueue is shared (its own lock + check_same_thread=False),
but per-thread resources (LLM, Memory — each holds a SQLite/HTTP connection)
are built INSIDE the worker thread via `ctx_factory`, never handed across
threads. Handlers receive that ctx.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional


class Worker(threading.Thread):
    def __init__(self, queue, handlers: dict[str, Callable[[dict, dict], None]], *,
                 ctx_factory: Callable[[], dict], kinds: Optional[list] = None,
                 owner: str = "worker", lease_seconds: float = 900.0,
                 gate: Optional[Callable[[], bool]] = None,
                 poll_seconds: float = 1.0, idle_seconds: float = 5.0,
                 log: Optional[Callable[[str], None]] = None):
        super().__init__(daemon=True, name=owner)
        self.queue = queue
        self.handlers = handlers
        self.ctx_factory = ctx_factory
        self.kinds = kinds or list(handlers.keys())
        self.owner = owner
        self.lease_seconds = lease_seconds
        self.gate = gate or (lambda: True)
        self.poll_seconds = poll_seconds
        self.idle_seconds = idle_seconds
        self.log = log or (lambda _m: None)
        self._stop_event = threading.Event()
        self._ctx: Optional[dict] = None
        self.processed = 0
        self.failed = 0

    def run(self) -> None:
        try:
            self._ctx = self.ctx_factory()
        except Exception as e:  # noqa: BLE001
            self.log(f"[worker:{self.owner}] ctx init failed: {e}")
            return
        while not self._stop_event.is_set():
            try:
                if not self.gate():
                    self._stop_event.wait(self.idle_seconds)
                    continue
                job = self.queue.claim(kinds=self.kinds, owner=self.owner,
                                       lease_seconds=self.lease_seconds)
            except Exception as e:  # noqa: BLE001 — never let the worker die
                self.log(f"[worker:{self.owner}] claim error: {e}")
                self._stop_event.wait(self.idle_seconds)
                continue
            if job is None:
                self._stop_event.wait(self.idle_seconds)
                continue
            handler = self.handlers.get(job["kind"])
            if handler is None:
                self.queue.fail(job["id"], "no handler")
                continue
            try:
                handler(job["payload"], self._ctx)
                self.queue.complete(job["id"])
                self.processed += 1
            except Exception as e:  # noqa: BLE001
                status = self.queue.fail(job["id"], f"{type(e).__name__}: {e}")
                self.failed += 1
                self.log(f"[worker:{self.owner}] job {job['id']} {status}: {e}")
            self._stop_event.wait(self.poll_seconds)
        # graceful shutdown: close per-thread resources
        self._close_ctx()

    def _close_ctx(self) -> None:
        for key in ("memory", "llm"):
            obj = (self._ctx or {}).get(key)
            try:
                if obj is not None and hasattr(obj, "close"):
                    obj.close()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop_event.set()


# ── handlers ─────────────────────────────────────────────────────────────────
def run_deep_read(payload: dict, ctx: dict) -> None:
    """Strong-model deep read of a spooled screenshot → stored observation.
    Idempotent-ish: a missing spool file (already purged) is a no-op success.
    Always deletes the spool file (the deferred pixel-drop)."""
    import os
    from .observer import (build_read_instruction, clean_vision_text,
                           store_screen_observation)
    from .knowledge import render_rules

    cfg = ctx["cfg"]
    llm = ctx["llm"]
    memory = ctx["memory"]
    spool = payload.get("spool")
    app = payload.get("app") or "unknown"
    vec = payload.get("vec")
    agency = payload.get("agency") or "active"

    if not spool or not os.path.exists(spool):
        return  # the screen is long gone; nothing to read
    model = (str((cfg.capture or {}).get("vision_model_strong", ""))
             or cfg.models.get("executive", ""))
    description = ""
    try:
        instruction = build_read_instruction(render_rules(cfg.db_path), app,
                                             deep=True, agency=agency)
        raw = llm.describe_image(model, instruction, spool)
        description = clean_vision_text(raw)
    finally:
        try:
            os.remove(spool)               # deferred PIXEL-DROP
        except OSError:
            pass
    blob = None
    if vec:
        from .embeddings import _pack_floats
        try:
            blob = _pack_floats(vec)
        except Exception:
            blob = None
    store_screen_observation(memory, app, description, embedding=blob,
                             salience=0.6, agency=agency)
    log = ctx.get("log")
    if log:
        log(f"  🧠 deep-read [{app}/{agency}] → {description[:120]}")


def run_wellness_check(payload: dict, ctx: dict) -> None:
    """Strong-model read of a spooled webcam still → wellness observation.
    Pixels always deleted; a missing spool / no-person reading is a no-op."""
    import os
    from .wellness import analyze_wellness, record_wellness

    cfg = ctx["cfg"]
    spool = payload.get("spool")
    if not spool or not os.path.exists(spool):
        return
    model = (str((cfg.capture or {}).get("vision_model_strong", ""))
             or cfg.models.get("executive", ""))
    try:
        reading = analyze_wellness(ctx["llm"], model, spool)
    finally:
        try:
            os.remove(spool)               # PIXEL-DROP, unconditionally
        except OSError:
            pass
    if reading is None:
        return
    record_wellness(ctx["memory"], reading)
    log = ctx.get("log")
    if log:
        log(f"  🪞 wellness: mood={reading['mood']} fatigue={reading['fatigue']} "
            f"— {reading['summary'][:100]}")


DEFAULT_HANDLERS = {"deep_read": run_deep_read,
                    "wellness_check": run_wellness_check}
