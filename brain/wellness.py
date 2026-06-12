"""Wellness — the brain occasionally looks at YOU.

The screen observer watches what you do; this watches how you're doing. Every
~25 minutes while you're present, it takes one webcam still, asks the strong
local VLM for the visible state (alert / tired / strained / distracted, eye and
posture cues, apparent mood), records ONLY the inference, and deletes the
pixels. The readings become wellness-tagged observations, so:

    python3 -m brain.recall "how did I look today?"        # queryable
    nightly journal: "…you looked increasingly tired after 22:00"
    python3 -m brain.wellness                                # one-shot, now

PRIVACY (same spine as screen capture, stricter):
  - refuses to run unless the LLM endpoint is local (privacy_ok),
  - the photo is spooled only until analyzed, then DELETED — never stored,
  - the prompt forbids identifying people or describing surroundings;
    only the primary person's visible state is summarized,
  - committed default is OFF.

Analysis is deferred through the job queue (kind='wellness_check') so the slow
model never blocks the daemon tick; TTL is short because a stale face reading
is worthless.
"""
from __future__ import annotations

import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from .observer import clean_vision_text, privacy_ok, store_screen_observation


def shoot_webcam(dest: str, *, warmup_seconds: float = 1.2,
                 device: str = "0") -> bool:
    """One webcam still via imagesnap (if installed) or ffmpeg/avfoundation.
    The short warmup avoids the dark first frames. Returns success."""
    imagesnap = shutil.which("imagesnap")
    try:
        if imagesnap:
            r = subprocess.run(
                [imagesnap, "-q", "-w", str(warmup_seconds), dest],
                capture_output=True, timeout=20)
            return r.returncode == 0 and os.path.exists(dest)
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return False
        r = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error",
             "-f", "avfoundation", "-framerate", "30", "-i", device,
             "-t", str(warmup_seconds), "-update", "1", "-y", dest],
            capture_output=True, timeout=25)
        return r.returncode == 0 and os.path.exists(dest)
    except Exception:
        return False


def build_wellness_instruction() -> str:
    return (
        "This is a webcam photo of the computer's user, taken with their "
        "consent for a private self-check.\n"
        "Assess ONLY their visible state: alertness vs fatigue (eyes, eyelids, "
        "posture), apparent focus vs distraction, tension, and overall mood. "
        "Do NOT identify anyone, do NOT describe the room, background, other "
        "people, or clothing.\n"
        "Return JSON exactly with these keys: "
        '{"person_visible": true|false, '
        '"fatigue": <0.0-1.0>, "tension": <0.0-1.0>, '
        '"mood": "<one word, e.g. focused|relaxed|tired|stressed|neutral>", '
        '"summary": "<one sentence about how they look right now>"}'
    )


def analyze_wellness(llm, model: str, photo_path: str) -> Optional[dict]:
    """VLM pass over a webcam still → structured reading, or None when no
    person is visible / output is degenerate. Never raises."""
    try:
        out = llm.chat_json_image(model, build_wellness_instruction(), photo_path) \
            if hasattr(llm, "chat_json_image") else None
    except Exception:
        out = None
    if out is None:
        # describe_image returns prose; parse the JSON out of it.
        try:
            raw = llm.describe_image(model, build_wellness_instruction(),
                                     photo_path)
        except Exception:
            return None
        out = _extract_json_obj(raw)
    if not isinstance(out, dict):
        return None
    if not out.get("person_visible", True):
        return None
    summary = clean_vision_text(str(out.get("summary") or "")).strip()
    if len(summary) < 8:
        return None
    def _clamp(v):
        try:
            return round(max(0.0, min(1.0, float(v))), 2)
        except (TypeError, ValueError):
            return None
    return {"fatigue": _clamp(out.get("fatigue")),
            "tension": _clamp(out.get("tension")),
            "mood": str(out.get("mood") or "neutral").strip().lower()[:24],
            "summary": summary[:240]}


def _extract_json_obj(text: str) -> Optional[dict]:
    import json
    import re
    if not text:
        return None
    m = re.search(r"\{[^{}]*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def record_wellness(memory, reading: dict) -> str:
    """Persist a reading as a wellness observation (Recall + journal pick it
    up like any other observation; mood becomes a tag)."""
    bits = [reading["summary"]]
    if reading.get("fatigue") is not None:
        bits.append(f"fatigue={reading['fatigue']}")
    if reading.get("tension") is not None:
        bits.append(f"tension={reading['tension']}")
    content = "; ".join(bits)
    return store_screen_observation(
        memory, "wellness", content, salience=0.55,
        agency="active")  # a self-check is about the human, who is present


class WellnessObserver:
    """Cadence + spool + enqueue. The slow analysis runs in the worker."""

    def __init__(self, cfg, job_queue, *, interval_seconds: float = 1500.0,
                 jitter_seconds: float = 300.0, ttl_seconds: float = 600.0,
                 warmup_seconds: float = 1.2, device: str = "0",
                 time_fn=time.monotonic, shoot_fn=shoot_webcam,
                 rng: Optional[random.Random] = None):
        self.cfg = cfg
        self.job_queue = job_queue
        self.interval_seconds = float(interval_seconds)
        self.jitter_seconds = float(jitter_seconds)
        self.ttl_seconds = float(ttl_seconds)
        self.warmup_seconds = float(warmup_seconds)
        self.device = device
        self._time_fn = time_fn
        self._shoot = shoot_fn
        self._rng = rng or random.Random()
        self._spool_dir = Path(cfg.sandbox_dir) / "spool"
        self._next_due = self._time_fn() + self._jittered_interval() * 0.25
        self.ok, self.reason = privacy_ok(cfg)
        if self.ok and not (shutil.which("imagesnap") or shutil.which("ffmpeg")):
            self.ok, self.reason = False, "no camera tool (imagesnap/ffmpeg)"
        self.checks = 0

    def _jittered_interval(self) -> float:
        return self.interval_seconds + self._rng.uniform(0, self.jitter_seconds)

    def maybe_check(self, *, present: Optional[bool]) -> dict[str, Any]:
        """Shoot + enqueue at most once per (jittered) interval, only while the
        user is present (no point photographing an empty chair). Never raises."""
        if not self.ok:
            return {"checked": False, "reason": self.reason}
        if present is False:
            return {"checked": False, "reason": "away"}
        now = self._time_fn()
        if now < self._next_due:
            return {"checked": False, "reason": "not due"}
        self._next_due = now + self._jittered_interval()
        try:
            self._spool_dir.mkdir(parents=True, exist_ok=True)
            dest = str(self._spool_dir / f"wellness_{int(time.time())}.spool.png")
            if not self._shoot(dest, warmup_seconds=self.warmup_seconds,
                               device=self.device):
                return {"checked": False, "reason": "camera capture failed"}
            self.job_queue.enqueue(
                "wellness_check", {"spool": dest},
                priority=4, dedup_key="wellness",
                not_after_ts=time.time() + self.ttl_seconds)
            self.checks += 1
            return {"checked": True, "spool": dest}
        except Exception as e:  # noqa: BLE001 — never break the tick
            return {"checked": False, "reason": f"{type(e).__name__}: {e}"}


# ── CLI: one-shot "how do I look right now?" ─────────────────────────────────
def main() -> int:
    import sys
    import tempfile
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    from brain.config import load_config
    from brain.embeddings import make_backend
    from brain.llm import LLM
    from brain.memory import Memory

    cfg = load_config()
    ok, reason = privacy_ok(cfg)
    if not ok:
        print(f"refused: {reason}")
        return 1
    photo = os.path.join(tempfile.mkdtemp(), "wellness.png")
    print("📷 taking one webcam still…")
    if not shoot_webcam(photo):
        print("camera capture failed (grant camera access to the terminal, "
              "or `brew install imagesnap`)")
        return 1
    llm = LLM(cfg)
    memory = Memory(cfg.db_path, backend=make_backend(cfg, llm=llm))
    try:
        model = (str((cfg.capture or {}).get("vision_model_strong", ""))
                 or cfg.models.get("executive", ""))
        reading = analyze_wellness(llm, model, photo)
        if reading is None:
            print("no clear reading (no person visible?) — nothing recorded")
            return 0
        record_wellness(memory, reading)
        print(f"\nmood={reading['mood']}  fatigue={reading['fatigue']}  "
              f"tension={reading['tension']}\n{reading['summary']}")
        print("\n(recorded; pixels deleted)")
    finally:
        try:
            os.remove(photo)
        except OSError:
            pass
        memory.close()
        llm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
