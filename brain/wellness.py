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
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from .observer import clean_vision_text, privacy_ok, store_screen_observation


def choose_camera(listing: str) -> Optional[str]:
    """Pick the most webcam-looking device index from an avfoundation device
    listing. Capture cards ('Guermok USB3 Video'), screen-capture pseudo-
    devices, and Desk View are poor choices — a dongle with no signal makes
    ffmpeg hang forever. Prefer built-in / FaceTime / *cam* devices. Pure +
    testable."""
    devices: list[tuple[str, str]] = []
    in_video = False
    for line in (listing or "").splitlines():
        if "video devices" in line.lower():
            in_video = True
            continue
        if "audio devices" in line.lower():
            break
        if in_video:
            m = re.search(r"\[(\d+)\]\s+(.+)$", line)
            if m:
                devices.append((m.group(1), m.group(2).strip()))
    best: Optional[str] = None
    best_score = -1
    for _idx, name in devices:
        n = name.lower()
        if "capture screen" in n:
            continue
        if "facetime" in n or "built-in" in n:
            score = 100
        elif "desk view" in n:
            score = 1
        elif "iphone" in n:
            score = 5            # continuity camera: works but flaky/absent
        elif "cam" in n:         # webcam / SmartCam / camera
            score = 80
        else:
            score = 10           # unknown video device (could be a dead dongle)
        if score > best_score:
            # Return the NAME, not the index: avfoundation indices shift as
            # iPhone continuity cameras appear/disappear, and a stale index can
            # land on a screen-capture device (we observed exactly that).
            best_score, best = score, name
    return best


def detect_camera_device() -> Optional[str]:
    """choose_camera() over the live avfoundation listing."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    try:
        r = subprocess.run(
            [ffmpeg, "-hide_banner", "-f", "avfoundation",
             "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=15)
        return choose_camera((r.stderr or "") + (r.stdout or ""))
    except Exception:
        return None


def shoot_webcam(dest: str, *, warmup_seconds: float = 3.0,
                 device: str = "auto") -> bool:
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
        if device in ("auto", "", None):
            device = detect_camera_device()
            if device is None:
                return False
        # Stop by FRAME COUNT, not duration: some webcams (e.g. EMEET) report a
        # bogus timebase ("not enough frames to estimate rate"), so a -t stop
        # never triggers and ffmpeg runs until killed. -update keeps
        # overwriting dest, so the surviving frame is the last (warmed-up) one.
        frames = max(3, int(warmup_seconds * 30))
        # Downscale in-capture: a 4K well-lit PNG is ~8MB — too heavy for the
        # local endpoint (requests come back empty). 1280px is plenty for a
        # face/posture read and keeps the payload ~1MB.
        r = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error",
             "-f", "avfoundation", "-framerate", "30", "-i", str(device),
             "-frames:v", str(frames), "-vf", "scale=1280:-2",
             "-update", "1", "-y", dest],
            capture_output=True, timeout=25)
        return r.returncode == 0 and os.path.exists(dest)
    except Exception:
        return False


def build_wellness_instruction() -> str:
    return (
        "This is a webcam photo of the computer's user, taken with their "
        "consent for a private self-check and journal.\n"
        "Assess their visible state: alertness vs fatigue (eyes, eyelids, "
        "posture), apparent focus vs distraction, tension, and overall mood. "
        "Also note WHAT THEY ARE WEARING (brief: garment + color, glasses, "
        "headphones) and anything INTERESTING or CURIOUS in view — an unusual "
        "object, a pet, a coffee mug, a change from the ordinary. "
        "Do NOT identify anyone by name and do NOT describe other people who "
        "may be visible.\n"
        "Return JSON exactly with these keys: "
        '{"person_visible": true|false, '
        '"fatigue": <0.0-1.0>, "tension": <0.0-1.0>, '
        '"mood": "<one word, e.g. focused|relaxed|tired|stressed|neutral>", '
        '"attire": "<brief, e.g. grey hoodie, glasses, over-ear headphones>", '
        '"notable": "<one short clause about anything curious, or empty>", '
        '"summary": "<one sentence about how they look right now>"}'
    )


def person_present(llm, model: str, *, shoot_fn=None,
                   warmup_seconds: float = 3.0) -> str:
    """Quick 'is the chair occupied?' check via one webcam frame → 'present' |
    'absent' | 'unknown'. Cheap yes/no (use the reflex model). Pixels dropped.
    Used to gate the window survey: sweep only when nobody's sitting."""
    import os
    import tempfile
    import re
    shoot = shoot_fn or shoot_webcam
    path = os.path.join(tempfile.mkdtemp(), "presence.jpg")
    prompt = ("Look at this webcam image. Is a PERSON visibly present (a face "
              "or body in frame)? Reply with EXACTLY one word: yes or no.")
    try:
        if not shoot(path, warmup_seconds=warmup_seconds):
            return "unknown"
        raw = ""
        for _ in range(2):                # retry once (LM Studio JIT reloads)
            try:
                raw = llm.describe_image(model, prompt, path, max_tokens=700)
            except Exception:
                raw = ""
            if raw.strip():
                break
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    ans = clean_vision_text(raw).lower()
    if not ans:
        return "unknown"
    has_yes = re.search(r"\byes\b", ans) is not None
    has_no = re.search(r"\bno\b", ans) is not None
    if has_yes and not has_no:
        return "present"
    if has_no and not has_yes:
        return "absent"
    # phrasing fallback when the model didn't give a clean yes/no
    if any(p in ans for p in ("no person", "nobody", "no one", "empty",
                              "unoccupied", "no human", "no face", "absent")):
        return "absent"
    if any(p in ans for p in ("person", "someone", "seated", "sitting", "a man",
                              "a woman", "individual", "face", "head", "human")):
        return "present"
    return "unknown"


def analyze_wellness(llm, model: str, photo_path: str) -> Optional[dict]:
    """VLM pass over a webcam still → structured reading, or None when no
    person is visible / output is degenerate. Never raises."""
    try:
        out = llm.chat_json_image(model, build_wellness_instruction(), photo_path) \
            if hasattr(llm, "chat_json_image") else None
    except Exception:
        out = None
    if out is None:
        # describe_image returns prose; parse the JSON out of it. Retry once:
        # LM Studio transiently 400s with "Model unloaded" while it JIT-reloads.
        out = None
        for _attempt in range(2):
            try:
                # Reasoning models burn the default 400-token budget on CoT
                # before emitting the JSON — give them headroom.
                raw = llm.describe_image(model, build_wellness_instruction(),
                                         photo_path, max_tokens=1500)
            except Exception:
                raw = ""
            out = _extract_json_obj(raw)
            if out is not None:
                break
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
    def _clean(key, cap):
        v = str(out.get(key) or "").strip()
        # the model sometimes echoes the template placeholder
        return "" if (not v or v.startswith("<")) else v[:cap]
    return {"fatigue": _clamp(out.get("fatigue")),
            "tension": _clamp(out.get("tension")),
            "mood": str(out.get("mood") or "neutral").strip().lower()[:24],
            "attire": _clean("attire", 120),
            "notable": _clean("notable", 160),
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
    if reading.get("mood"):
        bits.append(f"mood={reading['mood']}")
    if reading.get("attire"):
        bits.append(f"wearing: {reading['attire']}")
    if reading.get("notable"):
        bits.append(f"notable: {reading['notable']}")
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
                 warmup_seconds: float = 3.0, device: str = "auto",
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
        if reading.get("attire"):
            print(f"wearing: {reading['attire']}")
        if reading.get("notable"):
            print(f"notable: {reading['notable']}")
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
