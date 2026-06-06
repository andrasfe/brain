"""Presence — is the user actually at the machine?

Grounds the brain's wake/sleep in reality: the brain should sleep (and do its
heavy NREM/REM processing) precisely when the user has stepped away — screen
locked, screensaver on, or just idle — and wake when they return.

Detection uses macOS `ioreg`'s `HIDIdleTime` (nanoseconds since the last HID
input). No third-party dependency; pure subprocess. Returns None on
non-macOS / failure, so callers can fall back to the affect/clock model.
"""
from __future__ import annotations

import re
import subprocess
from typing import Optional

_IDLE_RE = re.compile(r'"HIDIdleTime"\s*=\s*(\d+)')


def idle_seconds() -> Optional[float]:
    """Seconds since the last keyboard/mouse input, or None if unknown."""
    try:
        out = subprocess.run(
            ["ioreg", "-c", "IOHIDSystem"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return None
    vals = [int(m) for m in _IDLE_RE.findall(out)]
    if not vals:
        return None
    # Multiple entries exist; the smallest reflects the most recent input.
    return min(vals) / 1e9


def user_present(away_threshold_seconds: float = 300.0) -> Optional[bool]:
    """True if the user seems present (idle below threshold), False if away,
    None if presence can't be determined (caller should fall back)."""
    idle = idle_seconds()
    if idle is None:
        return None
    return idle < away_threshold_seconds
