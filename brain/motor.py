"""Motor repair — deterministic correction of mis-selected screen actions.

Even with affordances in the prompt, a local model sometimes reaches for the
wrong embodied effector or the wrong arg names (e.g. proposing `shell` to send
Page-Down, or `screen_key` with `{scroll: "down"}`). Vetoing those just burns
cycles. Instead we *repair* the obvious cases into the correct screen effector
before gating/dispatch — turning churn into a successful action (and a real
world-model transition).

Pure, deterministic, no LLM. `repair_motor_action` returns
`(effector, args, note)`; `note` is "" when nothing was changed.
"""
from __future__ import annotations

import re
from typing import Sequence, Tuple

_SCROLL_DOWN = ("page down", "pagedown", "page_down", "scroll down",
                "scrolldown", "key code 121", "down arrow", "arrow down")
_SCROLL_UP = ("page up", "pageup", "page_up", "scroll up", "scrollup",
              "key code 116", "up arrow", "arrow up")

_DEFAULT_STEPS = 3   # a "comfortable" scroll nudge


def _is_int(v) -> bool:
    try:
        int(v)
        return True
    except (TypeError, ValueError):
        return False


def _is_frac(v) -> bool:
    try:
        f = float(v)
        return 0.0 <= f <= 1.0
    except (TypeError, ValueError):
        return False


def repair_motor_action(effector: str, args: dict,
                        available: Sequence[str],
                        embodied: bool = True) -> Tuple[str, dict, str]:
    """Repair common embodied-action mistakes. No-op (note="") when not
    embodied, when no screen effectors are available, or when nothing matches."""
    args = dict(args or {})
    has_screen = any(str(e).startswith("screen_") for e in available)
    if not embodied or not has_screen:
        return effector, args, ""

    # 1) shell misused to drive the GUI ─────────────────────────────────────
    if effector == "shell" and "shell" not in ("",):
        cmd = str(args.get("command", "")).lower()
        if any(h in cmd for h in _SCROLL_DOWN) and "screen_scroll" in available:
            return "screen_scroll", {"amount": -_DEFAULT_STEPS}, \
                "shell→screen_scroll (down): shell can't touch the GUI"
        if any(h in cmd for h in _SCROLL_UP) and "screen_scroll" in available:
            return "screen_scroll", {"amount": _DEFAULT_STEPS}, \
                "shell→screen_scroll (up): shell can't touch the GUI"
        m = re.search(r'keystroke\s+"([^"]+)"', cmd)
        if m and "screen_type" in available:
            return "screen_type", {"text": m.group(1)}, \
                "shell→screen_type: shell can't type into the GUI"

    # 2) screen_key misused for scrolling ───────────────────────────────────
    if effector == "screen_key":
        combo = str(args.get("combo") or args.get("key") or "").lower().strip()
        bad = str(args.get("scroll") or args.get("direction") or "").lower().strip()
        if not combo and bad in ("down", "up") and "screen_scroll" in available:
            amt = -_DEFAULT_STEPS if bad == "down" else _DEFAULT_STEPS
            return "screen_scroll", {"amount": amt}, \
                f"screen_key→screen_scroll ({bad}): wrong effector for scrolling"
        if combo in ("pagedown", "page_down", "page down") and "screen_scroll" in available:
            return "screen_scroll", {"amount": -_DEFAULT_STEPS}, \
                "screen_key(pagedown)→screen_scroll"
        if combo in ("pageup", "page_up", "page up") and "screen_scroll" in available:
            return "screen_scroll", {"amount": _DEFAULT_STEPS}, \
                "screen_key(pageup)→screen_scroll"

    # 3) screen_scroll arg normalization ────────────────────────────────────
    if effector == "screen_scroll":
        if not _is_int(args.get("amount")):
            d = str(args.get("direction") or args.get("amount") or "").lower().strip()
            if d == "down":
                return effector, {**{k: v for k, v in args.items()
                                     if k not in ("direction",)}, "amount": -_DEFAULT_STEPS}, \
                    "normalized scroll amount (down)"
            if d == "up":
                return effector, {**{k: v for k, v in args.items()
                                     if k not in ("direction",)}, "amount": _DEFAULT_STEPS}, \
                    "normalized scroll amount (up)"
            # no parseable direction → default to a gentle scroll down
            return effector, {**args, "amount": -_DEFAULT_STEPS}, \
                "defaulted missing scroll amount (down)"
        args["amount"] = int(args["amount"])

    # 4) screen_type alias ──────────────────────────────────────────────────
    if effector == "screen_type" and "text" not in args:
        for alias in ("content", "string", "value"):
            if isinstance(args.get(alias), str) and args[alias]:
                new = {k: v for k, v in args.items() if k != alias}
                new["text"] = args[alias]
                return effector, new, f"screen_type {alias}→text"

    # 5) screen_click coordinate aliasing ───────────────────────────────────
    if effector == "screen_click" and "x_pct" not in args:
        if _is_frac(args.get("x")) and _is_frac(args.get("y")):
            new = {k: v for k, v in args.items() if k not in ("x", "y")}
            new["x_pct"] = float(args["x"])
            new["y_pct"] = float(args["y"])
            return effector, new, "screen_click x,y→x_pct,y_pct"

    return effector, args, ""
