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


def _scroll_action(direction: str, available: Sequence[str]):
    """Pick the right scroll action for the AVAILABLE hands. Prefer a real
    screen_scroll; otherwise scroll via screen_key (PageDown/PageUp) — which is
    how macOS scrolls when the backend has no scroll capability. Returns
    (effector, args) or None when no scroll-capable effector exists."""
    down = direction == "down"
    if "screen_scroll" in available:
        return "screen_scroll", {"amount": -_DEFAULT_STEPS if down else _DEFAULT_STEPS}
    if "screen_key" in available:
        return "screen_key", {"combo": "pagedown" if down else "pageup"}
    return None


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
    if effector == "shell":
        cmd = str(args.get("command", "")).lower()
        if any(h in cmd for h in _SCROLL_DOWN):
            sa = _scroll_action("down", available)
            if sa:
                return sa[0], sa[1], f"shell→{sa[0]} (down): shell can't touch the GUI"
        if any(h in cmd for h in _SCROLL_UP):
            sa = _scroll_action("up", available)
            if sa:
                return sa[0], sa[1], f"shell→{sa[0]} (up): shell can't touch the GUI"
        m = re.search(r'keystroke\s+"([^"]+)"', cmd)
        if m and "screen_type" in available:
            return "screen_type", {"text": m.group(1)}, \
                "shell→screen_type: shell can't type into the GUI"

    # 2) screen_key carrying a scroll INTENT in the wrong arg ────────────────
    if effector == "screen_key":
        combo = str(args.get("combo") or args.get("key") or "").lower().strip()
        bad = str(args.get("scroll") or args.get("direction") or "").lower().strip()
        if not combo and bad in ("down", "up"):
            sa = _scroll_action(bad, available)
            if sa:
                return sa[0], sa[1], f"screen_key(scroll={bad})→{sa[0]}"
        # If a real screen_scroll exists, prefer it over paging keys.
        if combo in ("pagedown", "page_down", "page down") and "screen_scroll" in available:
            return "screen_scroll", {"amount": -_DEFAULT_STEPS}, \
                "screen_key(pagedown)→screen_scroll"
        if combo in ("pageup", "page_up", "page up") and "screen_scroll" in available:
            return "screen_scroll", {"amount": _DEFAULT_STEPS}, \
                "screen_key(pageup)→screen_scroll"
        # Normalize a bare 'key' alias to 'combo' (afferent accepts either, but
        # keep it canonical for repair idempotence).
        if not args.get("combo") and isinstance(args.get("key"), str):
            return "screen_key", {"combo": args["key"]}, "screen_key key→combo"

    # 3) screen_scroll requested but unavailable → fall back to paging key ───
    if effector == "screen_scroll" and "screen_scroll" not in available:
        d = "down"
        if _is_int(args.get("amount")):
            d = "down" if int(args["amount"]) <= 0 else "up"
        elif str(args.get("direction") or args.get("amount") or "").lower() == "up":
            d = "up"
        sa = _scroll_action(d, available)
        if sa:
            return sa[0], sa[1], f"screen_scroll unavailable→{sa[0]} ({d})"

    # 4) screen_scroll arg normalization (when it IS available) ──────────────
    if effector == "screen_scroll" and "screen_scroll" in available:
        if not _is_int(args.get("amount")):
            d = str(args.get("direction") or args.get("amount") or "").lower().strip()
            amt = _DEFAULT_STEPS if d == "up" else -_DEFAULT_STEPS
            return effector, {**{k: v for k, v in args.items()
                                 if k != "direction"}, "amount": amt}, \
                f"normalized scroll amount ({'up' if amt > 0 else 'down'})"
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
