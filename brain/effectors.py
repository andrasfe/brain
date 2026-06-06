"""Effectors — the brain's motor outputs. All confined to the sandbox dir.

Each effector takes a dict of args and returns (ok, result_text). The basal
ganglia selects an action; the orchestrator dispatches it here.

Two internal effectors that don't touch the sandbox or network:
  - `think`: a no-op for pure reasoning steps.
  - `remind_self`: register a prospective-memory item that re-surfaces in a
    future cycle when its trigger matches (keyword in percept / substring /
    absolute time). The brain's way of writing a sticky note to itself.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from .config import Config


class Effectors:
    def __init__(self, cfg: Config, confirm: Callable[[str], bool] | None = None,
                 memory: Optional[Any] = None, embodiment: Optional[Any] = None):
        self.cfg = cfg
        self.sandbox = cfg.sandbox_dir
        self.eff_cfg = cfg.effectors
        # confirm(prompt) -> bool; defaults to auto-allow (used in non-interactive runs)
        self.confirm = confirm or (lambda _msg: True)
        # memory may be None during tests/eval — `remind_self` becomes a no-op.
        self.memory = memory
        # embodiment is an afferent.Embodiment or None. When present, the
        # `screen_*` effectors give the brain hands on the host computer.
        # The afferent SafetyGate (read_only / confirm / rate limit) gates
        # every action — this is defense-in-depth on top of the basal
        # ganglia's go/no-go.
        self.embodiment = embodiment

    # ── dispatch ──────────────────────────────────────────────────────────────
    def available(self) -> list[str]:
        names = []
        if self.eff_cfg.get("filesystem", {}).get("enabled"):
            names += ["read_file", "write_file", "list_dir"]
        if self.eff_cfg.get("shell", {}).get("enabled"):
            names += ["shell"]
        if self.eff_cfg.get("web", {}).get("enabled"):
            names += ["web_fetch"]
        # Embodiment ("hands"): only advertise the verbs the attached body can
        # actually do. `look` (eyes) is offered whenever an embodiment exists.
        if self.embodiment is not None:
            caps = self.embodiment.capabilities()
            names += ["look"]
            if "click" in caps:
                names += ["screen_click"]
            if "type" in caps:
                names += ["screen_type"]
            if "key" in caps:
                names += ["screen_key"]
            if "scroll" in caps:
                names += ["screen_scroll"]
        names += ["think", "remind_self", "finish"]  # always-available internal effectors
        return names

    def execute(self, name: str, args: dict[str, Any]) -> tuple[bool, str]:
        fn = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "list_dir": self._list_dir,
            "shell": self._shell,
            "web_fetch": self._web_fetch,
            "think": self._think,
            "remind_self": self._remind_self,
            "look": self._look,
            "screen_click": self._screen_click,
            "screen_type": self._screen_type,
            "screen_key": self._screen_key,
            "screen_scroll": self._screen_scroll,
        }.get(name)
        if fn is None:
            return False, f"unknown effector: {name}"
        try:
            return fn(args)
        except Exception as e:  # noqa: BLE001 — surface errors back into the loop
            return False, f"{type(e).__name__}: {e}"

    # ── sandbox helpers ───────────────────────────────────────────────────────
    def _resolve(self, rel: str) -> Path:
        p = (self.sandbox / rel).resolve()
        if not str(p).startswith(str(self.sandbox)):
            raise PermissionError(f"path escapes sandbox: {rel}")
        return p

    # ── filesystem ────────────────────────────────────────────────────────────
    def _read_file(self, args: dict[str, Any]) -> tuple[bool, str]:
        p = self._resolve(args["path"])
        if not p.exists():
            return False, f"no such file: {args['path']}"
        return True, p.read_text(errors="replace")[:20000]

    def _write_file(self, args: dict[str, Any]) -> tuple[bool, str]:
        p = self._resolve(args["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args.get("content", ""))
        return True, f"wrote {len(args.get('content', ''))} bytes to {args['path']}"

    def _list_dir(self, args: dict[str, Any]) -> tuple[bool, str]:
        p = self._resolve(args.get("path", "."))
        if not p.exists():
            return False, f"no such dir: {args.get('path', '.')}"
        entries = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())
        return True, "\n".join(entries) or "(empty)"

    # ── shell ─────────────────────────────────────────────────────────────────
    def _shell(self, args: dict[str, Any]) -> tuple[bool, str]:
        cmd = args["command"]
        shcfg = self.eff_cfg.get("shell", {})
        if shcfg.get("require_confirmation", True):
            if not self.confirm(f"Run shell command in sandbox?\n  $ {cmd}"):
                return False, "shell command declined by user"
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(self.sandbox),
            capture_output=True,
            text=True,
            timeout=int(shcfg.get("timeout_seconds", 30)),
        )
        out = (proc.stdout + proc.stderr).strip()[:20000]
        return proc.returncode == 0, out or f"(exit {proc.returncode}, no output)"

    # ── web ───────────────────────────────────────────────────────────────────
    def _web_fetch(self, args: dict[str, Any]) -> tuple[bool, str]:
        url = args["url"]
        timeout = int(self.eff_cfg.get("web", {}).get("timeout_seconds", 30))
        r = httpx.get(url, timeout=timeout, follow_redirects=True,
                      headers={"User-Agent": "brain/0.1"})
        r.raise_for_status()
        text = r.text
        # crude tag strip for readability
        import re
        text = re.sub(r"<script.*?</script>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return True, text[:20000]

    # ── internal ──────────────────────────────────────────────────────────────
    def _think(self, args: dict[str, Any]) -> tuple[bool, str]:
        """A no-op effector for pure reasoning steps."""
        return True, args.get("note", "(thought)")

    def _remind_self(self, args: dict[str, Any]) -> tuple[bool, str]:
        """Register a prospective-memory item to re-surface later.

        Args:
          content: the reminder text (what to remember)
          trigger: 'keyword' | 'percept' | 'time' — how to detect re-surfacing
          pattern: query string (keyword/percept) or seconds-from-now (time)
          salience: 0..1, optional (default 0.75)
        """
        if self.memory is None:
            return False, "remind_self: memory not attached"
        content = (args.get("content") or "").strip()
        if not content:
            return False, "remind_self: empty content"
        trigger = (args.get("trigger") or "keyword").lower()
        pattern = str(args.get("pattern") or "").strip()
        salience = float(args.get("salience", 0.75))
        fires_after = None
        if trigger == "time":
            try:
                seconds = float(pattern)
                fires_after = time.time() + seconds
            except ValueError:
                return False, f"remind_self: time trigger needs numeric seconds, got {pattern!r}"
        if trigger not in ("keyword", "percept", "time"):
            return False, f"remind_self: unknown trigger {trigger!r}"
        if trigger != "time" and not pattern:
            return False, "remind_self: pattern required for keyword/percept"
        pid = self.memory.prospective_register(
            content=content, trigger_kind=trigger,
            trigger_pattern=pattern, salience=salience,
            fires_after_ts=fires_after,
        )
        return True, f"reminder #{pid} registered ({trigger}: {pattern[:40]})"

    # ── embodiment (eyes + hands via afferent) ──────────────────────────────────
    @staticmethod
    def _ar_text(res: Any) -> str:
        """Render an afferent ActionResult into the (ok, text) the loop wants,
        including any post-action screen state for world-model grounding."""
        bits = [res.reason or res.action]
        if res.steps is not None:
            bits.append(f"steps={res.steps}")
        if getattr(res, "state_after", None) is not None:
            bits.append("after: " + res.state_after.render_text(limit=12))
        return "; ".join(b for b in bits if b)

    def _look(self, args: dict[str, Any]) -> tuple[bool, str]:
        """Observe the screen (eyes). No side effects; always allowed."""
        if self.embodiment is None:
            return False, "look: no embodiment attached"
        obs = self.embodiment.observe()
        return True, obs.render_text(limit=20)

    def _screen_click(self, args: dict[str, Any]) -> tuple[bool, str]:
        if self.embodiment is None:
            return False, "screen_click: no embodiment attached"
        try:
            x = float(args["x_pct"]); y = float(args["y_pct"])
        except (KeyError, TypeError, ValueError):
            return False, "screen_click needs numeric x_pct, y_pct in [0,1]"
        res = self.embodiment.click_at(
            x, y, button=args.get("button", "left"), count=int(args.get("count", 1)))
        return res.ok, self._ar_text(res)

    def _screen_type(self, args: dict[str, Any]) -> tuple[bool, str]:
        if self.embodiment is None:
            return False, "screen_type: no embodiment attached"
        text = args.get("text")
        if not isinstance(text, str) or not text:
            return False, "screen_type needs non-empty 'text'"
        res = self.embodiment.type_text(
            text, secret=bool(args.get("secret", False)),
            append_enter=bool(args.get("append_enter", False)))
        return res.ok, self._ar_text(res)

    def _screen_key(self, args: dict[str, Any]) -> tuple[bool, str]:
        if self.embodiment is None:
            return False, "screen_key: no embodiment attached"
        combo = args.get("combo") or args.get("key")
        if not isinstance(combo, str) or not combo:
            return False, "screen_key needs 'combo' (e.g. 'cmd+c', 'return')"
        res = self.embodiment.key(combo)
        return res.ok, self._ar_text(res)

    def _screen_scroll(self, args: dict[str, Any]) -> tuple[bool, str]:
        if self.embodiment is None:
            return False, "screen_scroll: no embodiment attached"
        try:
            amount = int(args["amount"])
        except (KeyError, TypeError, ValueError):
            return False, "screen_scroll needs integer 'amount'"
        at = None
        if "x_pct" in args and "y_pct" in args:
            at = (float(args["x_pct"]), float(args["y_pct"]))
        res = self.embodiment.scroll(amount, at_pct=at)
        return res.ok, self._ar_text(res)
