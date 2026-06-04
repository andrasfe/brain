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
                 memory: Optional[Any] = None):
        self.cfg = cfg
        self.sandbox = cfg.sandbox_dir
        self.eff_cfg = cfg.effectors
        # confirm(prompt) -> bool; defaults to auto-allow (used in non-interactive runs)
        self.confirm = confirm or (lambda _msg: True)
        # memory may be None during tests/eval — `remind_self` becomes a no-op.
        self.memory = memory

    # ── dispatch ──────────────────────────────────────────────────────────────
    def available(self) -> list[str]:
        names = []
        if self.eff_cfg.get("filesystem", {}).get("enabled"):
            names += ["read_file", "write_file", "list_dir"]
        if self.eff_cfg.get("shell", {}).get("enabled"):
            names += ["shell"]
        if self.eff_cfg.get("web", {}).get("enabled"):
            names += ["web_fetch"]
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
