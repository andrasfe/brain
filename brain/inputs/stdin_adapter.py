"""StdinAdapter — non-blocking stdin line reader.

Formalizes what the daemon was doing inline. Each non-empty line becomes a
direct StreamItem (the user is speaking to the brain). Newline characters
are stripped; otherwise the content is passed through verbatim.
"""
from __future__ import annotations

import select
import sys
from typing import Optional

from .base import InputAdapter, StreamItem


class StdinAdapter(InputAdapter):
    name = "stdin"
    default_channel = "direct"

    def __init__(self, sender: Optional[str] = "user",
                 poll_timeout: float = 0.02):
        self.sender = sender
        self.poll_timeout = poll_timeout

    def poll(self) -> list[StreamItem]:
        if not sys.stdin or not sys.stdin.isatty():
            return []
        ready, _, _ = select.select([sys.stdin], [], [], self.poll_timeout)
        if not ready:
            return []
        line = sys.stdin.readline()
        if not line:
            return []
        text = line.strip()
        if not text:
            return []
        return [StreamItem(
            source=self.name, kind="task", content=text,
            channel="direct", sender=self.sender,
        )]
