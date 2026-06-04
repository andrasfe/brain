"""FileTailAdapter — tail -f for one or more files.

Each new line becomes an ambient StreamItem. Useful for plumbing in log
files, mail spool drops, IRC client output files, anything that writes
line-delimited events to disk. No threads; poll() just reads any new
bytes since last call.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from .base import InputAdapter, StreamItem


class FileTailAdapter(InputAdapter):
    name = "file_tail"
    default_channel = "ambient"

    def __init__(self, paths: Iterable[str | Path],
                 source_name: str = "file_tail",
                 channel: str = "ambient",
                 kind: str = "ambient",
                 sender: str | None = None,
                 start_at_end: bool = True,
                 max_lines_per_poll: int = 50):
        self.name = source_name
        self.default_channel = channel
        self.kind = kind
        self.sender = sender
        self.paths = [Path(os.path.expanduser(str(p))) for p in paths]
        self.start_at_end = start_at_end
        self.max_lines_per_poll = max_lines_per_poll
        self._offsets: dict[str, int] = {}

    def start(self) -> None:
        # Seed offsets: end-of-file if start_at_end else 0
        for p in self.paths:
            key = str(p)
            if p.exists() and self.start_at_end:
                self._offsets[key] = p.stat().st_size
            else:
                self._offsets[key] = 0

    def poll(self) -> list[StreamItem]:
        items: list[StreamItem] = []
        for p in self.paths:
            key = str(p)
            try:
                if not p.exists():
                    continue
                size = p.stat().st_size
                offset = self._offsets.get(key, 0)
                if size < offset:
                    # Truncated / rotated — re-seek from start of new file
                    offset = 0
                if size == offset:
                    continue
                with open(p, "rb") as fh:
                    fh.seek(offset)
                    chunk = fh.read(size - offset)
                self._offsets[key] = size
                text = chunk.decode("utf-8", errors="replace")
                for line in text.splitlines()[: self.max_lines_per_poll]:
                    line = line.strip()
                    if not line:
                        continue
                    items.append(StreamItem(
                        source=self.name, kind=self.kind, content=line,
                        channel=self.default_channel, sender=self.sender,
                        metadata={"path": str(p)},
                    ))
            except Exception:
                # Errors are non-fatal — the daemon keeps ticking
                continue
        return items
