"""InputAdapter — base interface for continuous information streams."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamItem:
    """One discrete piece of information arriving from outside the brain.

    `salience` arrives 0 from the adapter; the classifier sets it. Items
    whose final salience falls below `classifier.ambient_threshold` are
    dropped or stored as low-salience episodic. Items above
    `classifier.direct_threshold` are enqueued as tasks (one or coalesced
    into a batch); in between they become ambient broadcasts.
    """
    source: str                          # adapter name, e.g. "stdin", "webhook"
    kind: str                            # "ambient" | "notification" | "social" | "task" | "body"
    content: str                         # the natural-language payload
    channel: str = "ambient"             # adapter's intent: "ambient" | "direct"
    salience: float = 0.0                # set by classifier
    ts: float = field(default_factory=time.time)
    sender: str | None = None            # for direct items: who sent it
    metadata: dict[str, Any] = field(default_factory=dict)

    def short(self, n: int = 80) -> str:
        return self.content if len(self.content) <= n else self.content[: n - 1] + "…"


class InputAdapter(ABC):
    """Each adapter is a non-blocking source of StreamItems."""

    name: str = "input"
    default_channel: str = "ambient"

    def start(self) -> None:
        """Optional: bring up resources (threads, sockets, file handles).
        Called once before the first poll()."""

    @abstractmethod
    def poll(self) -> list[StreamItem]:
        """Return any items available now. MUST be non-blocking. Returning
        an empty list is fine and expected most ticks."""

    def close(self) -> None:
        """Tear down any resources."""
