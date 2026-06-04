"""WebhookAdapter — tiny built-in HTTP server, no deps.

Accepts POSTs and queues each request body as a StreamItem. The HTTP
handler runs in a background thread; `poll()` drains the queue. Channel
is configurable per request via a `?channel=direct` query string or
`X-Brain-Channel` header.

This is the simplest way to plug arbitrary local tools into the brain:
mail-watcher daemons, file-watcher scripts, Slack relays, calendar
notifiers — all just POST to http://localhost:<port>.

Usage:
    adapter = WebhookAdapter(port=8765,
                              senders_by_path={"/slack": "slack",
                                                "/mail": "mail"})
    adapter.start()
    ...
"""
from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .base import InputAdapter, StreamItem


class _Handler(BaseHTTPRequestHandler):
    # Set by the adapter
    _adapter: "WebhookAdapter" = None  # type: ignore

    def log_message(self, fmt, *args):  # silence stderr noise
        return

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        body = self.rfile.read(length).decode("utf-8", errors="replace") if length else ""
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        content_type = (self.headers.get("Content-Type") or "").lower()
        content = body
        metadata: dict = {"path": parsed.path}
        # If JSON, pull a `content` field if present, dump the rest into metadata
        if "application/json" in content_type:
            try:
                data = json.loads(body) if body else {}
                if isinstance(data, dict):
                    content = str(data.get("content") or data.get("text")
                                   or data.get("message") or body)
                    metadata.update({k: v for k, v in data.items()
                                      if k not in ("content", "text", "message")})
            except json.JSONDecodeError:
                pass

        # channel: from query, header, default by path mapping, or adapter default
        channel = (
            (qs.get("channel") or [None])[0]
            or self.headers.get("X-Brain-Channel")
            or self._adapter.channel_by_path.get(parsed.path)
            or self._adapter.default_channel
        )
        kind = (qs.get("kind") or [None])[0] or "notification"
        sender = (
            (qs.get("sender") or [None])[0]
            or self.headers.get("X-Brain-Sender")
            or self._adapter.senders_by_path.get(parsed.path)
        )
        item = StreamItem(
            source=self._adapter.name,
            kind=kind,
            content=content,
            channel=channel,
            sender=sender,
            metadata=metadata,
        )
        self._adapter._inbox.put(item)
        self.send_response(202)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')


class WebhookAdapter(InputAdapter):
    name = "webhook"
    default_channel = "ambient"

    def __init__(self, port: int = 8765, host: str = "127.0.0.1",
                 senders_by_path: Optional[dict[str, str]] = None,
                 channel_by_path: Optional[dict[str, str]] = None,
                 default_channel: str = "ambient",
                 max_queue: int = 1000):
        self.port = port
        self.host = host
        self.senders_by_path = senders_by_path or {}
        self.channel_by_path = channel_by_path or {}
        self.default_channel = default_channel
        self._inbox: "queue.Queue[StreamItem]" = queue.Queue(maxsize=max_queue)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._server is not None:
            return
        # Build a handler class with this adapter attached
        adapter = self
        class _BoundHandler(_Handler):
            pass
        _BoundHandler._adapter = adapter  # type: ignore[attr-defined]
        self._server = ThreadingHTTPServer((self.host, self.port), _BoundHandler)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                          name="brain-webhook",
                                          daemon=True)
        self._thread.start()

    def poll(self) -> list[StreamItem]:
        out: list[StreamItem] = []
        while True:
            try:
                out.append(self._inbox.get_nowait())
            except queue.Empty:
                break
        return out

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
