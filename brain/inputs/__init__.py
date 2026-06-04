"""Input adapters — continuous information streams into the always-on brain.

Each adapter implements `poll() -> list[StreamItem]` (non-blocking) and gets
its items routed through the SalienceClassifier before reaching the
workspace. Two channels:

  ambient  — low salience by default, posted as broadcasts that can grab
             the spotlight if a classifier rule says so. RSS, metrics,
             ambient body signals.
  direct   — high salience, intended as something to deliberate on. Goes
             into the daemon's input queue. DMs, mentions, explicit
             prompts.

Adapters are NEVER trusted to set final salience themselves; the classifier
decides. This is the load-shedding contract for streaming on a local LLM
where every cognitive cycle costs seconds.
"""
from .base import InputAdapter, StreamItem
from .classifier import SalienceClassifier
from .file_tail_adapter import FileTailAdapter
from .stdin_adapter import StdinAdapter
from .webhook_adapter import WebhookAdapter

__all__ = [
    "InputAdapter", "StreamItem",
    "SalienceClassifier",
    "FileTailAdapter", "StdinAdapter", "WebhookAdapter",
]
