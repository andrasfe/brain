"""Base class for brain regions (agents).

A Region is a specialized agent with a name, a tier-resolved model, a system
prompt describing its function, and a `step(workspace)` method that reads the
shared workspace and posts a Broadcast.
"""
from __future__ import annotations

from typing import Optional

from .config import Config
from .llm import LLM
from .workspace import Broadcast, Workspace


class Region:
    name: str = "region"
    system_prompt: str = "You are a brain region."

    def __init__(self, cfg: Config, llm: LLM):
        self.cfg = cfg
        self.llm = llm
        self.model = cfg.model_for(self.name)

    def step(self, ws: Workspace) -> Optional[Broadcast]:  # pragma: no cover - abstract
        raise NotImplementedError

    # convenience
    def _chat_json(self, user: str, **kw):
        return self.llm.chat_json(self.model, self.system_prompt, user, **kw)

    def _chat(self, user: str, **kw):
        return self.llm.chat(self.model, self.system_prompt, user, **kw)
