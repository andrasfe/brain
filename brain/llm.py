"""Thin OpenRouter client with per-tier model routing and JSON helpers."""
from __future__ import annotations

import json
import time
from typing import Any, Optional

import httpx

from .config import Config


class LLM:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Headers are provider-driven: omit Authorization for local endpoints
        # that don't require it (Ollama, LM Studio, llama.cpp …). extra_headers
        # is whatever the profile asked for (OpenRouter wants HTTP-Referer +
        # X-Title; locals usually want nothing).
        headers: dict[str, str] = {}
        if cfg.require_auth and cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        headers.update(cfg.extra_headers or {})
        self._client = httpx.Client(
            base_url=cfg.base_url,
            timeout=cfg.timeout_seconds,
            headers=headers,
        )

    def close(self) -> None:
        self._client.close()

    def chat(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                r = self._client.post("/chat/completions", json=payload)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"] or ""
            except Exception as e:  # noqa: BLE001 — retry on any transport/HTTP error
                last_err = e
                if attempt < self.cfg.max_retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"OpenRouter call failed after retries: {last_err}")

    def chat_json(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Chat expecting a single JSON object back. Tolerant of code fences/prose."""
        raw = self.chat(
            model,
            system + "\n\nRespond with a single valid JSON object and nothing else.",
            user,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _extract_json(raw)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    # Find the outermost {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_parse_error": True, "_raw": text}
