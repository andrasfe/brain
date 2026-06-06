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
                msg = data["choices"][0]["message"]
                # Some reasoning models (Nemotron, DeepSeek-R1, …) put their
                # full output in `reasoning_content` and leave `content`
                # empty until the chain-of-thought finishes. We accept either:
                # prefer `content`, fall back to `reasoning_content`.
                # `chat_json` then extracts the JSON block from whichever
                # field had the text.
                content = msg.get("content") or ""
                if not content:
                    content = msg.get("reasoning_content") or ""
                return content
            except Exception as e:  # noqa: BLE001 — retry on any transport/HTTP error
                last_err = e
                if attempt < self.cfg.max_retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    def describe_image(
        self,
        model: str,
        prompt: str,
        image_path: str,
        *,
        system: str = "You are a screen-reading assistant. Be concise and literal.",
        temperature: float = 0.2,
        max_tokens: int = 400,
    ) -> str:
        """Send a local image to a vision-capable model (OpenAI vision shape:
        a content array with an image_url data URI). Used by the occipital
        region to turn a screenshot into a textual percept. Returns "" on
        failure rather than raising — vision is best-effort eyes, not a hard
        dependency of a cognitive cycle."""
        import base64
        import mimetypes

        try:
            with open(image_path, "rb") as fh:
                blob = fh.read()
        except OSError:
            return ""
        mime = mimetypes.guess_type(image_path)[0] or "image/png"
        data_uri = f"data:{mime};base64," + base64.b64encode(blob).decode("ascii")
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ]},
            ],
        }
        for attempt in range(self.cfg.max_retries + 1):
            try:
                r = self._client.post("/chat/completions", json=payload)
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                return (msg.get("content") or msg.get("reasoning_content") or "").strip()
            except Exception:  # noqa: BLE001 — best-effort; eyes degrade quietly
                if attempt < self.cfg.max_retries:
                    time.sleep(1.0 * (attempt + 1))
        return ""

    def chat_json(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Chat expecting a single JSON object back. Tolerant of code fences/prose.

        Reasoning models (Nemotron, qwen3 a3b, DeepSeek-R1, …) emit verbose
        chain-of-thought before any JSON; when their token budget gets
        clipped mid-reasoning, no JSON is produced at all and the parser
        returns `_parse_error`. We retry ONCE with 3× the budget and a
        stricter system prompt that pushes the model toward emitting the
        JSON object first (or at least surviving its reasoning to get there).
        """
        json_directive = (
            "\n\nRespond with a single valid JSON object and nothing else."
        )
        raw = self.chat(
            model,
            system + json_directive,
            user,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = _extract_json(raw)
        if not parsed.get("_parse_error"):
            # Attach raw text on success too — callers (e.g. prefrontal's
            # empty-content fallback) can recover something usable when a
            # reasoning model emitted JSON with blank fields.
            parsed.setdefault("_raw", raw)
            return parsed

        # Retry: bigger budget, sterner instruction, lower temperature.
        strict_directive = (
            "\n\nYou MUST emit exactly one JSON object and nothing else. "
            "Do not narrate your reasoning. Do not include preface text, "
            "code fences, or trailing commentary. Output starts with '{' "
            "and ends with '}'."
        )
        raw2 = self.chat(
            model,
            system + strict_directive,
            user,
            temperature=max(0.0, temperature - 0.2),
            max_tokens=max(max_tokens, max_tokens * 3),
        )
        parsed2 = _extract_json(raw2)
        if not parsed2.get("_parse_error"):
            return parsed2
        # Still failed — surface the first attempt's raw text since it
        # tends to be more informative for debugging.
        return parsed


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
