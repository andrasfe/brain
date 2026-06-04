"""Configuration loading.

The LLM client section supports any OpenAI-compatible endpoint via the
`provider` key. Built-in shortcuts: 'openrouter' (default, requires
OPENROUTER_API_KEY), 'ollama' (localhost, no auth), 'lmstudio' (localhost,
no auth), 'vllm' / 'llamacpp' (custom). Any provider can be overridden
with explicit `base_url`, `require_auth`, `api_key_env`, and `extra_headers`.

This is what unblocks running the brain end-to-end against a local model
on an M4 Mac — no code changes needed to swap providers, only config.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _expand(p: str) -> Path:
    return Path(os.path.expanduser(p)).resolve()


# Built-in provider profiles. Each is a partial dict merged under the user's
# explicit overrides — user keys always win.
_PROVIDER_PROFILES: dict[str, dict[str, Any]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "require_auth": True,
        "api_key_env": "OPENROUTER_API_KEY",
        "extra_headers": {
            "HTTP-Referer": "https://localhost/brain",
            "X-Title": "brain",
        },
    },
    "ollama": {
        # Ollama exposes an OpenAI-compatible surface at /v1
        "base_url": "http://localhost:11434/v1",
        "require_auth": False,
        "api_key_env": "",
        "extra_headers": {},
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "require_auth": False,
        "api_key_env": "",
        "extra_headers": {},
    },
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "require_auth": False,
        "api_key_env": "",
        "extra_headers": {},
    },
    "llamacpp": {
        # llama.cpp's --api-server default
        "base_url": "http://localhost:8080/v1",
        "require_auth": False,
        "api_key_env": "",
        "extra_headers": {},
    },
    "custom": {
        # User MUST set base_url / require_auth themselves
        "require_auth": False,
        "api_key_env": "",
        "extra_headers": {},
    },
}


@dataclass
class Config:
    raw: dict[str, Any]
    api_key: str                       # may be "" when require_auth=False
    base_url: str
    models: dict[str, str]
    timeout_seconds: int
    max_retries: int
    sandbox_dir: Path
    db_path: Path
    loop: dict[str, Any]
    memory: dict[str, Any]
    effectors: dict[str, Any]
    regions: dict[str, str]
    # Provider profile fields default so older callers (tests, eval/) that
    # construct Config directly keep working. load_config always sets them.
    require_auth: bool = True
    extra_headers: dict[str, str] = field(default_factory=dict)

    def model_for(self, region: str) -> str:
        """Resolve a region name to a concrete model id via its tier."""
        tier = self.regions.get(region, "reflex")
        return self.models.get(tier, self.models["reflex"])


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else _REPO_ROOT / "config.yaml"
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)

    # Credentials from specter .env (back-compat) + process env
    env: dict[str, str] = {}
    specter_env = raw.get("specter_env")
    if specter_env:
        env_path = _expand(specter_env)
        if env_path.exists():
            env.update({k: v for k, v in dotenv_values(env_path).items()
                         if v is not None})

    # LLM endpoint config: the `openrouter` block stays the canonical name
    # for backwards compat; the `provider` field selects a profile and
    # explicit keys override the profile defaults.
    orc = dict(raw.get("openrouter", {}))
    provider = (orc.get("provider") or "openrouter").lower()
    profile = dict(_PROVIDER_PROFILES.get(provider, {}))
    # User overrides (anything in `orc`) win over profile
    profile.update({k: v for k, v in orc.items() if k != "provider"})

    base_url = profile.get("base_url")
    if not base_url:
        raise RuntimeError(
            f"provider={provider!r} has no base_url; either pick a built-in "
            f"profile (openrouter/ollama/lmstudio/vllm/llamacpp) or set "
            f"openrouter.base_url explicitly")
    require_auth = bool(profile.get("require_auth", True))
    api_key_env = str(profile.get("api_key_env") or "")
    extra_headers = dict(profile.get("extra_headers", {}) or {})

    api_key = ""
    if api_key_env:
        api_key = env.get(api_key_env) or os.environ.get(api_key_env, "")
    if require_auth and not api_key:
        hint = f" (looked up {api_key_env})" if api_key_env else ""
        raise RuntimeError(
            f"provider={provider!r} requires an API key{hint} but none found. "
            f"Set {api_key_env or 'the API key env var'} or switch to a "
            f"local provider (provider: ollama, lmstudio, vllm, llamacpp).")

    sandbox = _expand(raw["sandbox_dir"])
    sandbox.mkdir(parents=True, exist_ok=True)

    # Resolve persona_path relative to repo root if relative
    pp = raw.get("persona_path")
    if pp:
        pp_path = Path(os.path.expanduser(pp))
        if not pp_path.is_absolute():
            pp_path = (_REPO_ROOT / pp_path).resolve()
        raw["persona_path"] = str(pp_path)

    models = profile.get("models", {}) or orc.get("models", {})
    if not models:
        raise RuntimeError(
            "openrouter.models is empty — set models.reflex and "
            "models.executive")

    return Config(
        raw=raw,
        api_key=api_key,
        base_url=base_url,
        require_auth=require_auth,
        extra_headers=extra_headers,
        models=models,
        timeout_seconds=int(profile.get("timeout_seconds", 120)),
        max_retries=int(profile.get("max_retries", 2)),
        sandbox_dir=sandbox,
        db_path=_expand(raw["memory"]["db_path"]),
        loop=raw.get("loop", {}),
        memory=raw.get("memory", {}),
        effectors=raw.get("effectors", {}),
        regions=raw.get("regions", {}),
    )
