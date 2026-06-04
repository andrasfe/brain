"""Configuration loading: merges config.yaml with credentials from ~/specter/.env."""
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


@dataclass
class Config:
    raw: dict[str, Any]
    api_key: str
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

    def model_for(self, region: str) -> str:
        """Resolve a region name to a concrete model id via its tier."""
        tier = self.regions.get(region, "reflex")
        return self.models.get(tier, self.models["reflex"])


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else _REPO_ROOT / "config.yaml"
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)

    # Load credentials from the specter .env (falls back to process env).
    env: dict[str, str] = {}
    specter_env = raw.get("specter_env")
    if specter_env:
        env_path = _expand(specter_env)
        if env_path.exists():
            env.update({k: v for k, v in dotenv_values(env_path).items() if v is not None})

    api_key = env.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            f"OPENROUTER_API_KEY not found in {specter_env} or environment."
        )

    orc = raw["openrouter"]
    sandbox = _expand(raw["sandbox_dir"])
    sandbox.mkdir(parents=True, exist_ok=True)

    return Config(
        raw=raw,
        api_key=api_key,
        base_url=orc["base_url"],
        models=orc["models"],
        timeout_seconds=int(orc.get("timeout_seconds", 120)),
        max_retries=int(orc.get("max_retries", 2)),
        sandbox_dir=sandbox,
        db_path=_expand(raw["memory"]["db_path"]),
        loop=raw.get("loop", {}),
        memory=raw.get("memory", {}),
        effectors=raw.get("effectors", {}),
        regions=raw.get("regions", {}),
    )
