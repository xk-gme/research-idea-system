"""Configuration loading for the research idea system."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

SYSTEM_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = SYSTEM_ROOT.parent
DEFAULT_CONFIG_PATH = SYSTEM_ROOT / "config.yaml"

_DEFAULTS: dict[str, Any] = {
    "deepseek": {
        "model": "deepseek-flash",
        "thinking": "enabled",
        "temperature": 1.0,
        "max_tokens": 8192,
        "timeout_seconds": 180,
        "transient_retries": 2,
        "secret_path": "../project/secret.txt",
    },
    "search": {
        "backends": ["arxiv"],
        "max_results_per_query": 8,
        "politeness_seconds": 3.0,
        "request_timeout_seconds": 30,
    },
    "research": {
        "initial_queries": 5,
        "depth_levels": 2,
        "ideas_count": 10,
    },
    "coscientist": {
        "initial_hypotheses": 6,
        "rounds": 2,
        "elo_base": 1200,
        "elo_k": 32,
        "elo_matches_per_round": 8,
        "evolve_top_k": 3,
        "judge_temperature": 0.2,
    },
    "format": {
        "max_repair_attempts": 1,
    },
    "runtime": {
        "seed": 2026,
        "max_llm_calls": 200,
    },
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load config.yaml merged over defaults. Relative paths in the file are
    resolved against the directory holding the config file."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    overrides: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if loaded:
            overrides = loaded
    config = _merge(_DEFAULTS, overrides)
    secret_path = Path(config["deepseek"]["secret_path"])
    if not secret_path.is_absolute():
        secret_path = (config_path.parent / secret_path).resolve()
    config["deepseek"]["secret_path"] = str(secret_path)
    return config
