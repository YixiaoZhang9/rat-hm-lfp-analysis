"""Project-level path configuration loaded from config.yaml."""

import os
from copy import deepcopy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("RAT_HM_CONFIG", PROJECT_ROOT / "config.yaml"))

DEFAULT_CONFIG = {"paths": {}}


def _deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path=CONFIG_PATH):
    """Load project config, falling back to legacy local defaults."""
    config = deepcopy(DEFAULT_CONFIG)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        _deep_update(config, user_config)
    return config


def get_path(name, default=None, config_path=CONFIG_PATH):
    """Return a configured filesystem path as a pathlib.Path."""
    paths = load_config(config_path).get("paths", {})
    value = paths.get(name, default)
    if value is None:
        raise KeyError(
            f"Missing path config value: {name}. "
            "Copy config.example.yaml to config.yaml and set local paths."
        )
    return Path(value).expanduser()
