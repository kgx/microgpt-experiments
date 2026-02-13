"""
Load scenario configs from configs/<scenario>.json.
Resolve paths relative to project root (directory containing configs/).
"""

import json
import os

# Project root: directory containing configs/
_ROOT = os.path.dirname(os.path.abspath(__file__))


def _config_path(scenario: str) -> str:
    return os.path.join(_ROOT, "configs", f"{scenario}.json")


def load_config(scenario: str) -> dict:
    """Load config for a scenario. Raises FileNotFoundError if config does not exist."""
    path = _config_path(scenario)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return json.load(f)


def resolve_data_path(config: dict) -> str:
    """Return absolute path for data path in config."""
    return os.path.join(_ROOT, config["data"]["path"])


def resolve_model_path(scenario: str) -> str:
    """Return default model path for a scenario: models/<scenario>/model.json."""
    return os.path.join(_ROOT, "models", scenario, "model.json")


def list_scenarios() -> list[str]:
    """Return list of scenario names that have a config file."""
    configs_dir = os.path.join(_ROOT, "configs")
    if not os.path.isdir(configs_dir):
        return []
    return [
        os.path.splitext(f)[0]
        for f in os.listdir(configs_dir)
        if f.endswith(".json")
    ]
