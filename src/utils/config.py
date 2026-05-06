"""
Config loader.

Loads `config.yaml` from the project root. Centralising tunables in one place
keeps the CRISP-DM Evaluation phase clean — we can sweep chunk sizes, k, and
temperature without grepping through code.
"""
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> Dict[str, Any]:
    """Load and return the project configuration as a dict."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {CONFIG_PATH}. "
            "Make sure you're running from the project root."
        )
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_root() -> Path:
    """Return the absolute path to the project root."""
    return PROJECT_ROOT
