import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from practice.base.config import BaseConfig
from practice.base.context import ContextBase


def load_config_module(config_path: str, mode: str) -> tuple[BaseConfig, ContextBase]:
    """Load a Python config module and extract config and context/env based on mode.

    Args:
        config_path: Path to Python config file
        mode: Mode of operation - determines what to load

    Returns:
        For train mode: Tuple of (config, context)
        For play/push_to_hub modes: Tuple of (config, env)

    Raises:
        ImportError: If config module cannot be loaded
        AttributeError: If required functions are missing
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if not config_path.endswith(".py"):
        raise ValueError(f"Config file must be a Python file (.py): {config_path}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config module from: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Extract required functions
    if not hasattr(module, "get_app_config"):
        raise AttributeError(f"Config module must define 'get_app_config' function: {config_path}")

    # Get config
    config = module.get_app_config()

    assert mode in ["train", "play", "push_to_hub"]
    if not hasattr(module, "generate_context"):
        raise AttributeError(
            f"Config module must define 'generate_context' function: {config_path}"
        )
    context = module.generate_context(config)
    return cast(BaseConfig, config), cast(ContextBase, context)


def get_utc_time_str() -> str:
    """Get the UTC-time string."""
    return datetime.now(timezone.utc).isoformat()
