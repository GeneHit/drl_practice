import importlib.util
import time
from pathlib import Path
from typing import Union, cast

from practice.base.config import BaseConfig
from practice.base.context import ContextBase
from practice.base.env_typing import EnvType


def load_config_module(
    config_path: str, mode: str
) -> tuple[BaseConfig, Union[ContextBase, EnvType]]:
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

    if mode == "train":
        # For train mode, we need the full context
        if not hasattr(module, "generate_context"):
            raise AttributeError(
                f"Config module must define 'generate_context' function: {config_path}"
            )
        context = module.generate_context(config)
        return cast(BaseConfig, config), cast(ContextBase, context)
    else:
        # For play and push_to_hub modes, we only need the environment
        if not hasattr(module, "get_env_for_play_and_hub"):
            raise AttributeError(
                f"Config module must define 'get_env_for_play_and_hub' function: {config_path}"
            )
        env = module.get_env_for_play_and_hub(config)
        return cast(BaseConfig, config), cast(EnvType, env)


def get_utc_time_str() -> str:
    """Get the UTC-time string."""
    return f"UTC {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}"
