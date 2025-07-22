import argparse
import importlib.util
import sys
from pathlib import Path
from typing import cast

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# below has to be imported after sys.path.insert(0, str(project_root))
from practice.exercise6_a3c.a3c_exercise import A3CConfig, a3c_train  # noqa: E402
from practice.utils.cli_utils import get_utc_time_str  # noqa: E402


def _load_config_module(config_path: str) -> A3CConfig:
    """Load a Python config module and extract config.

    Args:
        config_path: Path to Python config file

    Returns:
        The config for the A3C algorithm.

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
    return cast(A3CConfig, module.get_app_config())


def main() -> None:
    """The main function for the A3C algorithm."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = _load_config_module(args.config)
    # show UTC-time logging
    print(f"[{get_utc_time_str()}] A3C Training...")
    a3c_train(config)


if __name__ == "__main__":
    main()
