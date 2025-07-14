#!/usr/bin/env python3
"""Unified CLI for practice exercises with support for different modes.

This CLI can run various RL exercises by loading Python config modules and supports:
1. train: Train the model, then play game and generate video
2. play: Only play game and generate video (no training)
3. push_to_hub: Push model to hub with optional play/video generation
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# below has to be imported after sys.path.insert(0, str(project_root))
from practice.utils.train_utils import train_and_evaluate_network  # noqa: E402


def load_config_module(config_path: str) -> tuple[Any, Any]:
    """Load a Python config module and extract config and context functions.

    Args:
        config_path: Path to Python config file

    Returns:
        Tuple of (config, context) from the config module

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

    if not hasattr(module, "generate_context"):
        raise AttributeError(
            f"Config module must define 'generate_context' function: {config_path}"
        )

    # Get config and context
    config = module.get_app_config()
    context = module.generate_context(config)

    return config, context


def train_wrapper(config: Any, context: Any) -> None:
    """Wrapper for training that matches the expected signature."""
    train_and_evaluate_network(
        config=config,
        ctx=context,
        Trainer=context.trainer_name,
    )


def play_wrapper(config: Any, context: Any) -> None:
    """Simplified play wrapper for practice exercises."""
    print("Play mode functionality will be implemented when needed.")
    print("For now, you can check the results directory for videos generated during training.")


def push_to_hub_wrapper(config: Any, context: Any, username: str) -> None:
    """Simplified push to hub wrapper for practice exercises."""
    print("Push to hub functionality will be implemented when needed.")
    print(f"Model would be pushed to hub for user: {username}")
    print(f"Model file: {config.artifact_config.model_filename}")
    print(f"Repo ID: {config.artifact_config.repo_id}")


def _create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the unified CLI."""
    epilog = """
Mode descriptions:
  train: Train and evaluate the model.
  play: Only play game and generate video (no training) [NOT IMPLEMENTED YET]
  push_to_hub: Push model to hub with optional play/video generation [NOT IMPLEMENTED YET]

Examples:
  # Train mode (default)
  python practice/cli.py --config practice/exercise4_curiosity/config_mountain_car.py

  # Only play and generate video
  python practice/cli.py --config practice/exercise4_curiosity/config_mountain_car.py --mode play

  # Push to hub with play/video
  python practice/cli.py --config practice/exercise4_curiosity/config_mountain_car.py --mode push_to_hub --username myuser

  # Push to hub without play/video
  python practice/cli.py --config practice/exercise4_curiosity/config_mountain_car.py --mode push_to_hub --username myuser --skip_play
    """

    parser = argparse.ArgumentParser(
        description="Unified CLI for practice RL exercises",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to Python configuration file (e.g., practice/exercise4_curiosity/config_mountain_car.py)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "play", "push_to_hub"],
        default="train",
        help="Mode of operation (default: train)",
    )
    parser.add_argument(
        "--username",
        type=str,
        default="",
        help="Hugging Face username (required for push_to_hub mode)",
    )
    parser.add_argument(
        "--skip_play",
        action="store_true",
        default=False,
        help="Skip playing the game and video generation (only for push_to_hub mode)",
    )

    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate command line arguments."""
    # Validate username requirement for push_to_hub mode
    if args.mode == "push_to_hub" and not args.username:
        parser.error("--username is required when using push_to_hub mode")

    # Validate skip_play usage
    if args.skip_play and args.mode != "push_to_hub":
        parser.error("--skip_play can only be used with push_to_hub mode")


def main() -> None:
    """Main CLI function."""
    parser = _create_parser()
    args = parser.parse_args()

    # Validate arguments
    _validate_args(args, parser)

    try:
        # Load config and context from Python module
        print(f"Loading configuration from: {args.config}")
        config, context = load_config_module(args.config)

        # Execute the requested mode
        if args.mode == "train":
            print("=== Training Mode ===")
            train_wrapper(config, context)

        elif args.mode == "play":
            print("=== Play Mode ===")
            play_wrapper(config, context)

        elif args.mode == "push_to_hub":
            print("=== Push to Hub Mode ===")

            if not args.skip_play:
                print("Would play game and generate video first.")
                play_wrapper(config, context)

            push_to_hub_wrapper(config, context, args.username)

        print("=== Operation Complete ===")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
