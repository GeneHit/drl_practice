#!/usr/bin/env python3
"""Unified CLI for practice exercises with support for different modes.

This CLI can run various RL exercises by loading Python config modules and supports:
1. train: Train the model, then play game and generate video
2. play: Only play game and generate video (no training)
3. push_to_hub: Push model to hub with optional play/video generation
"""

import argparse
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# below has to be imported after sys.path.insert(0, str(project_root))
from practice.base.context import ContextBase  # noqa: E402
from practice.utils.cli_utils import get_utc_time_str, load_config_module  # noqa: E402
from practice.utils.hub_utils import push_to_hub_generic  # noqa: E402
from practice.utils.play_utils_new import play_and_generate_video_generic  # noqa: E402
from practice.utils.train_utils_new import train_and_evaluate_network  # noqa: E402


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


def _close_context_envs(context: ContextBase) -> None:
    """Close all environments in the context."""
    try:
        context.train_env.close()
    except Exception as e:
        print(f"Warning: Failed to close training environment: {e}")

    try:
        context.eval_env.close()
    except Exception as e:
        print(f"Warning: Failed to close evaluation environment: {e}")


def main() -> None:
    """Main CLI function."""
    parser = _create_parser()
    args = parser.parse_args()

    # Validate arguments
    _validate_args(args, parser)

    # Load config and context/env from Python module based on mode
    print(f"Loading configuration from: {args.config}")
    config, context = load_config_module(args.config, args.mode)
    try:
        # Execute the requested mode
        time_str = f"[{get_utc_time_str()}] "
        if args.mode == "train":
            print(f"{time_str}=== Training Mode ===")
            train_and_evaluate_network(config=config, ctx=context)

        elif args.mode == "play":
            print(f"{time_str}=== Play Mode ===")
            play_and_generate_video_generic(config=config, ctx=context)

        elif args.mode == "push_to_hub":
            print(f"{time_str}=== Push to Hub Mode ===")
            # Environment cleanup is handled by the CLI try-finally block
            if not args.skip_play:
                print("Would play game and generate video first.")
                play_and_generate_video_generic(config=config, ctx=context)

            push_to_hub_generic(config=config, env=context.eval_env, username=args.username)
        print("=== Operation Complete ===")

    except Exception as e:
        print(f"Error occurs while run {args.mode} mode")
        raise e
    finally:
        _close_context_envs(context)


if __name__ == "__main__":
    main()
