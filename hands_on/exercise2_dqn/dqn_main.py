"""Main DQN script with multiple modes of operation.

This script supports different modes:
1. train (default): Train the model, then play game and generate video
2. push_to_hub: Push model to hub, optionally play game and generate video
3. play_only: Only play game and generate video (no training or hub operations)
"""

import argparse
from pathlib import Path
from typing import Any

from hands_on.exercise2_dqn.dqn_hub import (
    create_dqn_agent_from_config,
    push_dqn_to_hub,
)
from hands_on.exercise2_dqn.dqn_train_1 import main as train_main
from hands_on.utils.env_utils import make_1d_env, make_image_env
from hands_on.utils.evaluation_utils import play_game_once
from hands_on.utils.file_utils import load_config_from_json


def create_env_from_config(
    env_params: dict[str, Any], render_mode: str = "rgb_array"
) -> Any:
    if env_params.get("use_image", False):
        env, _ = make_image_env(
            env_id=env_params["env_id"],
            render_mode=render_mode,
            resize_shape=tuple(env_params["resize_shape"]),
            frame_stack_size=env_params["frame_stack_size"],
        )
    else:
        env, _ = make_1d_env(
            env_id=env_params["env_id"], render_mode=render_mode
        )

    return env


def train_mode(cfg_data: dict[str, Any], skip_play: bool = False) -> None:
    """Train the DQN model, optionally play game and generate video."""
    print("=== Training DQN Model ===")
    # Run the training
    train_main(cfg_data=cfg_data)

    if not skip_play:
        print("=== Training Complete, Playing Game and Generating Video ===")
        # After training, play the game and generate video
        play_and_generate_video(cfg_data=cfg_data)
    else:
        print("=== Training Complete (Skipping Play/Video Generation) ===")


def push_to_hub_mode(
    cfg_data: dict[str, Any], username: str, skip_play: bool = False
) -> None:
    """Push model to hub, optionally play game and generate video."""
    if not skip_play:
        print("=== Playing Game and Generating Video ===")
        play_and_generate_video(cfg_data=cfg_data)

    print("=== Pushing Model to Hub ===")
    # Create environment for hub operations
    env = create_env_from_config(cfg_data["env_params"])

    try:
        push_dqn_to_hub(username=username, cfg_data=cfg_data, env=env)
        print(f"Model successfully pushed to hub for user: {username}")
    finally:
        env.close()


def play_only_mode(cfg_data: dict[str, Any]) -> None:
    """Only play game and generate video."""
    print("=== Playing Game and Generating Video ===")
    play_and_generate_video(cfg_data=cfg_data)


def play_and_generate_video(cfg_data: dict[str, Any]) -> None:
    """Play the game using trained model and generate video."""
    # Create environment from config
    env = create_env_from_config(cfg_data["env_params"])

    # Load the trained DQN model
    output_params = cfg_data["output_params"]
    output_dir = Path(output_params["output_dir"])
    model_path = output_dir / output_params["model_filename"]

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    dqn_agent = create_dqn_agent_from_config(model_path)

    # Play the game and save video
    video_path = output_dir / output_params.get(
        "replay_video_filename", "replay.mp4"
    )

    try:
        play_game_once(
            env=env,
            policy=dqn_agent,
            save_video=True,
            video_pathname=str(video_path),
            fps=10,
            seed=10,
        )
        print(f"Game replay saved to: {video_path}")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DQN Main Script with Multiple Modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Mode descriptions:
            train (default): Train the DQN model, then play game and generate video
            push_to_hub: Push model to hub, optionally play game and generate video
            play_only: Only play game and generate video (no training or hub operations)

            Examples:
            # Train mode (default) - train model then play game and generate video
            python dqn_main.py --config config.json

            # Train mode without play/video generation
            python dqn_main.py --config config.json --skip_play

            # Push to hub with play/video
            python dqn_main.py --config config.json --mode push_to_hub --username myuser

            # Push to hub without play/video
            python dqn_main.py --config config.json --mode push_to_hub --username myuser --skip_play

            # Only play and generate video
            python dqn_main.py --config config.json --mode play_only
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration JSON file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "push_to_hub", "play_only"],
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
        help="Skip playing the game and video generation",
    )

    args = parser.parse_args()

    # Load config using the provided config path
    cfg_data = load_config_from_json(args.config)

    # Validate arguments based on mode
    if args.mode == "push_to_hub" and not args.username:
        parser.error("--username is required when using push_to_hub mode")

    if args.skip_play and args.mode not in ["push_to_hub", "train"]:
        parser.error(
            "--skip_play can only be used with push_to_hub or train mode"
        )

    # Execute based on mode
    if args.mode == "train":
        train_mode(cfg_data, args.skip_play)
    elif args.mode == "push_to_hub":
        push_to_hub_mode(cfg_data, args.username, args.skip_play)
    elif args.mode == "play_only":
        play_only_mode(cfg_data)


if __name__ == "__main__":
    main()
