import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch

from hands_on.exercise2_dqn.dqn_train import DQNAgent
from hands_on.utils.env_utils import get_device, make_1d_env, make_image_env
from hands_on.utils.evaluation_utils import play_game_once
from hands_on.utils.file_utils import load_config_from_json
from hands_on.utils.hub_play_utils import (
    get_env_name_and_metadata,
    push_model_to_hub,
)


def create_dqn_agent_from_config(model_path: Path) -> DQNAgent:
    """Create and load a DQN agent from config and model file."""
    # Set device
    device = get_device()

    # Load the trained model directly
    q_network = torch.load(model_path, map_location=device, weights_only=False)
    q_network = q_network.to(device)

    # Create DQN agent with minimal requirements for inference
    # We use a dummy optimizer since it won't be used for inference
    dqn_agent = DQNAgent(q_network=q_network, optimizer=None, action_n=None)
    dqn_agent.set_train_flag(False)  # Set to evaluation mode

    return dqn_agent


def push_dqn_to_hub(
    username: str,
    cfg_data: dict[str, Any],
    env: gym.Env[Any, Any],
) -> None:
    """Push the DQN model to the Hub."""
    env_id = cfg_data["env_params"]["env_id"]
    metadata = get_env_name_and_metadata(
        env_id=env_id,
        env=env,
        algorithm_name="dqn",
        extra_tags=["deep-q-learning", "pytorch"],
    )

    repo_id = f"{username}/{cfg_data['hub_params']['repo_id']}"

    model_card = f"""
    # **DQN** Agent playing **{env_id}**
    This is a trained model of a **DQN** agent playing **{env_id}**.

    ## Usage

    model = load_from_hub(repo_id="{repo_id}", filename="dqn.pth")

    # Don't forget to check if you need to add additional wrapper to the
    # environment for the image observation.
    env = gym.make(model["env_id"])
    env = gym.wrappers.AddRenderObservation(env, render_only=True)
    ...
    """

    push_model_to_hub(
        repo_id=repo_id,
        output_params=cfg_data["output_params"],
        model_card=model_card,
        metadata=metadata,
    )


def main(cfg_data: dict[str, Any], args: argparse.Namespace) -> None:
    """Main function that loads config and handles play/hub operations."""
    # Create environment from config
    env_params = cfg_data["env_params"]
    if env_params.get("use_image", False):
        env, _ = make_image_env(
            env_id=env_params["env_id"],
            render_mode="rgb_array",  # use rgb_array for video recording
            resize_shape=tuple(env_params["resize_shape"]),
            frame_stack_size=env_params["frame_stack_size"],
        )
    else:
        env, _ = make_1d_env(
            env_id=env_params["env_id"], render_mode="rgb_array"
        )

    if not args.skip_play:
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

    if args.push_to_hub:
        if not args.username:
            raise ValueError("Username is required when pushing to Hub")
        push_dqn_to_hub(username=args.username, cfg_data=cfg_data, env=env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Play DQN agent and optionally push to hub"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration JSON file",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        default=False,
        help="Push model to Hugging Face Hub",
    )
    parser.add_argument(
        "--username",
        type=str,
        default="",
        help="Hugging Face username (required for --push_to_hub)",
    )
    parser.add_argument(
        "--skip_play",
        action="store_true",
        default=False,
        help="Skip playing the game",
    )
    args = parser.parse_args()

    cfg_data = load_config_from_json(args.config)
    main(cfg_data=cfg_data, args=args)
