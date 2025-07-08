import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym

from hands_on.exercise1_q_learning.q_learning_train import QTable
from hands_on.utils.config_utils import load_config_from_json
from hands_on.utils.env_utils import make_discrete_env_with_kwargs
from hands_on.utils.evaluation_utils import play_game_once
from hands_on.utils.hub_utils import push_to_hub


def push_q_table_to_hub(
    username: str,
    cfg_data: dict[str, Any],
    env: gym.Env[Any, Any],
) -> None:
    """Push the Q-table to the Hub."""
    hub_params = cfg_data["hub_params"]
    repo_id = f"{username}/{hub_params['repo_id']}"
    env_id = cfg_data["env_params"]["env_id"]

    model_card = f"""
    # **Q-Learning** Agent playing **{env_id}**
    This is a trained model of a **Q-Learning** agent playing **{env_id}** .

    ## Usage

    model = load_from_hub(repo_id="{repo_id}", filename="q-learning.pkl")

    # Don't forget to check if you need to add additional attributes (is_slippery=False etc)
    env = gym.make(model["env_id"])
    """
    env_name = env_id
    if env.spec is not None:
        map_name = env.spec.kwargs.get("map_name")
        if map_name:
            env_name = env_name + "-" + map_name
        is_slippery = env.spec.kwargs.get("is_slippery", True)
        if not is_slippery:
            env_name += "-noSlippery"

    metadata = {
        "env_name": env_name,
        "tags": [
            env_name,
            "q-learning",
            "reinforcement-learning",
            "custom-implementation",
        ],
    }

    output_params = cfg_data["output_params"]
    output_dir = Path(output_params["output_dir"])

    push_to_hub(
        repo_id=repo_id,
        model_card=model_card,
        file_pathnames=[
            str(output_dir / output_params["model_filename"]),
            str(output_dir / output_params["params_filename"]),
            str(output_dir / output_params["train_result_filename"]),
            str(
                output_dir
                / output_params.get("replay_video_filename", "replay.mp4")
            ),
        ],
        eval_result_pathname=str(
            output_dir / output_params["eval_result_filename"]
        ),
        metadata=metadata,
        local_repo_path=str(output_dir / "hub"),
    )


def main(cfg_data: dict[str, Any], args: argparse.Namespace) -> None:
    """Main function that loads config and handles play/hub operations."""
    # Create environment from config
    env, _ = make_discrete_env_with_kwargs(
        env_id=cfg_data["env_params"]["env_id"],
        kwargs=cfg_data["env_params"]["kwargs"],
    )

    # Load the trained Q-table
    output_params = cfg_data["output_params"]
    output_dir = Path(output_params["output_dir"])
    model_path = output_dir / output_params["model_filename"]

    if not args.skip_play:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        q_table = QTable.load(str(model_path))

        # Play the game and save video
        video_filename = output_params.get(
            "replay_video_filename", "replay.mp4"
        )
        video_path = output_dir / video_filename

        try:
            play_game_once(
                env=env,
                policy=q_table,
                save_video=True,
                video_pathname=str(video_path),
            )
            print(f"Game replay saved to: {video_path}")
        finally:
            env.close()

    if args.push_to_hub:
        if not args.username:
            raise ValueError("Username is required when pushing to Hub")
        push_q_table_to_hub(
            username=args.username,
            cfg_data=cfg_data,
            env=env,
        )
        print(
            f"Model pushed to hub: {args.username}/{cfg_data['hub_params']['repo_id']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Play Q-learning agent and optionally push to hub"
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
