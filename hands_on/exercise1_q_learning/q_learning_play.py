import argparse
from typing import Any

import gymnasium as gym

from common.evaluation_utils import play_game_once
from common.hub_utils import push_to_hub
from hands_on.exercise1_q_learning.q_learning_train import (
    EXERCISE1_RESULT_DIR,
    QTable,
)


def push_q_table_to_hub(
    username: str,
    repo_name: str,
    env: gym.Env[Any, Any],
    env_id: str,
) -> None:
    """Push the Q-table to the Hub."""
    repo_id = f"{username}/{repo_name}"
    model_card = f"""
    # **Q-Learning** Agent playing1 **{env_id}**
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

    push_to_hub(
        repo_id=repo_id,
        model_card=model_card,
        model_pathname=str(EXERCISE1_RESULT_DIR / "q_table.pkl"),
        hyperparameters_pathname=str(
            EXERCISE1_RESULT_DIR / "hyperparameters.json"
        ),
        eval_result_pathname=str(EXERCISE1_RESULT_DIR / "eval_result.json"),
        video_pathname=str(EXERCISE1_RESULT_DIR / "replay.mp4"),
        metadata=metadata,
        local_repo_path=str(EXERCISE1_RESULT_DIR / "hub"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    # add a --push_to_hub argument if provided
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    parser.add_argument("--skip_render", action="store_true", default=False)
    parser.add_argument("--username", type=str, default="")
    args = parser.parse_args()

    if args.push_to_hub:
        assert args.username != "", "Username is required when pushing to Hub"

    env_id = "FrozenLake-v1"
    env = gym.make(
        env_id,
        map_name="4x4",
        is_slippery=False,
        render_mode="rgb_array",
    )
    repo_name = "q-FrozenLake-v1-4x4-noSlippery"
    # env_id = "Taxi-v3"
    # env = gym.make(env_id, render_mode="rgb_array",)
    # repo_name = "q-Taxi-v3"
    if not args.skip_render:
        # load the q_table from the file
        q_table = QTable.load(str(EXERCISE1_RESULT_DIR / "q_table.pkl"))
        # play the game
        try:
            play_game_once(
                env=env,
                policy=q_table,
                save_video=True,
                video_pathname=str(EXERCISE1_RESULT_DIR / "replay.mp4"),
            )
        finally:
            env.close()

    if args.push_to_hub:
        push_q_table_to_hub(
            username=args.username,
            repo_name=repo_name,
            env=env,
            env_id=env_id,
        )


if __name__ == "__main__":
    main()
