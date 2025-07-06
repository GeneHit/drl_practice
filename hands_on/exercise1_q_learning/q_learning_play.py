import argparse

import gymnasium as gym

from common.evaluation_utils import play_game_once
from common.hub_utils import push_to_hub
from hands_on.exercise1_q_learning.q_learning_train import (
    EXERCISE1_RESULT_DIR,
    QTable,
)

ENV_ID = "FrozenLake-v1"


def push_q_table_to_hub(username: str) -> None:
    """Push the Q-table to the Hub."""
    repo_name = "q-FrozenLake-v1-4x4-noSlippery"
    repo_id = f"{username}/{repo_name}"
    model_card = f"""
    # **Q-Learning** Agent playing1 **{ENV_ID}**
    This is a trained model of a **Q-Learning** agent playing **{ENV_ID}** .

    ## Usage

    model = load_from_hub(repo_id="{repo_id}", filename="q-learning.pkl")

    # Don't forget to check if you need to add additional attributes (is_slippery=False etc)
    env = gym.make(model["env_id"])
    """
    env_name = "FrozenLake-v1-4x4-noSlippery"
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

    if not args.skip_render:
        # load the q_table from the file
        q_table = QTable.load(str(EXERCISE1_RESULT_DIR / "q_table.pkl"))
        env = gym.make(
            "FrozenLake-v1",
            map_name="4x4",
            is_slippery=False,
            render_mode="rgb_array",
        )

        # play the game
        play_game_once(
            env=env,
            policy=q_table,
            save_video=True,
            video_pathname=str(EXERCISE1_RESULT_DIR / "replay.mp4"),
        )

    if args.push_to_hub:
        push_q_table_to_hub(args.username)


if __name__ == "__main__":
    main()
