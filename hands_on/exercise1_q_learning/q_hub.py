from typing import Any

import gymnasium as gym

from hands_on.utils.hub_play_utils import (
    get_env_name_and_metadata,
    push_model_to_hub,
)


def push_q_table_to_hub(
    username: str,
    cfg_data: dict[str, Any],
    env: gym.Env[Any, Any],
) -> None:
    """Push the Q-table to the Hub."""
    env_id = cfg_data["env_params"]["env_id"]
    hub_params = cfg_data["hub_params"]
    repo_id = f"{username}/{hub_params['repo_id']}"

    metadata = get_env_name_and_metadata(
        env_id=env_id, env=env, algorithm_name="q-learning"
    )

    model_card = f"""
    # **Q-Learning** Agent playing **{env_id}**
    This is a trained model of a **Q-Learning** agent playing **{env_id}** .

    ## Usage

    model = load_from_hub(repo_id="{repo_id}", filename="q-learning.pkl")

    # Don't forget to check if you need to add additional attributes (is_slippery=False etc)
    env = gym.make(model["env_id"])
    """

    push_model_to_hub(
        repo_id=repo_id,
        output_params=cfg_data["output_params"],
        model_card=model_card,
        metadata=metadata,
    )
