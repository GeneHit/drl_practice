from typing import Any

import gymnasium as gym

from hands_on.utils.hub_play_utils import (
    get_env_name_and_metadata,
    push_model_to_hub,
)


def push_reinforce_to_hub(
    username: str,
    cfg_data: dict[str, Any],
    env: gym.Env[Any, Any],
) -> None:
    """Push the REINFORCE model to the Hub."""
    env_id = cfg_data["env_params"]["env_id"]
    metadata = get_env_name_and_metadata(
        env_id=env_id,
        env=env,
        algorithm_name="reinforce",
        extra_tags=["policy-gradient", "pytorch"],
    )

    repo_id = f"{username}/{cfg_data['hub_params']['repo_id']}"

    model_card = f"""
    # **REINFORCE** Agent playing **{env_id}**
    This is a trained model of a **REINFORCE** agent playing **{env_id}**.

    ## Usage

    model = load_from_hub(repo_id="{repo_id}", filename="reinforce.pth")

    # Don't forget to check if you need to add additional wrapper to the
    # environment for the observation.
    env = gym.make(model["env_id"])
    ...
    """

    push_model_to_hub(
        repo_id=repo_id,
        output_params=cfg_data["output_params"],
        model_card=model_card,
        metadata=metadata,
    )
