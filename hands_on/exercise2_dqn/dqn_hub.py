from typing import Any

import gymnasium as gym

from hands_on.utils.hub_play_utils import (
    get_env_name_and_metadata,
    push_model_to_hub,
)


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
