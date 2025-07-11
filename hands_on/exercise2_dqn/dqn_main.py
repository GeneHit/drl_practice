"""Main DQN script with multiple modes of operation.

This script supports different modes:
1. train (default): Train the model, then play game and generate video
2. push_to_hub: Push model to hub, optionally play game and generate video
3. play_only: Only play game and generate video (no training or hub operations)
"""

from typing import Any

from hands_on.exercise2_dqn.dqn_exercise import DQNAgent
from hands_on.exercise2_dqn.dqn_hub import push_dqn_to_hub
from hands_on.exercise2_dqn.dqn_train import main as train_main
from hands_on.utils.cli_utils import (
    create_env_from_config,
    create_main_function,
    play_and_generate_video_generic,
)
from hands_on.utils.env_utils import get_device


def push_to_hub_wrapper(cfg_data: dict[str, Any], username: str) -> None:
    """Wrapper for push_dqn_to_hub to match expected signature."""
    env = create_env_from_config(cfg_data["env_params"])
    try:
        push_dqn_to_hub(username=username, cfg_data=cfg_data, env=env)
    finally:
        env.close()


def play_and_generate_video(cfg_data: dict[str, Any]) -> None:
    """Play the game using trained DQN model and generate video."""
    play_and_generate_video_generic(
        cfg_data=cfg_data,
        env_creator=create_env_from_config,
        model_loader=DQNAgent,
        device=get_device(),
        fps=10,
        seed=99,
    )


# Create the main function using the CLI utilities
main = create_main_function(
    script_name="dqn_main.py",
    algorithm_name="DQN",
    train_fn=train_main,
    push_to_hub_fn=push_to_hub_wrapper,
    play_and_generate_video_fn=play_and_generate_video,
    config_example="config.json",
)


if __name__ == "__main__":
    main()
