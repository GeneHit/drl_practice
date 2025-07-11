"""Main REINFORCE script with multiple modes of operation.

This script supports different modes:
1. train (default): Train the model, then play game and generate video
2. push_to_hub: Push model to hub, optionally play game and generate video
3. play_only: Only play game and generate video (no training or hub operations)
"""

from typing import Any

from hands_on.exercise3_reinforce.reinforce_train import (
    reinforce_main as train_main,
)
from hands_on.utils.agent_utils import NNAgent
from hands_on.utils.cli_utils import (
    create_env_from_config,
    create_main_function,
    play_and_generate_video_generic,
    push_to_hub_generic,
)
from hands_on.utils.env_utils import get_device


def push_to_hub_wrapper(cfg_data: dict[str, Any], username: str) -> None:
    """Wrapper for REINFORCE hub push to match expected signature."""
    push_to_hub_generic(
        cfg_data=cfg_data,
        username=username,
        algorithm_name="REINFORCE",
        model_filename="reinforce.pth",
        extra_tags=["policy-gradient", "pytorch"],
        usage_instructions="# Don't forget to check if you need to add additional wrapper to the\n    # environment for the observation.",
    )


def play_and_generate_video(cfg_data: dict[str, Any]) -> None:
    """Play the game using trained REINFORCE model and generate video."""
    play_and_generate_video_generic(
        cfg_data=cfg_data,
        env_creator=create_env_from_config,
        model_loader=NNAgent,
        device=get_device(),
        fps=10,
        seed=99,
    )


# Create the main function using the CLI utilities
main = create_main_function(
    script_name="reinforce_main.py",
    algorithm_name="REINFORCE",
    train_fn=train_main,
    push_to_hub_fn=push_to_hub_wrapper,
    play_and_generate_video_fn=play_and_generate_video,
    config_example="cartpole_config.json",
)


if __name__ == "__main__":
    main()
