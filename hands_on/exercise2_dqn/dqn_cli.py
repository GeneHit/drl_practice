"""Main DQN script with multiple modes of operation.

This script supports different modes:
1. train (default): Train the model, then play game and generate video
2. push_to_hub: Push model to hub, optionally play game and generate video
3. play_only: Only play game and generate video (no training or hub operations)
"""

from typing import Any

from hands_on.exercise2_dqn.dqn_train import main as dqn_train
from hands_on.utils.agent_utils import NNAgent
from hands_on.utils.cli_utils import (
    create_env_from_config,
    create_main_function,
    play_and_generate_video_generic,
    push_to_hub_generic,
)
from hands_on.utils.env_utils import get_device


def push_to_hub_wrapper(cfg_data: dict[str, Any], username: str) -> None:
    """Wrapper for DQN hub push to match expected signature."""
    push_to_hub_generic(
        cfg_data=cfg_data,
        username=username,
        algorithm_name="DQN",
        model_filename="dqn.pth",
        extra_tags=["deep-q-learning", "pytorch"],
        usage_instructions="""# Don't forget to check if you need to add additional wrapper to the
    # environment for the image observation.
    env = gym.wrappers.AddRenderObservation(env, render_only=True)""",
    )


def play_and_generate_video(cfg_data: dict[str, Any]) -> None:
    """Play the game using trained DQN model and generate video."""
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
    script_name="dqn_main.py",
    algorithm_name="DQN",
    train_fn=dqn_train,
    push_to_hub_fn=push_to_hub_wrapper,
    play_and_generate_video_fn=play_and_generate_video,
    config_example="config.json",
)


if __name__ == "__main__":
    main()
