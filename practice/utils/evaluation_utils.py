import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym

from hands_on.base import AgentBase
from hands_on.utils.env_utils import describe_wrappers
from hands_on.utils.evaluation_utils import evaluate_agent
from practice.base.config import BaseConfig


def evaluate_and_save_results(
    env: gym.Env[Any, Any],
    agent: AgentBase,
    config: BaseConfig,
) -> None:
    """Evaluate agent and save all results (model, training data, evaluation data, config).

    This function centralizes the evaluation and saving workflow that's common
    across all training scripts.

    Args:
        env: The environment for evaluation
        agent: The trained agent to evaluate
        cfg_data: Configuration dictionary containing eval_params and output_params
        train_result: Training results dictionary (must contain "episode_rewards")
    """

    # Perform evaluation
    mean_reward, std_reward = evaluate_agent(
        env=env,
        policy=agent,
        max_steps=None,
        episodes=config.eval_episodes,
        seed=config.eval_random_seed,
        record_video=config.eval_video_num is not None,
        video_dir=os.path.join(config.artifact_config.output_dir, "video"),
        video_num=config.eval_video_num,
    )

    # Create evaluation result dictionary
    eval_result = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    save_model_and_result(env=env, agent=agent, config=config, eval_result=eval_result)


def save_model_and_result(
    env: gym.Env[Any, Any],
    agent: AgentBase,
    config: BaseConfig,
    eval_result: dict[str, Any],
) -> None:
    """Save the model, parameters and the result to the JSON file."""
    artifact_config = config.artifact_config
    if not artifact_config.save_result:
        return

    # create the output directory
    out_dir = Path(artifact_config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # save the model
    agent.only_save_model(str(out_dir / artifact_config.model_filename))
    # save the eval result
    with open(out_dir / artifact_config.eval_result_filename, "w") as f:
        json.dump(eval_result, f)

    wrappers = describe_wrappers(env)
    # save all the config data
    config.save_to_json(
        filepath=str(out_dir / artifact_config.params_filename),
        with_data={"env_wrappers": wrappers},
    )
