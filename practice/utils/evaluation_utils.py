import json
import os
import random
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from practice.base.chest import AgentBase
from practice.base.config import BaseConfig
from practice.utils.cli_utils import get_utc_time_str
from practice.utils.env_utils import dump_env_wrappers


def evaluate_agent(
    env: gym.Env[Any, Any],
    policy: AgentBase,
    max_steps: int | None,
    episodes: int,
    seed: int | None,
    record_video: bool = False,
    video_dir: str = "./video",
    video_num: int | None = None,
) -> tuple[float, float]:
    """Evaluate the agent.

    Args:
        env (gym.Env): The environment.
        policy (AgentBase): The policy.
        max_steps (int): The maximum number of steps per episode.
        episodes (int): The number of episodes to evaluate.
        seed (Sequence[int]): The seed.
        record_video (bool): Whether to record videos during evaluation.
        video_dir (str): The directory to save the videos.

    Returns:
        tuple[float, float]: The average and std of the reward.
    """
    # Wrap the environment with RecordVideo if video recording is requested
    if record_video:
        # Create the video folder if it doesn't exist
        os.makedirs(video_dir, exist_ok=True)
        num = video_num if video_num is not None else 10
        trigger_step = episodes // num
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda x: x % trigger_step == 0,
            disable_logger=True,
        )

    rewards = []
    rnd = random.Random(seed)
    seeks = [rnd.randint(0, 1000) for _ in range(episodes)]
    for episode in tqdm(range(episodes), desc="Evaluating"):
        state, _ = env.reset(seed=seeks[episode])

        total_rewards_ep = 0.0

        def step() -> bool:
            nonlocal state, total_rewards_ep
            action = policy.action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_rewards_ep += float(reward)
            return terminated or truncated

        if max_steps is None:
            while not step():
                pass
        else:
            for _ in range(max_steps):
                if step():
                    break

        rewards.append(total_rewards_ep)

    return float(np.mean(rewards)), float(np.std(rewards))


def evaluate_and_save_results(
    env: gym.Env[Any, Any],
    agent: AgentBase,
    config: BaseConfig,
    meta_data: dict[str, Any] = {},
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
        "mean_reward": f"{mean_reward:.2f}",
        "std_reward": f"{std_reward:.2f}",
        "datetime": get_utc_time_str(),
        **meta_data,
    }

    _save_model_and_result(env=env, agent=agent, config=config, eval_result=eval_result)


def _save_model_and_result(
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
        json.dump(eval_result, f, indent=4)

    # save the env setup
    env_setup = {"user_config": config.to_dict()["env_config"], **dump_env_wrappers(env)}
    with open(out_dir / artifact_config.env_setup_filename, "w") as f:
        json.dump(env_setup, f, indent=4)

    # save all the config data
    config.save_to_json(filepath=str(out_dir / artifact_config.params_filename))
