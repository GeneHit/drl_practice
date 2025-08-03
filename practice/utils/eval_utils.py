import json
import os
import random
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from practice.base.config import BaseConfig
from practice.exercise1_q.q_table_exercise import QTable
from practice.utils.cli_utils import get_utc_time_str
from practice.utils.dist_utils import unwrap_model
from practice.utils.env_utils import dump_env_wrappers
from practice.utils_for_coding.network_utils import save_model
from practice.utils_for_coding.numpy_tensor_utils import get_tensor_expanding_axis


def evaluate_and_save_results(
    env: gym.Env[Any, Any],
    agent: nn.Module | QTable,
    config: BaseConfig,
    meta_data: dict[str, Any] = {},
) -> None:
    """Evaluate agent and save all results (model, training data, evaluation data, config).

    This function centralizes the evaluation and saving workflow that's common
    across all training scripts.

    Parameters
    ----------
        env : gym.Env
            The environment for evaluation
        agent : nn.Module | QTable
            The trained agent to evaluate
        config : BaseConfig
            Configuration dictionary containing eval_params and output_params
        meta_data : dict[str, Any]
            Extra meta data to save
    """

    # Perform evaluation
    mean_reward, std_reward = _evaluate_agent(
        env=env,
        agent=agent,
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
        "datetime": get_utc_time_str(),
        **meta_data,
    }

    _save_model_and_result(env=env, agent=agent, config=config, eval_result=eval_result)


def get_action(agent: nn.Module | QTable, state: Any) -> Any:
    """Get the action from the agent.

    Parameters
    ----------
        agent : nn.Module | QTable
            The agent.
        state : Any
            The state of the environment. Type: numpy array or np.integer.
        device : torch.device
            The device to run the agent on.

    Returns
    -------
        Any: The action from the agent. Type: np.int64 (discrete) or numpy array float (continuous).
    """
    if isinstance(agent, nn.Module):
        assert isinstance(state, np.ndarray)
        # get the device from the agent
        device = next(agent.parameters()).device
        state_tensor = get_tensor_expanding_axis(state).to(device)
        with torch.no_grad():
            return unwrap_model(agent).action(state_tensor)  # type: ignore

    return agent.action(state)


def _evaluate_agent(
    env: gym.Env[Any, Any],
    agent: nn.Module | QTable,
    episodes: int,
    seed: int | None,
    record_video: bool = False,
    video_dir: str = "./video",
    video_num: int | None = None,
) -> tuple[float, float]:
    """Evaluate the agent.

    Parameters
    ----------
        env : gym.Env
            The environment.
        agent : nn.Module | QTable
            The agent. When using nn.Module, it has to have a method `action` that takes a
            Tensor and returns a Tensor.
        episodes : int
            The number of episodes to evaluate.
        seed : int | None
            The seed for the random number generator.
        record_video : bool
            Whether to record videos during evaluation.
        video_dir : str
            The directory to save the videos.
        video_num : int | None
            The number of videos to record.

    Returns
    -------
        tuple[float, float]:
            The average and std of the reward.
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

        while True:
            # the network has to decide the returned type of action
            action = get_action(agent, state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_rewards_ep += float(reward)
            if terminated or truncated:
                break

        rewards.append(total_rewards_ep)

    return float(np.mean(rewards)), float(np.std(rewards))


def _save_model_and_result(
    env: gym.Env[Any, Any],
    agent: nn.Module | QTable,
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
    path = str(out_dir / artifact_config.model_filename)
    if isinstance(agent, QTable):
        # save the q table
        agent.only_save_model(path)
    else:
        assert isinstance(agent, nn.Module)
        # save the full model
        save_model(agent, path, full_model=True)
        # save the state dict
        path = str(out_dir / artifact_config.state_dict_filename)
        save_model(agent, path, full_model=False)

    # save the eval result
    with open(out_dir / artifact_config.eval_result_filename, "w") as f:
        json.dump(eval_result, f, indent=4)

    # save the env setup
    env_setup = {"user_config": config.to_dict()["env_config"], **dump_env_wrappers(env)}
    with open(out_dir / artifact_config.env_setup_filename, "w") as f:
        json.dump(env_setup, f, indent=4)

    # save all the config data
    config.save_to_json(filepath=str(out_dir / artifact_config.params_filename))
