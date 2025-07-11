import argparse
from typing import Any

import gymnasium as gym
import numpy as np
import pickle5 as pickle
from numpy.typing import NDArray

from hands_on.base import ActType
from hands_on.exercise1_q_learning.config import QTableTrainConfig
from hands_on.exercise1_q_learning.q_exercise import (
    ObsType,
    QTable,
    q_table_train_loop,
)
from hands_on.utils.env_utils import make_discrete_env_with_kwargs
from hands_on.utils.evaluation_utils import evaluate_and_save_results
from hands_on.utils.file_utils import (
    load_config_from_json,
)


def q_table_train(
    env: gym.Env[ObsType, ActType],
    cfg_data: dict[str, Any],
) -> None:
    """Main Q-table training function that uses configuration data."""
    # Load or create Q-table
    checkpoint_pathname = cfg_data.get("checkpoint_pathname", None)
    if checkpoint_pathname:
        with open(checkpoint_pathname, "rb") as f:
            q_table: NDArray[np.float32] = pickle.load(f)
    else:
        obs_space = env.observation_space
        act_space = env.action_space
        assert isinstance(obs_space, gym.spaces.Discrete)
        assert isinstance(act_space, gym.spaces.Discrete)
        q_table = np.zeros((obs_space.n, act_space.n), dtype=np.float32)

    # Train the Q-table using config parameters
    train_data = q_table_train_loop(
        env=env,
        q_table=q_table,
        q_config=QTableTrainConfig.from_dict(cfg_data["hyper_params"]),
    )
    assert "episode_rewards" in train_data, "episode_rewards must be in train_data"

    # Create agent and evaluate/save results
    q_agent = QTable(q_table=q_table)
    evaluate_and_save_results(
        env=env,
        agent=q_agent,
        cfg_data=cfg_data,
        train_result=train_data,
    )


def main(cfg_data: dict[str, Any]) -> None:
    """Main function that creates environment and calls training."""
    # make the environment by the config
    env, env_info = make_discrete_env_with_kwargs(
        env_id=cfg_data["env_params"]["env_id"],
        kwargs=cfg_data["env_params"]["kwargs"],
    )
    # Update the original cfg_data with the new environment parameters
    cfg_data["env_params"].update(env_info)

    try:
        q_table_train(env=env, cfg_data=cfg_data)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg_data = load_config_from_json(args.config)
    main(cfg_data=cfg_data)
