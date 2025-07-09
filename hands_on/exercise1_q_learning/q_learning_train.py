import argparse
from datetime import datetime
from typing import Any, TypeAlias

import gymnasium as gym
import numpy as np
import pickle5 as pickle
from numpy.typing import NDArray
from tqdm import tqdm

from hands_on.base import ActType, PolicyBase
from hands_on.exercise1_q_learning.config import QTableTrainConfig
from hands_on.utils.env_utils import make_discrete_env_with_kwargs
from hands_on.utils.evaluation_utils import evaluate_agent
from hands_on.utils.file_utils import (
    load_config_from_json,
    save_model_and_result,
)

# because q_table is use for the discrete action and observation space
ObsType: TypeAlias = np.int64


def greedy_policy(q_table: NDArray[np.float32], state: int) -> ActType:
    """Take the action with the highest state, action value.

    Args:
        q_table (NDArray[np.float32]): The Q-table.
        state (int): The current state.

    Returns:
        int: The action to take.
    """
    return np.argmax(q_table[state])


def epsilon_greedy_policy(
    q_table: NDArray[np.float32], state: int, epsilon: float
) -> ActType:
    """Take an action with the epsilon-greedy strategy.

    2 strategies:
    - Exploration: take a random action with probability epsilon.
    - Exploitation: take the action with the highest state, action value.

    Args:
        q_table (NDArray[np.float32]): The Q-table.
        state (int): The current state.
        epsilon (float): The exploration rate.

    Returns:
        int: The action to take.
    """
    if np.random.rand() < epsilon:
        return np.random.randint(q_table.shape[1], dtype=ActType)
    else:
        return np.argmax(q_table[state])


class QTable(PolicyBase):
    """Q-table."""

    def __init__(self, q_table: NDArray[np.float32]):
        self._q_table = q_table

        self._train_flag = False

    def set_train_flag(self, train_flag: bool) -> None:
        self._train_flag = train_flag

    def action(self, state: ObsType, epsilon: float | None = None) -> ActType:
        assert isinstance(state, int)
        if self._train_flag:
            assert epsilon is not None, (
                "epsilon must be provided in training mode"
            )
            return epsilon_greedy_policy(self._q_table, state, epsilon)
        else:
            return greedy_policy(self._q_table, state)

    def get_score(self, state: ObsType, action: ActType | None = None) -> float:
        assert isinstance(state, int)
        if action is None:
            return float(max(self._q_table[state]))
        else:
            return float(self._q_table[state, action])

    def update(
        self,
        state: ObsType | None,
        action: ActType | None,
        reward_target: float,
    ) -> None:
        assert state is not None
        assert action is not None
        self._q_table[state, action] = reward_target

    def only_save_model(self, pathname: str) -> None:
        """Save the Q-table to a file."""
        assert pathname.endswith(".pkl")
        with open(pathname, "wb") as f:
            pickle.dump(self._q_table, f)

    @classmethod
    def load_for_evaluation(cls, pathname: str) -> "QTable":
        """Load the Q-table from a file."""
        assert pathname.endswith(".pkl")
        with open(pathname, "rb") as f:
            q_table = pickle.load(f)
        return cls(q_table=q_table)


def q_table_train_loop(
    env: gym.Env[ObsType, ActType],
    q_table: QTable,
    q_config: QTableTrainConfig,
) -> dict[str, Any]:
    """Train the Q-table.

    For each episode:
    - Reduce epsilon (since we need less and less exploration)
    - Reset the environment

    For step in max timesteps:
    - Choose the action At using epsilon greedy policy
    - Take the action (a) and observe the outcome state(s') and reward (r)
    - Update the Q-value Q(s,a) using Bellman equation Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
    - If done, finish the episode
    - Our next state is the new state

    Args:
        env (gym.Env): The environment.
        q_table (QTable): The Q-table.
        q_config (QTableTrainConfig): The training configuration.

    Returns:
        dict[str, Any]: The training metadata.
    """
    q_table.set_train_flag(train_flag=True)
    episode_rewards = []
    for episode in tqdm(range(q_config.episodes)):
        rewards = 0.0
        epsilon = q_config.min_epsilon + (
            q_config.max_epsilon - q_config.min_epsilon
        ) * np.exp(-q_config.decay_rate * episode)
        state, _ = env.reset()

        for _ in range(q_config.max_steps):
            action = q_table.action(state=state, epsilon=epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            rewards += float(reward)
            old_score = q_table.get_score(state=state, action=action)
            new_score = old_score + q_config.learning_rate * (
                float(reward)
                + q_config.gamma
                * q_table.get_score(state=next_state, action=None)
                - old_score
            )
            q_table.update(state=state, action=action, reward_target=new_score)

            if terminated or truncated:
                break
            state = next_state
        episode_rewards.append(rewards)

    return {"episode_rewards": episode_rewards}


def q_table_main(
    env: gym.Env[ObsType, ActType],
    cfg_data: dict[str, Any],
) -> None:
    """Main Q-table training function that uses configuration data."""
    # Load or create Q-table
    checkpoint_pathname = cfg_data.get("checkpoint_pathname", None)
    if checkpoint_pathname:
        q_table = QTable.load_for_evaluation(checkpoint_pathname)
    else:
        obs_space = env.observation_space
        act_space = env.action_space
        assert isinstance(obs_space, gym.spaces.Discrete)
        assert isinstance(act_space, gym.spaces.Discrete)
        q_table = QTable(
            q_table=np.zeros((obs_space.n, act_space.n), dtype=np.float32)
        )

    # Train the Q-table using config parameters
    train_data = q_table_train_loop(
        env=env,
        q_table=q_table,
        q_config=QTableTrainConfig.from_dict(cfg_data["hyper_params"]),
    )

    # evaluate the agent
    eval_params = cfg_data["eval_params"]
    mean_reward, std_reward = evaluate_agent(
        env=env,
        policy=q_table,
        max_steps=eval_params["max_steps"],
        episodes=eval_params["eval_episodes"],
        seed=eval_params["eval_seed"],
    )
    print(f"{mean_reward=}")

    # save the result
    save_result = cfg_data["output_params"].get("save_result", False)
    if save_result:
        eval_result = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "eval_episodes": eval_params["eval_episodes"],
            "eval_seed": eval_params["eval_seed"],
            "max_steps": eval_params["max_steps"],
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_model_and_result(cfg_data, train_data, eval_result, agent=q_table)


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
        q_table_main(env=env, cfg_data=cfg_data)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg_data = load_config_from_json(args.config)
    main(cfg_data=cfg_data)
