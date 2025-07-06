import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pickle5 as pickle
from numpy.typing import NDArray
from tqdm import tqdm

from common.base import PolicyBase
from common.evaluation_utils import evaluate_agent

EXERCISE1_RESULT_DIR = Path("results/exercise1_q_learning/")


def greedy_policy(q_table: NDArray[np.float32], state: int) -> int:
    """Take the action with the highest state, action value.

    Args:
        q_table (NDArray[np.float32]): The Q-table.
        state (int): The current state.

    Returns:
        int: The action to take.
    """
    return int(np.argmax(q_table[state]))


def epsilon_greedy_policy(
    q_table: NDArray[np.float32], state: int, epsilon: float
) -> int:
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
        return int(np.random.randint(q_table.shape[1]))
    else:
        return int(np.argmax(q_table[state]))


@dataclass(kw_only=True, frozen=True)
class QTableConfig:
    state_space: int | None
    action_space: int | None
    q_table: NDArray[np.float32] | None = None


class QTable(PolicyBase):
    """Q-table."""

    def __init__(self, config: QTableConfig):
        self._config = config
        if config.q_table is None:
            assert (
                config.state_space is not None
                and config.action_space is not None
            )
            self._q_table: NDArray[Any] = np.zeros(
                (config.state_space, config.action_space)
            )
        else:
            self._q_table = config.q_table

        self._train_flag = False

    def set_train_flag(self, train_flag: bool) -> None:
        self._train_flag = train_flag

    def action(self, state: int, epsilon: float | None = None) -> int:
        assert isinstance(state, int)
        if self._train_flag:
            assert epsilon is not None, (
                "epsilon must be provided in training mode"
            )
            return epsilon_greedy_policy(self._q_table, state, epsilon)
        else:
            return greedy_policy(self._q_table, state)

    def get_score(self, state: int, action: int | None = None) -> float:
        assert isinstance(state, int)
        if action is None:
            return float(max(self._q_table[state]))
        else:
            return float(self._q_table[state, action])

    def update(self, state: int | None, action: int | None, score: Any) -> None:
        assert state is not None
        assert action is not None
        assert isinstance(score, float)

        self._q_table[state, action] = score

    def save(self, pathname: str) -> None:
        """Save the Q-table to a file."""
        assert pathname.endswith(".pkl")
        # ensure the directory exists
        os.makedirs(os.path.dirname(pathname), exist_ok=True)
        with open(pathname, "wb") as f:
            pickle.dump(self._q_table, f)

    @classmethod
    def load(cls, pathname: str) -> "QTable":
        """Load the Q-table from a file."""
        assert pathname.endswith(".pkl")
        with open(pathname, "rb") as f:
            q_table = pickle.load(f)
        return cls(
            QTableConfig(action_space=None, state_space=None, q_table=q_table)
        )


def q_table_train(
    env: gym.Env[Any, Any],
    q_table: QTable,
    episodes: int,
    max_steps: int,
    lr: float,
    gamma: float,
    min_epsilon: float,
    max_epsilon: float,
    decay_rate: float,
) -> None:
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
        episodes (int): The number of episodes to train.
        max_steps (int): The maximum number of steps per episode.
        lr (float): The learning rate.
        gamma (float): The discount factor.
        min_epsilon (float): The minimum exploration rate.
        max_epsilon (float): The maximum exploration rate.
        decay_rate (float): The decay rate of the exploration rate.
    """
    q_table.set_train_flag(train_flag=True)
    for episode in tqdm(range(episodes)):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -decay_rate * episode
        )
        state, _ = env.reset()
        # TODO: see whether the rendering can check the machine is suitable for this-task reanding.
        img_raw: Any = env.render()
        assert img_raw is not None, (
            "The image is None, please check the environment for rendering."
        )

        for _ in range(max_steps):
            action = q_table.action(state=state, epsilon=epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            old_score = q_table.get_score(state=state, action=action)
            new_score = old_score + lr * (
                float(reward)
                + gamma * q_table.get_score(state=next_state, action=None)
                - old_score
            )
            q_table.update(state=state, action=action, score=new_score)

            if terminated or truncated:
                break
            state = next_state


def main() -> None:
    parser = argparse.ArgumentParser()
    # add a --model_pathname argument if provided, use the default value if not provided
    parser.add_argument("--model_pathname", type=str, default="")
    args = parser.parse_args()

    # Training parameters
    episodes = 10000  # Total training episodes
    learning_rate = 0.7  # Learning rate
    max_steps = 99  # Max steps per episode
    gamma = 0.95  # Discounting rate

    # Exploration parameters
    max_epsilon = 1.0  # Exploration probability at start
    min_epsilon = 0.05  # Minimum exploration probability
    decay_rate = 0.0005  # Exponential decay rate for exploration prob

    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=False,
        render_mode="rgb_array",
    )
    if args.model_pathname:
        q_table = QTable.load(args.model_pathname)
    else:
        q_table = QTable(
            QTableConfig(
                # the shape of observation and action doesn't work, n cann't pass mypy
                state_space=env.observation_space.n,  # type: ignore
                action_space=env.action_space.n,  # type: ignore
            )
        )
    q_table_train(
        env=env,
        q_table=q_table,
        episodes=episodes,
        max_steps=max_steps,
        lr=learning_rate,
        gamma=gamma,
        min_epsilon=min_epsilon,
        max_epsilon=max_epsilon,
        decay_rate=decay_rate,
    )

    # Save the Q-table.
    q_table.save(str(EXERCISE1_RESULT_DIR / "q_table.pkl"))
    # save the hyperparameters
    hyperparameters = {
        "env_id": "FrozenLake-v1",
        "map_name": "4x4",
        "is_slippery": False,
        "render_mode": "rgb_array",
        "n_training_episodes": episodes,
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "gamma": gamma,
        "max_epsilon": max_epsilon,
        "min_epsilon": min_epsilon,
        "decay_rate": decay_rate,
    }
    with open(EXERCISE1_RESULT_DIR / "hyperparameters.json", "w") as f:
        json.dump(hyperparameters, f)

    # Evaluation parameters
    eval_episodes = 100  # Total number of test episodes
    eval_seed: list[int] = []  # The evaluation seed of the environment

    mean_reward, std_reward = evaluate_agent(
        env=env,
        policy=q_table,
        max_steps=max_steps,
        episodes=eval_episodes,
        seed=eval_seed,
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    # save the eval result
    eval_result = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "eval_episodes": eval_episodes,
        "eval_seed": eval_seed,
        "max_steps": max_steps,
    }
    with open(EXERCISE1_RESULT_DIR / "eval_result.json", "w") as f:
        json.dump(eval_result, f)


if __name__ == "__main__":
    main()
