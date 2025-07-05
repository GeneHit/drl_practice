from typing import Any

import gymnasium as gym
import numpy as np
import pickle5 as pickle
from numpy.typing import NDArray

from common.base import PolicyBase
from common.evaluation_utils import evaluate_agent


def greedy_policy(q_table: NDArray[np.float32], state: int) -> int:
    """Take the action with the highest state, action value.

    Args:
        q_table (NDArray[np.float32]): The Q-table.
        state (int): The current state.

    Returns:
        int: The action to take.
    """
    raise NotImplementedError("Not implemented")


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
    raise NotImplementedError("Not implemented")


class QTable(PolicyBase):
    """Q-table."""

    def __init__(self, state_space: int, action_space: int):
        self._q_table = np.zeros((state_space, action_space))
        self._train_flag = False

    def set_train_flag(self, train_flag: bool) -> None:
        self._train_flag = train_flag

    def action(
        self, state: int | NDArray[np.float32]
    ) -> int | NDArray[np.float32]:
        assert isinstance(state, int)
        raise NotImplementedError("Not implemented")

    def update(
        self, state: int, action: int, next_state: int, reward: float
    ) -> None:
        raise NotImplementedError("Not implemented")


def train(
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
    raise NotImplementedError("Not implemented")


if __name__ == "__main__":
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
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    assert state_shape == (1,)
    assert action_shape == (1,)
    q_table = QTable(state_space=state_shape[0], action_space=action_shape[0])
    train(
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

    # Save the Q-table
    with open("results/q_table.pkl", "wb") as f:
        pickle.dump(q_table._q_table, f)
