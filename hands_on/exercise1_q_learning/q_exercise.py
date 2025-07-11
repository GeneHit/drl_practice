from typing import TypeAlias

import gymnasium as gym
import numpy as np
import pickle5 as pickle
import torch
from numpy.typing import NDArray
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from hands_on.base import ActType, AgentBase, ScheduleBase
from hands_on.exercise1_q_learning.config import QTableTrainConfig
from hands_on.utils_for_coding.scheduler_utils import ExponentialSchedule

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


def epsilon_greedy_policy(q_table: NDArray[np.float32], state: int, epsilon: float) -> ActType:
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


class QTable(AgentBase):
    """Q-table for evaluation/gameplay.

    This class is focused on action selection using a trained Q-table.
    It does not handle training-specific operations.
    """

    def __init__(self, q_table: NDArray[np.float32]):
        self._q_table = q_table

    def action(self, state: ObsType) -> ActType:
        assert isinstance(state, int)
        # Always use greedy policy for evaluation
        return greedy_policy(self._q_table, state)

    def only_save_model(self, pathname: str) -> None:
        """Save the Q-table to a file."""
        assert pathname.endswith(".pkl")
        with open(pathname, "wb") as f:
            pickle.dump(self._q_table, f)

    @classmethod
    def load_from_checkpoint(cls, pathname: str, device: torch.device | None) -> "QTable":
        """Load the Q-table from a file."""
        with open(pathname, "rb") as f:
            q_table = pickle.load(f)
        return cls(q_table=q_table)


class QTableTrainer:
    """Handles Q-table training operations."""

    def __init__(
        self,
        q_table: NDArray[np.float32],
        learning_rate: float,
        gamma: float,
        epsilon_schedule: ScheduleBase,
    ) -> None:
        self._q_table = q_table
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._epsilon_schedule = epsilon_schedule

    def action(self, state: ObsType, episode: int) -> ActType:
        """Get action using epsilon-greedy policy."""
        assert isinstance(state, int)
        epsilon = self._epsilon_schedule(episode)
        return epsilon_greedy_policy(self._q_table, state, epsilon)

    def update(
        self,
        state: ObsType,
        action: ActType,
        reward: float,
        next_state: ObsType,
    ) -> float:
        """Update Q-table using Bellman equation.

        Returns:
            td_error: The temporal difference error for logging
        """
        assert isinstance(state, int)
        assert isinstance(next_state, int)

        old_score = self._q_table[state, action]
        next_max_q = float(max(self._q_table[next_state]))

        td_target = reward + self._gamma * next_max_q
        td_error = td_target - old_score

        new_score = old_score + self._learning_rate * td_error
        self._q_table[state, action] = new_score

        return abs(td_error)


def q_table_train_loop(
    env: gym.Env[ObsType, ActType],
    q_table: NDArray[np.float32],
    q_config: QTableTrainConfig,
    log_dir: str,
) -> None:
    """Train the Q-table.

    For each episode:
    - Reduce epsilon (since we need less and less exploration)
    - Reset the environment

    For step in max timesteps:
    - Choose the action At using epsilon greedy policy
    - Take the action (a) and observe the outcome state(s') and reward (r)
    - Update the Q-value Q(s,a) using Bellman equation:
        Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
    - If done, finish the episode
    - Our next state is the new state

    Args:
        env (gym.Env): The environment.
        q_table (NDArray[np.float32]): The Q-table.
        q_config (QTableTrainConfig): The training configuration.
        log_dir: Directory for tensorboard logs.
    """
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir)
    epsilon_schedule = ExponentialSchedule(
        start_e=q_config.min_epsilon,
        end_e=q_config.max_epsilon,
        decay_rate=q_config.decay_rate,
    )
    trainer = QTableTrainer(
        q_table=q_table,
        learning_rate=q_config.learning_rate,
        gamma=q_config.gamma,
        epsilon_schedule=epsilon_schedule,
    )

    global_step = 0
    for episode in tqdm(range(q_config.episodes)):
        state, _ = env.reset()

        # Track episode-level statistics
        episode_reward = 0.0
        episode_steps = 0

        for _ in range(q_config.max_steps):
            action = trainer.action(state=state, episode=episode)
            next_state, reward, terminated, truncated, infos = env.step(action)

            # Update episode tracking
            episode_reward += float(reward)
            episode_steps += 1

            td_error = trainer.update(
                state=state,
                action=action,
                reward=float(reward),
                next_state=next_state,
            )
            global_step += 1

            # Log training metrics
            writer.add_scalar("training/td_error", td_error, global_step)
            writer.add_scalar("training/epsilon", epsilon_schedule(episode), global_step)

            if terminated or truncated:
                # Log our manually tracked episode statistics
                writer.add_scalar("episode/reward", episode_reward, global_step)
                writer.add_scalar("episode/length", episode_steps, global_step)
                break
            state = next_state

    # Close writer
    writer.close()
