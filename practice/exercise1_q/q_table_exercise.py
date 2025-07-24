from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, Union

import gymnasium as gym
import numpy as np
import pickle5 as pickle
import torch
from numpy.typing import NDArray
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from practice.base.chest import AgentBase, ScheduleBase
from practice.base.config import BaseConfig
from practice.base.context import ContextBase
from practice.base.env_typing import ActType
from practice.base.trainer import TrainerBase

# Type alias for discrete observations
ObsType: TypeAlias = np.int64

# Type alias for single environment
EnvType = gym.Env[ObsType, ActType]


def greedy_policy(q_table: NDArray[np.float32], state: int) -> ActType:
    """Take the action with the highest state, action value.

    Args:
        q_table: The Q-table.
        state: The current state.

    Returns:
        The action to take.
    """
    return np.argmax(q_table[state])


def epsilon_greedy_policy(q_table: NDArray[np.float32], state: int, epsilon: float) -> ActType:
    """Take an action with the epsilon-greedy strategy.

    2 strategies:
    - Exploration: take a random action with probability epsilon.
    - Exploitation: take the action with the highest state, action value.

    Args:
        q_table: The Q-table.
        state: The current state.
        epsilon: The exploration rate.

    Returns:
        The action to take.
    """
    if np.random.rand() < epsilon:
        return np.random.randint(q_table.shape[1], dtype=ActType)
    else:
        return np.argmax(q_table[state])


@dataclass(kw_only=True, frozen=True)
class QTableConfig(BaseConfig):
    """Configuration for Q-table learning algorithm."""

    episodes: int
    min_epsilon: float = 0.05
    max_epsilon: float = 1.0
    decay_rate: float = 0.0005
    epsilon_schedule: ScheduleBase


class QTable(AgentBase):
    """Q-table for evaluation/gameplay.

    This class is focused on action selection using a trained Q-table.
    It does not handle training-specific operations.

    Check practice/utils_for_coding/agent_utils.py for more Agents.
    """

    def __init__(self, q_table: NDArray[np.float32]) -> None:
        self._q_table = q_table

    def action(self, state: Union[ObsType, int, np.integer]) -> ActType:  # type: ignore
        # Always use greedy policy for evaluation
        return greedy_policy(self._q_table, int(state))

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


class QTableTrainer(TrainerBase):
    """A trainer for the Q-table algorithm."""

    def __init__(self, config: QTableConfig, ctx: ContextBase) -> None:
        super().__init__(config=config, ctx=ctx)
        self._config: QTableConfig = config
        self._ctx: ContextBase = ctx

    def train(self) -> None:
        """Train the Q-table agent."""
        # Initialize tensorboard writer
        writer = SummaryWriter(
            log_dir=Path(self._config.artifact_config.output_dir) / "tensorboard"
        )
        # Use single environment from context
        env = self._ctx.env
        # Create trainer pod
        pod = _QTablePod(config=self._config, ctx=self._ctx)

        # Training loop
        global_steps = 0
        pbar = tqdm(total=self._config.episodes, desc="Training")
        for episode in range(self._config.episodes):
            state_raw, _ = env.reset()
            # Convert state to int for Q-table indexing
            state = int(state_raw)
            episode_reward = 0.0
            episode_step = 0

            done = False
            while not done:
                action = pod.action(state=state, episode=episode)
                next_state_raw, reward, terminated, truncated, info = env.step(action)
                # Convert next_state to int for Q-table indexing
                next_state = int(next_state_raw)
                td_error = pod.update(
                    state=state, action=action, reward=float(reward), next_state=next_state
                )

                # Log training metrics
                writer.add_scalar("training/td_error", td_error, global_steps)
                writer.add_scalar(
                    "training/epsilon", self._config.epsilon_schedule(episode), global_steps
                )

                done = bool(terminated or truncated)
                episode_reward += float(reward)
                episode_step += 1
                state = next_state
                global_steps += 1

            writer.add_scalar("episode/reward", episode_reward, episode)
            writer.add_scalar("episode/length", episode_step, episode)
            pbar.update(1)

        # Cleanup
        writer.close()


class _QTablePod:
    """Internal pod for Q-table training logic."""

    def __init__(self, config: QTableConfig, ctx: ContextBase) -> None:
        self._config = config
        self._ctx = ctx

    def action(self, state: int, episode: int) -> ActType:
        """Get action using epsilon-greedy policy."""
        return epsilon_greedy_policy(
            q_table=self._ctx.table,
            state=state,
            epsilon=self._config.epsilon_schedule(episode),
        )

    def update(self, state: int, action: ActType, reward: float, next_state: int) -> float:
        """Update Q-table using Bellman equation.

        Returns:
            td_error: The temporal difference error for logging
        """
        q_table = self._ctx.table
        old_score = q_table[state, action]
        next_max_q = float(max(q_table[next_state]))

        td_target = reward + self._config.gamma * next_max_q
        td_error = td_target - old_score

        new_score = old_score + self._config.learning_rate * td_error
        q_table[state, action] = new_score

        return float(abs(td_error))
