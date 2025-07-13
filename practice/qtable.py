"""Q-table implementation for discrete environments."""

import pickle
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from practice.base import ActType, AgentBase, BaseConfig, TrainerBase
from practice.utils.schedules import ExponentialSchedule


@dataclass
class QTableConfig(BaseConfig):
    """Configuration for Q-table learning."""

    # Q-learning specific parameters
    learning_rate: float
    min_epsilon: float
    max_epsilon: float
    decay_rate: float

    def validate(self) -> None:
        """Validate Q-table specific parameters."""
        if not (0 < self.learning_rate <= 1):
            raise ValueError(f"learning_rate must be in (0, 1], got {self.learning_rate}")

        if not (0 <= self.min_epsilon <= 1):
            raise ValueError(f"min_epsilon must be in [0, 1], got {self.min_epsilon}")

        if not (0 <= self.max_epsilon <= 1):
            raise ValueError(f"max_epsilon must be in [0, 1], got {self.max_epsilon}")

        if self.min_epsilon > self.max_epsilon:
            raise ValueError("min_epsilon must be <= max_epsilon")

        if self.decay_rate <= 0:
            raise ValueError(f"decay_rate must be positive, got {self.decay_rate}")


class QTable(AgentBase[np.int64]):
    """Q-table agent for evaluation."""

    def __init__(self, q_table: NDArray[np.float32]) -> None:
        self.q_table = q_table

    def action(self, state: np.int64) -> ActType:
        """Get greedy action for given state."""
        return np.argmax(self.q_table[state])

    def save_model(self, pathname: str) -> None:
        """Save Q-table to file."""
        with open(pathname, "wb") as f:
            pickle.dump(self.q_table, f)

    @classmethod
    def load_from_checkpoint(cls, pathname: str, device: torch.device | None = None) -> "QTable":
        """Load Q-table from checkpoint."""
        with open(pathname, "rb") as f:
            q_table = pickle.load(f)
        return cls(q_table)


class QTableTrainer(TrainerBase[np.int64]):
    """Q-table trainer with epsilon-greedy exploration."""

    def __init__(self, obs_space_n: int, action_space_n: int) -> None:
        self.obs_space_n = obs_space_n
        self.action_space_n = action_space_n
        self.q_table: NDArray[np.float32] | None = None
        self.epsilon_schedule: ExponentialSchedule | None = None
        self.learning_rate = 0.0
        self.gamma = 0.0

    def _init_training(self, config: QTableConfig) -> None:
        """Initialize training parameters."""
        self.q_table = np.zeros((self.obs_space_n, self.action_space_n), dtype=np.float32)
        self.epsilon_schedule = ExponentialSchedule(
            min_value=config.min_epsilon, max_value=config.max_epsilon, decay_rate=config.decay_rate
        )
        self.learning_rate = config.learning_rate
        self.gamma = config.gamma

    def action(
        self, state: np.int64, episode: int = 0, eval_mode: bool = False, **kwargs: Any
    ) -> ActType:
        """Get action with epsilon-greedy exploration during training."""
        if self.q_table is None:
            raise RuntimeError("Must call train() before using action()")

        if eval_mode:
            return ActType(np.argmax(self.q_table[state]))

        assert self.epsilon_schedule is not None
        epsilon = self.epsilon_schedule(episode)
        if np.random.rand() < epsilon:
            return ActType(np.random.randint(self.action_space_n))
        else:
            return ActType(np.argmax(self.q_table[state]))

    def update(
        self, state: np.int64, action: ActType, reward: float, next_state: np.int64
    ) -> float:
        """Update Q-table using Q-learning update rule."""
        if self.q_table is None:
            raise RuntimeError("Must call train() before using update()")

        old_q_value = self.q_table[state, action]
        max_next_q_value = np.max(self.q_table[next_state])

        # Q-learning update
        td_target = reward + self.gamma * max_next_q_value
        td_error = td_target - old_q_value
        self.q_table[state, action] = old_q_value + self.learning_rate * td_error

        return float(abs(td_error))

    def train(self, config: QTableConfig, log_dir: str) -> QTable:
        """Train Q-table agent."""
        self._init_training(config)
        assert self.q_table is not None

        # Create environment
        env = gym.make(config.env_id, **config.env_kwargs)

        # Setup logging
        writer = SummaryWriter(log_dir)

        # Training statistics
        episode_rewards = []
        episode_lengths = []
        td_errors = []

        for episode in tqdm(range(config.episodes), desc="Training Q-table"):
            state, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            episode_td_errors = []

            for step in range(config.max_steps):
                action = self.action(state, episode)
                next_state, reward, terminated, truncated, _ = env.step(action)

                td_error = self.update(state, action, float(reward), next_state)
                episode_td_errors.append(td_error)

                episode_reward += float(reward)
                episode_length += 1

                state = next_state

                if terminated or truncated:
                    break

            # Log episode statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            td_errors.extend(episode_td_errors)

            # Log to tensorboard
            if episode % 100 == 0:
                assert self.epsilon_schedule is not None
                writer.add_scalar("Episode/Reward", episode_reward, episode)
                writer.add_scalar("Episode/Length", episode_length, episode)
                writer.add_scalar("Episode/Epsilon", self.epsilon_schedule(episode), episode)

                if episode_td_errors:
                    writer.add_scalar("Episode/Mean_TD_Error", np.mean(episode_td_errors), episode)

        env.close()
        writer.close()

        # Save training results
        if config.save_result:
            model_path = f"{config.output_dir}/{config.model_filename}"
            agent = QTable(self.q_table)
            agent.save_model(model_path)

        return QTable(self.q_table)
