"""Q-table implementation for discrete environments with enhanced architecture."""

import pickle
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from practice.base import ActType, AgentBase, TrainerBase
from practice.exercise1_qtable.qtable_config import QTableConfig
from practice.utils.schedules import ExponentialSchedule


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

    def __init__(
        self, config: QTableConfig, network: Any = None, device: torch.device | None = None
    ) -> None:
        super().__init__(config, network, device)

        # Cast config for type safety
        self.config = config

        # Create temporary environment to get dimensions
        temp_env = gym.make(config.env_id, **config.env_kwargs)
        if not hasattr(temp_env.observation_space, "n"):
            raise ValueError("Q-table requires discrete observation space")
        if not hasattr(temp_env.action_space, "n"):
            raise ValueError("Q-table requires discrete action space")

        self.obs_space_n = temp_env.observation_space.n
        self.action_space_n = temp_env.action_space.n
        temp_env.close()

        # Initialize Q-table and schedules
        self.q_table: NDArray[np.float32] | None = None
        self.epsilon_schedule: ExponentialSchedule | None = None

        # Training statistics
        self.training_results = {
            "episode_rewards": [],
            "episode_lengths": [],
            "td_errors": [],
            "epsilons": [],
        }

    def _init_training(self) -> None:
        """Initialize training parameters."""
        self.q_table = np.zeros((self.obs_space_n, self.action_space_n), dtype=np.float32)
        self.epsilon_schedule = ExponentialSchedule(
            min_value=self.config.min_epsilon,
            max_value=self.config.max_epsilon,
            decay_rate=self.config.decay_rate,
        )

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
        td_target = reward + self.config.gamma * max_next_q_value
        td_error = td_target - old_q_value
        self.q_table[state, action] = old_q_value + self.config.learning_rate * td_error

        return float(abs(td_error))

    def train(self, log_dir: str) -> QTable:
        """Train Q-table agent."""
        print(f"Training Q-table on {self.config.env_id}...")

        self._init_training()
        assert self.q_table is not None

        # Create environment
        env = gym.make(self.config.env_id, **self.config.env_kwargs)

        # Setup logging
        writer = SummaryWriter(log_dir)

        # Training loop
        for episode in tqdm(range(self.config.episodes), desc="Training Q-table"):
            state, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            episode_td_errors = []

            for step in range(self.config.max_steps):
                action = self.action(state, episode)
                next_state, reward, terminated, truncated, _ = env.step(action)

                td_error = self.update(state, action, float(reward), next_state)
                episode_td_errors.append(td_error)

                episode_reward += float(reward)
                episode_length += 1

                state = next_state

                if terminated or truncated:
                    break

            # Store episode statistics
            self.training_results["episode_rewards"].append(episode_reward)
            self.training_results["episode_lengths"].append(episode_length)
            self.training_results["td_errors"].extend(episode_td_errors)

            # Log to tensorboard
            if episode % 100 == 0:
                assert self.epsilon_schedule is not None
                writer.add_scalar("Episode/Reward", episode_reward, episode)
                writer.add_scalar("Episode/Length", episode_length, episode)
                epsilon = self.epsilon_schedule(episode)
                writer.add_scalar("Episode/Epsilon", epsilon, episode)
                self.training_results["epsilons"].append(epsilon)

                if episode_td_errors:
                    writer.add_scalar("Episode/Mean_TD_Error", np.mean(episode_td_errors), episode)

        env.close()
        writer.close()

        # Save training results
        self.save_training_results()

        # Create and save agent
        agent = QTable(self.q_table)
        if self.config.save_result:
            model_path = f"{self.config.output_dir}/{self.config.model_filename}"
            agent.save_model(model_path)

        return agent


# Main function using enhanced CLI
if __name__ == "__main__":
    from practice.enhanced_cli import create_enhanced_main_function
    from practice.exercise1_qtable.qtable_config import QTableConfig

    main = create_enhanced_main_function(
        algorithm_name="QTable",
        config_class=QTableConfig,
        trainer_class=QTableTrainer,
        model_loader=QTable,
        network_factory=None,  # Q-table doesn't need a network factory
        config_example="qtable_config.py",
    )

    main()
