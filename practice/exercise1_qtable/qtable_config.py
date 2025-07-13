"""Q-table configuration for FrozenLake environment."""

from dataclasses import dataclass

from practice.base import BaseConfig


@dataclass
class QTableConfig(BaseConfig):
    """Configuration for Q-table learning."""

    # Q-learning specific parameters
    learning_rate: float = 0.7
    min_epsilon: float = 0.05
    max_epsilon: float = 1.0
    decay_rate: float = 0.0005

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


# Example configuration instance
config = QTableConfig(
    # Environment parameters
    env_id="FrozenLake-v1",
    env_kwargs={"map_name": "4x4", "is_slippery": False, "render_mode": "rgb_array"},
    # Training parameters
    episodes=1000,
    max_steps=99,
    gamma=0.95,
    # Q-learning specific parameters
    learning_rate=0.7,
    min_epsilon=0.05,
    max_epsilon=1.0,
    decay_rate=0.0005,
    # Evaluation parameters
    eval_episodes=100,
    eval_seed=None,
    # Output parameters
    output_dir="results/exercise1_qtable/frozen_lake/",
    save_result=True,
    model_filename="qtable.pkl",
    params_filename="params.json",
    train_result_filename="train_result.json",
    eval_result_filename="eval_result.json",
    # Hub parameters
    repo_id="qtable-FrozenLake-v1",
)
