"""Configuration dataclasses for DQN training."""

import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass
class DQNTrainConfig:
    """Configuration for DQN training that matches dqn_train function parameters."""

    # Training hyperparameters
    timesteps: int
    max_steps: int | None
    start_epsilon: float
    end_epsilon: float
    exploration_fraction: float
    replay_buffer_capacity: int
    batch_size: int
    gamma: float
    train_interval: int
    target_update_interval: int
    update_start_step: int
    learning_rate: float

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self._validate_parameters()

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "DQNTrainConfig":
        """Load DQNTrainConfig from dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            DQNTrainConfig instance loaded from the dictionary

        Raises:
            KeyError: If required parameters are missing from the dictionary
            ValueError: If parameter values are invalid
        """
        # Get dataclass fields and determine which are required
        fields = dataclasses.fields(cls)
        field_names = {field.name for field in fields}

        # Required fields are those without defaults (or with MISSING default)
        required_params = {
            field.name
            for field in fields
            if field.default == dataclasses.MISSING
            and field.default_factory == dataclasses.MISSING
        }

        missing_params = required_params - set(config_dict.keys())
        if missing_params:
            raise KeyError(
                f"Missing required parameters in hyper_params: {missing_params}"
            )

        # Filter config_dict to only include valid dataclass fields
        filtered_params = {
            k: v for k, v in config_dict.items() if k in field_names
        }

        return cls(**filtered_params)

    def _validate_parameters(self) -> None:
        """Validate parameter values.

        Raises:
            ValueError: If any parameter value is invalid
        """
        # Validate positive integers
        if not isinstance(self.timesteps, int) or self.timesteps <= 0:
            raise ValueError(
                f"timesteps must be a positive integer, got {self.timesteps}"
            )

        if (
            not isinstance(self.replay_buffer_capacity, int)
            or self.replay_buffer_capacity <= 0
        ):
            raise ValueError(
                f"replay_buffer_capacity must be a positive integer, got {self.replay_buffer_capacity}"
            )

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be a positive integer, got {self.batch_size}"
            )

        if not isinstance(self.train_interval, int) or self.train_interval <= 0:
            raise ValueError(
                f"train_interval must be a positive integer, got {self.train_interval}"
            )

        if (
            not isinstance(self.target_update_interval, int)
            or self.target_update_interval <= 0
        ):
            raise ValueError(
                f"target_update_interval must be a positive integer, got {self.target_update_interval}"
            )

        if (
            not isinstance(self.update_start_step, int)
            or self.update_start_step <= 0
        ):
            raise ValueError(
                f"update_start_step must be a positive integer, got {self.update_start_step}"
            )

        # Validate probability ranges (0-1)
        if (
            not isinstance(self.start_epsilon, (int, float))
            or not 0 <= self.start_epsilon <= 1
        ):
            raise ValueError(
                f"start_epsilon must be between 0 and 1, got {self.start_epsilon}"
            )

        if (
            not isinstance(self.end_epsilon, (int, float))
            or not 0 <= self.end_epsilon <= 1
        ):
            raise ValueError(
                f"end_epsilon must be between 0 and 1, got {self.end_epsilon}"
            )

        if (
            not isinstance(self.exploration_fraction, (int, float))
            or not 0 <= self.exploration_fraction <= 1
        ):
            raise ValueError(
                f"exploration_fraction must be between 0 and 1, got {self.exploration_fraction}"
            )

        if not isinstance(self.gamma, (int, float)) or not 0 <= self.gamma <= 1:
            raise ValueError(f"gamma must be between 0 and 1, got {self.gamma}")

        # Validate learning rate
        if (
            not isinstance(self.learning_rate, (int, float))
            or self.learning_rate <= 0
        ):
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )

        # Validate epsilon relationship
        if self.start_epsilon < self.end_epsilon:
            raise ValueError(
                "start_epsilon should be >= end_epsilon for exploration decay"
            )

        # Validate update_start_step vs timesteps
        if self.update_start_step >= self.timesteps:
            raise ValueError("update_start_step should be < timesteps")
