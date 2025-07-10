"""Configuration dataclasses for Q-learning training."""

import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass
class QTableTrainConfig:
    """Configuration for Q-table training that matches q_table_train function parameters."""

    # Training hyperparameters
    episodes: int
    max_steps: int
    learning_rate: float
    gamma: float
    min_epsilon: float
    max_epsilon: float
    decay_rate: float

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self._validate_parameters()

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "QTableTrainConfig":
        """Load QTableTrainConfig from dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            QTableTrainConfig instance loaded from the dictionary

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
        if not isinstance(self.episodes, int) or self.episodes <= 0:
            raise ValueError(
                f"episodes must be a positive integer, got {self.episodes}"
            )

        if not isinstance(self.max_steps, int) or self.max_steps <= 0:
            raise ValueError(
                f"max_steps must be a positive integer, got {self.max_steps}"
            )

        # Validate probability ranges (0-1)
        if (
            not isinstance(self.min_epsilon, (int, float))
            or not 0 <= self.min_epsilon <= 1
        ):
            raise ValueError(
                f"min_epsilon must be between 0 and 1, got {self.min_epsilon}"
            )

        if (
            not isinstance(self.max_epsilon, (int, float))
            or not 0 <= self.max_epsilon <= 1
        ):
            raise ValueError(
                f"max_epsilon must be between 0 and 1, got {self.max_epsilon}"
            )

        if not isinstance(self.gamma, (int, float)) or not 0 <= self.gamma <= 1:
            raise ValueError(f"gamma must be between 0 and 1, got {self.gamma}")

        # Validate learning rate (0-1)
        if (
            not isinstance(self.learning_rate, (int, float))
            or not 0 < self.learning_rate <= 1
        ):
            raise ValueError(
                f"learning_rate must be between 0 and 1, got {self.learning_rate}"
            )

        # Validate decay rate (positive)
        if (
            not isinstance(self.decay_rate, (int, float))
            or self.decay_rate <= 0
        ):
            raise ValueError(
                f"decay_rate must be positive, got {self.decay_rate}"
            )

        # Validate epsilon relationship
        if self.min_epsilon > self.max_epsilon:
            raise ValueError(
                "min_epsilon should be <= max_epsilon for exploration decay"
            )
