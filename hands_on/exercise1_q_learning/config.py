"""Configuration dataclasses for Q-learning training."""

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
        # Validate required parameters
        required_params = {
            "episodes",
            "max_steps",
            "learning_rate",
            "gamma",
            "min_epsilon",
            "max_epsilon",
            "decay_rate",
        }

        missing_params = required_params - set(config_dict.keys())
        if missing_params:
            raise KeyError(
                f"Missing required parameters in hyper_params: {missing_params}"
            )

        # Validate parameter types and ranges
        cls._validate_parameters(config_dict)

        return cls(
            episodes=config_dict["episodes"],
            max_steps=config_dict["max_steps"],
            learning_rate=config_dict["learning_rate"],
            gamma=config_dict["gamma"],
            min_epsilon=config_dict["min_epsilon"],
            max_epsilon=config_dict["max_epsilon"],
            decay_rate=config_dict["decay_rate"],
        )

    @staticmethod
    def _validate_parameters(params: dict[str, Any]) -> None:
        """Validate parameter values.

        Args:
            params: Dictionary of parameters to validate

        Raises:
            ValueError: If any parameter value is invalid
        """
        # Validate positive integers
        positive_int_params = {
            "episodes",
            "max_steps",
        }
        for param in positive_int_params:
            if not isinstance(params[param], int) or params[param] <= 0:
                raise ValueError(
                    f"{param} must be a positive integer, got {params[param]}"
                )

        # Validate probability ranges (0-1)
        probability_params = {
            "min_epsilon",
            "max_epsilon",
            "gamma",
        }
        for param in probability_params:
            if (
                not isinstance(params[param], (int, float))
                or not 0 <= params[param] <= 1
            ):
                raise ValueError(
                    f"{param} must be between 0 and 1, got {params[param]}"
                )

        # Validate learning rate (0-1)
        if (
            not isinstance(params["learning_rate"], (int, float))
            or not 0 < params["learning_rate"] <= 1
        ):
            raise ValueError(
                f"learning_rate must be between 0 and 1, got {params['learning_rate']}"
            )

        # Validate decay rate (positive)
        if (
            not isinstance(params["decay_rate"], (int, float))
            or params["decay_rate"] <= 0
        ):
            raise ValueError(
                f"decay_rate must be positive, got {params['decay_rate']}"
            )

        # Validate epsilon relationship
        if params["min_epsilon"] > params["max_epsilon"]:
            raise ValueError(
                "min_epsilon should be <= max_epsilon for exploration decay"
            )
