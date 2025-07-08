"""Configuration dataclasses for DQN training."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass
class DQNTrainConfig:
    """Configuration for DQN training that matches dqn_train function parameters."""

    # Training hyperparameters
    global_steps: int
    max_steps: int
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
        # Validate required parameters
        required_params = {
            "global_steps",
            "max_steps",
            "start_epsilon",
            "end_epsilon",
            "exploration_fraction",
            "replay_buffer_capacity",
            "batch_size",
            "gamma",
            "train_interval",
            "target_update_interval",
            "update_start_step",
            "learning_rate",
        }

        missing_params = required_params - set(config_dict.keys())
        if missing_params:
            raise KeyError(
                f"Missing required parameters in hyper_params: {missing_params}"
            )

        # Validate parameter types and ranges
        cls._validate_parameters(config_dict)

        return cls(
            global_steps=config_dict["global_steps"],
            max_steps=config_dict["max_steps"],
            start_epsilon=config_dict["start_epsilon"],
            end_epsilon=config_dict["end_epsilon"],
            exploration_fraction=config_dict["exploration_fraction"],
            replay_buffer_capacity=config_dict["replay_buffer_capacity"],
            batch_size=config_dict["batch_size"],
            gamma=config_dict["gamma"],
            train_interval=config_dict["train_interval"],
            target_update_interval=config_dict["target_update_interval"],
            update_start_step=config_dict["update_start_step"],
            learning_rate=config_dict["learning_rate"],
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
            "global_steps",
            "max_steps",
            "replay_buffer_capacity",
            "batch_size",
            "train_interval",
            "target_update_interval",
            "update_start_step",
        }
        for param in positive_int_params:
            if not isinstance(params[param], int) or params[param] <= 0:
                raise ValueError(
                    f"{param} must be a positive integer, got {params[param]}"
                )

        # Validate probability ranges (0-1)
        probability_params = {
            "start_epsilon",
            "end_epsilon",
            "exploration_fraction",
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

        # Validate learning rate
        if (
            not isinstance(params["learning_rate"], (int, float))
            or params["learning_rate"] <= 0
        ):
            raise ValueError(
                f"learning_rate must be positive, got {params['learning_rate']}"
            )

        # Validate epsilon relationship
        if params["start_epsilon"] < params["end_epsilon"]:
            raise ValueError(
                "start_epsilon should be >= end_epsilon for exploration decay"
            )

        # Validate update_start_step vs global_steps
        if params["update_start_step"] >= params["global_steps"]:
            raise ValueError("update_start_step should be < global_steps")

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary format suitable for dqn_train function.

        Returns:
            Dictionary with all parameters needed for dqn_train
        """
        return {
            "global_steps": self.global_steps,
            "max_steps": self.max_steps,
            "start_epsilon": self.start_epsilon,
            "end_epsilon": self.end_epsilon,
            "exploration_fraction": self.exploration_fraction,
            "replay_buffer_capacity": self.replay_buffer_capacity,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "train_interval": self.train_interval,
            "target_update_interval": self.target_update_interval,
            "update_start_step": self.update_start_step,
        }


def load_config_from_json(config_path: str | Path) -> dict[str, Any]:
    """Load the DQN training configuration from a JSON file.

    Args:
        config_path: The path to the JSON configuration file

    Returns:
        dict[str, Any]: The configuration loaded from the JSON file
    """
    with open(config_path, "r") as f:
        config_data = cast(dict[str, Any], json.load(f))
    return config_data
