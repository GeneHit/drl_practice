"""Base classes for RL trainers and configurations."""

import abc
from pathlib import Path
from typing import TypeAlias

import numpy as np

from practice.base.config import BaseConfig

ActType: TypeAlias = np.int64
# ObsType = TypeVar("ObsType")


class TrainerBase(abc.ABC):
    """Base class for all RL trainers."""

    def __init__(self, config: BaseConfig) -> None:
        """Initialize trainer with config."""
        self._config = config

    @abc.abstractmethod
    def train(self) -> None:
        """Train the agent and return the trained agent."""

    def save_config(self) -> None:
        """Save configuration to JSON file."""
        artifact_config = self._config.artifact_config
        if artifact_config.save_result:
            config_path = Path(artifact_config.output_dir) / artifact_config.params_filename
            self._config.save_to_json(str(config_path))
