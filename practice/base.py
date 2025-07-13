"""Base classes for RL trainers and configurations."""

import abc
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Generic, TypeAlias, TypeVar

import numpy as np
import torch
from numpy.typing import NDArray

ActType: TypeAlias = np.int64
ObsType = TypeVar("ObsType")


@dataclass
class BaseConfig(abc.ABC):
    """Base configuration class for all algorithms."""
    
    # Environment parameters
    env_id: str
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Training parameters
    episodes: int = 1000
    max_steps: int = 500
    gamma: float = 0.99
    
    # Evaluation parameters
    eval_episodes: int = 100
    eval_seed: list[int] | None = None
    
    # Output parameters
    output_dir: str = "results/"
    save_result: bool = True
    model_filename: str = "model.pth"
    params_filename: str = "params.json"
    train_result_filename: str = "train_result.json"
    eval_result_filename: str = "eval_result.json"
    
    # Hub parameters (optional)
    repo_id: str = ""
    
    def __post_init__(self) -> None:
        """Initialize derived values and validate parameters."""
        if self.eval_seed is None:
            # Generate deterministic eval seeds
            np.random.seed(42)
            self.eval_seed = np.random.randint(0, 1000, self.eval_episodes).tolist()
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.validate()
    
    @abc.abstractmethod
    def validate(self) -> None:
        """Validate configuration parameters."""
        pass
    
    def save_to_json(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=4)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


class AgentBase(abc.ABC, Generic[ObsType]):
    """Base class for all RL agents."""
    
    @abc.abstractmethod
    def action(self, state: ObsType) -> ActType:
        """Get action for given state."""
        pass
    
    @abc.abstractmethod
    def save_model(self, pathname: str) -> None:
        """Save the model."""
        pass
    
    @classmethod
    @abc.abstractmethod
    def load_from_checkpoint(cls, pathname: str, device: torch.device | None = None) -> "AgentBase[Any]":
        """Load model from checkpoint."""
        pass


class TrainerBase(abc.ABC, Generic[ObsType]):
    """Base class for all RL trainers."""
    
    def __init__(self, config: BaseConfig, network: Any = None, device: torch.device | None = None) -> None:
        """Initialize trainer with config, network, and device."""
        self.config = config
        self.network = network
        self.device = device if device is not None else torch.device("cpu")
        self.training_results: Dict[str, Any] = {}
    
    @abc.abstractmethod
    def train(self, log_dir: str) -> AgentBase[ObsType]:
        """Train the agent and return the trained agent."""
        pass
    
    @abc.abstractmethod
    def action(self, state: ObsType, **kwargs: Any) -> ActType:
        """Get action during training (may include exploration)."""
        pass
    
    def save_training_results(self) -> None:
        """Save training results to JSON file."""
        if self.config.save_result and self.training_results:
            result_path = Path(self.config.output_dir) / self.config.train_result_filename
            with open(result_path, 'w') as f:
                json.dump(self.training_results, f, indent=4)
    
    def save_config(self) -> None:
        """Save configuration to JSON file."""
        if self.config.save_result:
            config_path = Path(self.config.output_dir) / self.config.params_filename
            self.config.save_to_json(str(config_path)) 