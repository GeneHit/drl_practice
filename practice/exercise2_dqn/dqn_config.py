"""DQN configuration for both 1D and 2D observations with multi-environment support."""

from dataclasses import dataclass

from practice.base import BaseConfig


@dataclass
class DQNConfig(BaseConfig):
    """Configuration for DQN training with multi-environment support."""

    # DQN specific hyperparameters
    timesteps: int = 200000
    learning_rate: float = 1e-4
    gamma: float = 0.99
    start_epsilon: float = 1.0
    end_epsilon: float = 0.01
    exploration_fraction: float = 0.1
    replay_buffer_capacity: int = 120000
    batch_size: int = 64
    train_interval: int = 1
    target_update_interval: int = 250
    update_start_step: int = 1000
    num_envs: int = 6
    use_multi_processing: bool = True

    # Network architecture (auto-detected based on observation space)
    hidden_dim: int = 128
    num_layers: int = 2

    def validate(self) -> None:
        """Validate DQN specific parameters."""
        if not (0 < self.learning_rate <= 1):
            raise ValueError(f"learning_rate must be in (0, 1], got {self.learning_rate}")

        if not (0 <= self.start_epsilon <= 1):
            raise ValueError(f"start_epsilon must be in [0, 1], got {self.start_epsilon}")

        if not (0 <= self.end_epsilon <= 1):
            raise ValueError(f"end_epsilon must be in [0, 1], got {self.end_epsilon}")

        if self.end_epsilon > self.start_epsilon:
            raise ValueError("end_epsilon must be <= start_epsilon")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.replay_buffer_capacity <= 0:
            raise ValueError(
                f"replay_buffer_capacity must be positive, got {self.replay_buffer_capacity}"
            )

        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}")

        if not (0 < self.exploration_fraction <= 1):
            raise ValueError(
                f"exploration_fraction must be in (0, 1], got {self.exploration_fraction}"
            )

    @property
    def epsilon_decay_duration(self) -> int:
        """Calculate epsilon decay duration based on exploration fraction."""
        return int(self.exploration_fraction * self.timesteps)

    def is_2d_observation(self) -> bool:
        """Check if configuration is for 2D observations (image-based)."""
        return (
            "use_image" in self.env_kwargs
            and self.env_kwargs["use_image"]
            or "resize_shape" in self.env_kwargs
            or "frame_stack_size" in self.env_kwargs
        )
