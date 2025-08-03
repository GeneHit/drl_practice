import abc
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Type

import torch

from practice.base.chest import AgentBase
from practice.utils.dist_utils import get_world_size

# reason: https://github.com/python/mypy/issues/11000
if TYPE_CHECKING:
    from practice.base.trainer import TrainerBase


@dataclass(frozen=True, kw_only=True)
class ArtifactConfig(abc.ABC):
    """Base configuration class for all artifacts."""

    trainer_type: "Type[TrainerBase]"
    """The type of the trainer. Use type[TrainerBase] to avoid circular import."""
    # TODO: remove this
    agent_type: type[AgentBase]
    """The type of the agent."""
    play_full_model: bool = False
    """Whether to load the full model for playing."""

    # Output parameters
    output_dir: str
    """The output directory for the artifacts."""
    save_result: bool = True
    """Whether to save the result."""
    model_filename: str = "full_model.pt"
    """The filename for the full model."""
    state_dict_filename: str = "state_dict.pt"
    """The filename for the state dict."""
    params_filename: str = "params.json"
    """The filename for the parameters."""
    train_result_filename: str = "train_result.json"
    """The filename for the train result."""
    eval_result_filename: str = "eval_result.json"
    """The filename for the eval result."""
    env_setup_filename: str = "env_setup.json"
    """The filename for the env setup."""
    tensorboard_dir: str = "tensorboard"
    """The directory for the tensorboard."""

    # Hub parameters (optional)
    repo_id: str
    """The repository ID for the huggingface hub."""
    seek_for_play: int = 42
    """The seed for the replay video."""
    replay_video_filename: str = "replay.mp4"
    """The filename for the replay video."""
    fps: int = 30
    """The frame rate of the replay video."""
    fps_skip: int = 1
    """The frame rate to skip."""
    algorithm_name: str
    """The name of the algorithm."""
    extra_tags: tuple[str, ...] = ()
    """The extra tags for the huggingface hub."""
    usage_instructions: str = ""
    """The usage instructions for the huggingface hub."""

    def get_tensorboard_dir(self) -> str:
        """Get the tensorboard directory."""
        return str(Path(self.output_dir) / self.tensorboard_dir)


@dataclass(frozen=True, kw_only=True)
class EnvConfig(abc.ABC):
    """Configuration class for the gymnasium environments."""

    env_id: str
    """The id of the environment."""
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    """The kwargs for the environment."""
    max_steps: int | None = None
    """The maximum number of steps for the environment."""
    normalize_obs: bool = False
    """Whether to normalize the observations."""
    use_image: bool = False
    """Whether to use image observations."""

    vector_env_num: int | None = None
    """Number of vector environments to use.

    If None, it will not use vector environments.
    """
    use_multi_processing: bool = False
    """Whether to use multi-processing to create the vector environments."""

    # Image parameters
    image_shape: tuple[int, int] | None = None
    """Shape of the image (height, width)."""
    frame_stack: int = 1
    """Number of frames to stack."""
    frame_skip: int = 1
    """Number of frames to skip."""

    training_render_mode: str | None = None
    """Render mode for training."""


@dataclass(frozen=True, kw_only=True)
class BaseConfig(abc.ABC):
    """Base configuration class for all algorithms."""

    # Environment parameters
    env_config: EnvConfig
    """The configuration for the gymnasium environment."""

    device: torch.device
    """The device to run the neural network."""

    # Training parameters
    learning_rate: float
    """The learning rate for the optimizer."""
    gamma: float = 0.99
    """The discount factor."""
    checkpoint_pathname: str = ""
    """The pathname of the checkpoint to load."""
    max_grad_norm: float | None = None
    """The maximum gradient norm for gradient clipping."""
    log_interval: int = 100
    """The interval to log the stats."""

    # Evaluation parameters
    eval_episodes: int = 100
    """The number of episodes to evaluate the policy."""
    eval_random_seed: int = 42
    """The seed for the evaluation."""
    eval_video_num: int | None = None
    """The number of videos to record.

    If None, the video will not be recorded.
    """

    artifact_config: ArtifactConfig
    """The configuration for the saving, naming, etc."""

    def save_to_json(self, filepath: str, with_data: dict[str, Any] = {}) -> None:
        """Save configuration to JSON file, except for artifact_config. Optionally merge extra data."""
        # Build dict, excluding artifact_config
        config_dict = self.to_dict()
        if with_data:
            config_dict.update(with_data)

        # Add world_size to the config if distributed
        world_size = get_world_size()
        if world_size > 1:
            config_dict["world_size"] = world_size

        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=4)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary, excluding artifact_config."""
        result = asdict(self)
        result.pop("artifact_config", None)

        # Make the result JSON serializable by converting complex objects
        result = self._make_json_serializable(result)

        return result

    def _make_json_serializable(self, obj: Any) -> Any:
        """Recursively convert objects to JSON-serializable format."""
        if isinstance(obj, torch.device):
            return str(obj)
        elif hasattr(obj, "__dict__"):
            # For objects with attributes (like schedules), convert to dict representation
            if hasattr(obj, "__class__"):
                return {
                    "_type": obj.__class__.__name__,
                    "_module": obj.__class__.__module__,
                    **{k: self._make_json_serializable(v) for k, v in obj.__dict__.items()},
                }
            else:
                return {k: self._make_json_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            # For any other type, convert to string representation
            return str(obj)
