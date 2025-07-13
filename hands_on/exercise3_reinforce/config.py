import dataclasses
from typing import Any


@dataclasses.dataclass
class ReinforceConfig:
    """The config for the reinforce algorithm."""

    global_episode: int
    gamma: float
    grad_acc: int
    lr: float
    baseline_decay: float = 0.99
    entropy_coef: float = 0.01

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ReinforceConfig":
        """Load ReinforceConfig from dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            ReinforceConfig instance loaded from the dictionary

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
            if field.default == dataclasses.MISSING and field.default_factory == dataclasses.MISSING
        }

        missing_params = required_params - set(config_dict.keys())
        if missing_params:
            raise KeyError(f"Missing required parameters in hyper_params: {missing_params}")

        # Filter config_dict to only include valid dataclass fields
        filtered_params = {k: v for k, v in config_dict.items() if k in field_names}

        return cls(**filtered_params)
