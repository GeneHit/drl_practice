import json
from pathlib import Path
from typing import Any, cast


def load_config_from_json(config_path: str | Path) -> dict[str, Any]:
    """Load the Q-learning training configuration from a JSON file.

    Args:
        config_path: The path to the JSON configuration file

    Returns:
        dict[str, Any]: The configuration loaded from the JSON file
    """
    with open(config_path, "r") as f:
        config_data = cast(dict[str, Any], json.load(f))
    return config_data
