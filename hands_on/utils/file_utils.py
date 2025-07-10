import json
from pathlib import Path
from typing import Any, cast

from hands_on.base import AgentBase


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


def save_model_and_result(
    cfg_data: dict[str, Any],
    train_result: dict[str, Any],
    eval_result: dict[str, Any],
    agent: AgentBase,
) -> None:
    """Save the model, parameters and the result to the JSON file."""
    save_result = cfg_data["output_params"].get("save_result", False)
    if not save_result:
        return

    # create the output directory
    output_params = cfg_data["output_params"]
    out_dir = Path(output_params["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    # save the model
    agent.only_save_model(str(out_dir / output_params["model_filename"]))
    # save the train result
    with open(out_dir / output_params["train_result_filename"], "w") as f:
        json.dump(train_result, f)
    # save all the config data
    with open(out_dir / output_params["params_filename"], "w") as f:
        json.dump(cfg_data, f)
    with open(out_dir / output_params["eval_result_filename"], "w") as f:
        json.dump(eval_result, f)
