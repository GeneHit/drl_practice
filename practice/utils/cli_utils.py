import importlib.util
from pathlib import Path
from typing import Union, cast

from hands_on.base import AgentBase
from hands_on.exercise2_dqn.dqn_exercise import EnvType
from hands_on.utils.evaluation_utils import play_game_once
from practice.base.config import BaseConfig
from practice.base.context import ContextBase


def load_config_module(
    config_path: str, mode: str
) -> tuple[BaseConfig, Union[ContextBase, EnvType]]:
    """Load a Python config module and extract config and context/env based on mode.

    Args:
        config_path: Path to Python config file
        mode: Mode of operation - determines what to load

    Returns:
        For train mode: Tuple of (config, context)
        For play/push_to_hub modes: Tuple of (config, env)

    Raises:
        ImportError: If config module cannot be loaded
        AttributeError: If required functions are missing
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if not config_path.endswith(".py"):
        raise ValueError(f"Config file must be a Python file (.py): {config_path}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config module from: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Extract required functions
    if not hasattr(module, "get_app_config"):
        raise AttributeError(f"Config module must define 'get_app_config' function: {config_path}")

    # Get config
    config = module.get_app_config()

    if mode == "train":
        # For train mode, we need the full context
        if not hasattr(module, "generate_context"):
            raise AttributeError(
                f"Config module must define 'generate_context' function: {config_path}"
            )
        context = module.generate_context(config)
        return cast(BaseConfig, config), cast(ContextBase, context)
    else:
        # For play and push_to_hub modes, we only need the environment
        if not hasattr(module, "get_env_for_play_and_hub"):
            raise AttributeError(
                f"Config module must define 'get_env_for_play_and_hub' function: {config_path}"
            )
        env = module.get_env_for_play_and_hub(config)
        return cast(BaseConfig, config), cast(EnvType, env)


def play_and_generate_video_generic(config: BaseConfig, env: EnvType) -> None:
    """Generic function to play game and generate video.

    Args:
        cfg_data: Configuration data
        env_creator: Function that creates environment from env_params
        model_loader: Model class with load_from_checkpoint method
        device: Device to load the model on
        fps: Video frame rate
        seed: Random seed for gameplay
    """
    # Load the trained model
    model = _load_model_from_config(config)
    artifact_config = config.artifact_config

    # Get video path
    video_path = Path(artifact_config.output_dir) / artifact_config.replay_video_filename

    # Play the game and save video
    play_game_once(
        env=env,
        policy=model,
        save_video=True,
        video_pathname=str(video_path),
        fps=artifact_config.fps,
        seed=artifact_config.seed,
    )
    print(f"Game replay saved to: {video_path}")


def _load_model_from_config(cfg: BaseConfig) -> AgentBase:
    model_path = Path(cfg.artifact_config.output_dir) / cfg.artifact_config.model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return cfg.artifact_config.agent_type.load_from_checkpoint(str(model_path), device=cfg.device)
