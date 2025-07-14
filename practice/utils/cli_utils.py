from pathlib import Path

from hands_on.base import AgentBase
from hands_on.exercise2_dqn.dqn_exercise import EnvType
from hands_on.utils.evaluation_utils import play_game_once
from practice.base.config import BaseConfig


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
