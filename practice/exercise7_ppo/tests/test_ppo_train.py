import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Generator

import pytest

from practice.exercise7_ppo.config_cartpole import generate_context, get_app_config
from practice.exercise7_ppo.ppo_exercise import PPOConfig
from practice.utils.train_utils import train_and_evaluate_network


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def _minimal_ppo_config(temp_output_dir: Path) -> PPOConfig:
    """Return a minimal PPOConfig for fast testing."""
    config = get_app_config()
    # Reduce steps and vector_env_num for fast test
    artifact_config = replace(
        config.artifact_config,
        output_dir=str(temp_output_dir),
        save_result=True,
    )
    env_config = replace(
        config.env_config,
        vector_env_num=2,  # minimal parallel envs
        use_multi_processing=False,
    )
    # total_steps = rollout_len * vector_env_num * 2 (just 2 updates)
    minimal_config = replace(
        config,
        rollout_len=8,
        total_steps=32,
        eval_episodes=2,
        eval_video_num=1,
        env_config=env_config,
        artifact_config=artifact_config,
    )
    return minimal_config


def test_ppo_trainer_basic_flow(temp_output_dir: Path) -> None:
    """Test PPOTrainer training flow with minimal config and no file saving."""
    config = _minimal_ppo_config(temp_output_dir)
    context = generate_context(config)
    try:
        train_and_evaluate_network(config=config, ctx=context)
    finally:
        # Clean up environments
        # For vector envs, use train_env and eval_env directly
        if hasattr(context, "train_env") and context.train_env:
            context.train_env.close()
        if hasattr(context, "eval_env") and context.eval_env:
            context.eval_env.close()
