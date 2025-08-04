import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Generator

import pytest

from practice.exercise8_td3.config_pendulum import generate_context, get_app_config
from practice.exercise8_td3.td3_exercise import TD3Config
from practice.utils.train_utils_new import train_and_evaluate_network


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def _minimal_td3_config(temp_output_dir: Path) -> TD3Config:
    """Return a minimal TD3Config for fast testing."""
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
    # total_steps = total_steps // vector_env_num = 32 // 2 = 16
    minimal_config = replace(
        config,
        total_steps=32,
        update_start_step=8,  # start updating early for testing
        batch_size=8,  # smaller batch size for testing
        replay_buffer_capacity=64,  # smaller buffer for testing
        eval_episodes=2,
        eval_video_num=1,
        env_config=env_config,
        artifact_config=artifact_config,
    )
    return minimal_config


def test_td3_trainer_basic_flow(temp_output_dir: Path) -> None:
    """Test TD3Trainer training flow with minimal config and no file saving."""
    config = _minimal_td3_config(temp_output_dir)
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
