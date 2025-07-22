import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Generator

import pytest

from practice.exercise6_a3c.a3c_exercise import A3CConfig, a3c_train
from practice.exercise6_a3c.config_cartpole import get_app_config


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def _minimal_a3c_config(temp_output_dir: Path) -> A3CConfig:
    """Return a minimal A3CConfig for fast testing."""
    config = get_app_config()
    # Reduce steps, workers, and vector_env_num for fast test
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
    minimal_config = replace(
        config,
        num_workers=1,
        rollout_len=8,
        total_steps=32,  # just 4 rollouts per worker
        eval_episodes=2,
        eval_video_num=1,
        env_config=env_config,
        artifact_config=artifact_config,
    )
    return minimal_config


def test_a3c_train_basic_flow(temp_output_dir: Path) -> None:
    """Test a3c_train flow with minimal config and no file saving."""
    config = _minimal_a3c_config(temp_output_dir)
    a3c_train(config)
