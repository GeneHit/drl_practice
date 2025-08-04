import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Generator

import pytest

from practice.exercise10_ddp_ppo.config_ddp_reacher import generate_context, get_app_config
from practice.exercise10_ddp_ppo.ppo_rnd_exercise import ContPPOConfig
from practice.utils.play_utils_new import play_and_generate_video_generic
from practice.utils.train_utils_new import train_and_evaluate_network


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def _minimal_ppo_config(temp_output_dir: Path) -> ContPPOConfig:
    """Return a minimal ContPPOConfig for fast testing."""
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
    # total_steps = timesteps * rollout_len * vector_env_num = 2 * 8 * 2 = 32
    minimal_config = replace(
        config,
        timesteps=2,  # minimal timesteps for testing
        rollout_len=8,  # smaller rollout length for testing
        num_epochs=2,  # fewer epochs for testing
        minibatch_num=2,  # smaller minibatch number for testing
        eval_episodes=2,
        eval_video_num=None,
        env_config=env_config,
        artifact_config=artifact_config,
    )
    return minimal_config


def test_ppo_trainer_basic_flow(temp_output_dir: Path) -> None:
    """Test ContPPOTrainer training flow with minimal config and no file saving."""
    config = _minimal_ppo_config(temp_output_dir)
    context = generate_context(config)
    try:
        train_and_evaluate_network(config=config, ctx=context)
        # don't save the video for online CI testing
        play_and_generate_video_generic(config=config, ctx=context, save_video=False)
    finally:
        # Clean up environments
        # For vector envs, use train_env and eval_env directly
        if hasattr(context, "train_env") and context.train_env:
            context.train_env.close()
        if hasattr(context, "eval_env") and context.eval_env:
            context.eval_env.close()
