import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Generator

import pytest

from practice.core.utils.env_utils import get_device
from practice.core.utils.play_utils import play_and_generate_video_generic
from practice.core.utils.train_utils import train_and_evaluate_network
from practice.exercise12_mbpo.config_reacher import generate_context, get_app_config
from practice.exercise12_mbpo.mbpo_exercise import MBPOConfig


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def _minimal_mbpo_config(temp_output_dir: Path) -> MBPOConfig:
    """Return a minimal MBPOConfig for fast testing."""
    config = get_app_config()
    # Reduce steps and vector_env_num for fast test
    artifact_config = replace(
        config.artifact_config,
        output_dir=str(temp_output_dir),
        save_result=True,
    )
    env_config = replace(
        config.env_config,
        vector_env_num=1,  # minimal parallel envs
        use_multi_processing=False,
    )
    # Reduce model-based config for faster testing
    model_based_config = replace(
        config.model_based_config,
        num_models=2,  # fewer models for testing
        train=replace(
            config.model_based_config.train,
            epochs=2,  # fewer epochs for testing
            batch_size=32,  # smaller batch size
            early_stop_patience=3,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
        ),
    )
    # Reduce rollout config for faster testing
    model_rollout_config = replace(
        config.model_rollout_config,
        rollout_num=2,  # fewer rollouts
        replay_buffer_capacity=32,  # smaller buffer
    )
    # total_steps = total_steps // vector_env_num = 64 // 2 = 32
    minimal_config = replace(
        config,
        total_steps=32,
        update_start_step=8,  # start updating early for testing
        batch_size=8,  # smaller batch size for testing
        replay_buffer_capacity=128,  # smaller buffer for testing
        eval_episodes=2,
        eval_video_num=None,
        sac_update_interval=4,
        update_num_per_epoch=1,  # fewer updates per epoch
        model_update_interval=8,
        env_config=env_config,
        artifact_config=artifact_config,
        model_based_config=model_based_config,
        model_rollout_config=model_rollout_config,
    )
    return minimal_config


def test_mbpo_trainer_basic_flow(temp_output_dir: Path) -> None:
    """Test MBPOTrainer training flow with minimal config and no file saving."""
    config = _minimal_mbpo_config(temp_output_dir)
    context = generate_context(config)
    try:
        train_and_evaluate_network(config=config, ctx=context)
        play_and_generate_video_generic(config=config, ctx=context, save_video=False)
    finally:
        # Clean up environments
        # For vector envs, use train_env and eval_env directly
        if hasattr(context, "train_env") and context.train_env:
            context.train_env.close()
        if hasattr(context, "eval_env") and context.eval_env:
            context.eval_env.close()


def test_mbpo_trainer_with_device(temp_output_dir: Path) -> None:
    """Test MBPOTrainer training flow with device detection."""
    device = get_device()
    # Only run this test if device is not CPU
    if device.type == "cpu":
        pytest.skip("Skipping device test on CPU")

    config = _minimal_mbpo_config(temp_output_dir)
    # Update config to use the detected device
    config = replace(config, device=device)
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
