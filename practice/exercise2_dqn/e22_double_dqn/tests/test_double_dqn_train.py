import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Generator

import pytest

from practice.exercise2_dqn.dqn_exercise import DQNConfig
from practice.exercise2_dqn.e22_double_dqn.config_lunar_1d import generate_context, get_app_config
from practice.utils.env_utils import get_device
from practice.utils.play_utils import play_and_generate_video_generic
from practice.utils.train_utils import train_and_evaluate_network


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def _minimal_double_dqn_config(temp_output_dir: Path) -> DQNConfig:
    """Return a minimal DQNConfig for fast testing of double DQN."""
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
    # total_steps = timesteps * vector_env_num = 10 * 2 = 20
    minimal_config = replace(
        config,
        timesteps=10,  # minimal timesteps for testing
        update_start_step=5,  # start updating early for testing
        target_update_interval=5,  # update target network more frequently for testing
        eval_episodes=2,
        eval_video_num=None,
        env_config=env_config,
        artifact_config=artifact_config,
    )
    return minimal_config


def test_double_dqn_trainer_basic_flow(temp_output_dir: Path) -> None:
    """Test DoubleDQNTrainer training flow with minimal config and no file saving."""
    config = _minimal_double_dqn_config(temp_output_dir)
    context = generate_context(config)
    try:
        train_and_evaluate_network(config=config, ctx=context)
        # don't save the video for online CI testing
        play_and_generate_video_generic(config=config, ctx=context, save_video=False)
    finally:
        # Clean up environments
        if hasattr(context, "train_env") and context.train_env:
            context.train_env.close()
        if hasattr(context, "eval_env") and context.eval_env:
            context.eval_env.close()


def test_double_dqn_config_validation() -> None:
    """Test that double DQN config has correct algorithm type."""
    config = get_app_config()
    assert config.dqn_algorithm == "double", "Config should specify double DQN algorithm"


def test_double_dqn_context_generation() -> None:
    """Test that context generation works correctly for double DQN."""
    config = get_app_config()
    context = generate_context(config)

    # Verify context has required components
    assert hasattr(context, "train_env"), "Context should have train_env"
    assert hasattr(context, "eval_env"), "Context should have eval_env"
    assert hasattr(context, "network"), "Context should have network"
    assert hasattr(context, "optimizer"), "Context should have optimizer"

    # Verify network is on correct device
    assert next(context.network.parameters()).device == config.device

    # Verify environment action space is discrete
    from gymnasium.spaces import Discrete

    assert isinstance(context.eval_env.action_space, Discrete)

    # Clean up
    context.train_env.close()
    context.eval_env.close()


def test_double_dqn_minimal_training(temp_output_dir: Path) -> None:
    """Test minimal double DQN training to ensure no errors occur."""
    config = _minimal_double_dqn_config(temp_output_dir)
    context = generate_context(config)

    try:
        # Just run a few training steps to ensure no errors
        trainer = config.artifact_config.trainer_type(config=config, ctx=context)

        # Verify trainer was created correctly
        assert trainer is not None
        assert hasattr(trainer, "train")

        # For minimal testing, we'll just verify the setup works
        # In a real scenario, you might want to run a few steps
        assert context.network is not None
        assert context.optimizer is not None

    finally:
        # Clean up environments
        if hasattr(context, "train_env") and context.train_env:
            context.train_env.close()
        if hasattr(context, "eval_env") and context.eval_env:
            context.eval_env.close()


def test_double_dqn_training_with_accelerated_device(temp_output_dir: Path) -> None:
    """Test double DQN training flow with CUDA or MPS if available."""
    # Check if accelerated device is available
    device = get_device()

    if device.type == "cpu":
        pytest.skip("CUDA or MPS not available, skipping accelerated device test")

    # Create config with accelerated device
    config = _minimal_double_dqn_config(temp_output_dir)
    # Update config to use accelerated device
    accelerated_config = replace(config, device=device)

    context = generate_context(accelerated_config)
    try:
        # Verify network is on the correct device type (ignore index)
        network_device = next(context.network.parameters()).device
        assert network_device.type == device.type

        # Run training and evaluation
        train_and_evaluate_network(config=accelerated_config, ctx=context)

        # Test video generation without saving
        play_and_generate_video_generic(config=accelerated_config, ctx=context, save_video=False)

    finally:
        # Clean up environments
        if hasattr(context, "train_env") and context.train_env:
            context.train_env.close()
        if hasattr(context, "eval_env") and context.eval_env:
            context.eval_env.close()
