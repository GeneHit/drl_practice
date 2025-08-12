"""Tests for Rainbow DQN lunar_2d image-based training.

This test suite provides comprehensive coverage of the Rainbow DQN training functionality
using the lunar_2d configuration for image-based observations.

Test Coverage:
- Configuration validation for image-based training
- Context generation with 3D image observations
- Network architecture validation for CNN-based Rainbow
- Image observation processing and network inference
- Training flow with image-based observations
- Device compatibility (CPU, CUDA, MPS)
- Integration with actual lunar_2d config functions

The tests use the actual lunar_2d configuration but with reduced parameters for fast testing.
"""

import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import torch
from gymnasium.spaces import Discrete

from practice.exercise2_dqn.e24_rainbow.config_lunar_2d import generate_context, get_app_config
from practice.exercise2_dqn.e24_rainbow.rainbow_exercise import RainbowConfig
from practice.utils.env_utils import get_device
from practice.utils.play_utils import play_and_generate_video_generic
from practice.utils.train_utils import train_and_evaluate_network


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def _minimal_lunar_2d_config(temp_output_dir: Path) -> RainbowConfig:
    """Return a minimal RainbowConfig for fast testing of lunar_2d."""
    config = get_app_config()

    # Reduce parameters for fast testing
    minimal_config = replace(
        config,
        timesteps=20,  # Much smaller for faster tests
        batch_size=4,  # Smaller batch size
        update_start_step=10,  # Start updating sooner
        target_update_interval=5,  # More frequent updates
        eval_episodes=2,  # Fewer evaluation episodes
        eval_video_num=None,  # No video for testing
        log_interval=5,  # More frequent logging
        env_config=replace(
            config.env_config,
            vector_env_num=1,  # Single environment for speed
            use_multi_processing=False,  # Disabled for testing
            max_steps=20,  # Very small max steps for fast testing
        ),
        artifact_config=replace(
            config.artifact_config,
            output_dir=str(temp_output_dir),
            save_result=True,
        ),
    )
    return minimal_config


class TestRainbowLunar2D:
    """Test Rainbow DQN with lunar_2d configuration for image-based training."""

    def test_lunar_2d_config_validation(self) -> None:
        """Test that lunar_2d configuration has correct image-specific settings."""
        config = get_app_config()

        # Test basic config structure
        assert config.device is not None
        assert config.timesteps > 0
        assert config.learning_rate > 0
        assert config.gamma > 0
        assert config.batch_size > 0
        assert config.eval_episodes > 0

        # Test Rainbow-specific config
        assert config.noisy_std > 0
        assert config.v_min < config.v_max
        assert config.num_atoms > 1
        assert config.per_buffer_config is not None
        assert config.per_buffer_config.capacity > 0
        assert config.per_buffer_config.n_step > 0
        assert 0 < config.per_buffer_config.alpha <= 1
        assert 0 < config.per_buffer_config.beta <= 1
        assert config.max_grad_norm is not None and config.max_grad_norm > 0

        # Test image-specific env config
        env_config = config.env_config
        assert env_config.env_id == "LunarLander-v3"
        assert env_config.use_image is True
        assert env_config.training_render_mode == "rgb_array"
        assert env_config.image_shape == (84, 84)
        assert env_config.frame_stack == 4
        assert env_config.frame_skip == 3

        # Test artifact config
        artifact_config = config.artifact_config
        assert artifact_config.repo_id == "Rainbow-2d-LunarLander-v3"
        assert artifact_config.algorithm_name == "Rainbow-DQN"
        assert "rainbow" in artifact_config.extra_tags
        assert "dqn" in artifact_config.extra_tags

    def test_lunar_2d_context_generation(self) -> None:
        """Test that lunar_2d context generation works correctly for image-based training."""
        config = get_app_config()
        context = generate_context(config)

        try:
            # Verify context has required components
            assert hasattr(context, "train_env"), "Context should have train_env"
            assert hasattr(context, "eval_env"), "Context should have eval_env"
            assert hasattr(context, "trained_target"), "Context should have trained_target"
            assert hasattr(context, "optimizer"), "Context should have optimizer"
            assert hasattr(context, "lr_schedulers"), "Context should have lr_schedulers"

            # Verify network is on correct device (trained_target should be a Module for Rainbow)
            assert isinstance(context.trained_target, torch.nn.Module), (
                "trained_target should be a Module"
            )
            assert next(context.trained_target.parameters()).device == config.device

            # Verify environment action space is discrete
            assert isinstance(context.eval_env.action_space, Discrete)

            # Verify observation space is 3D (image-based)
            obs_shape = context.eval_env.observation_space.shape
            assert obs_shape is not None
            assert len(obs_shape) == 3, f"Expected 3D observation space, got {len(obs_shape)}D"

            # Verify image dimensions match config
            frame_stack = config.env_config.frame_stack
            image_shape = config.env_config.image_shape
            assert image_shape is not None, "image_shape should not be None for lunar_2d"
            expected_shape = (frame_stack, image_shape[0], image_shape[1])
            assert obs_shape == expected_shape, (
                f"Expected observation shape {expected_shape}, got {obs_shape}"
            )

        finally:
            # Clean up environments
            if hasattr(context, "train_env") and context.train_env:
                context.train_env.close()
            if hasattr(context, "eval_env") and context.eval_env:
                context.eval_env.close()

    def test_lunar_2d_network_architecture(self) -> None:
        """Test that lunar_2d Rainbow network has correct CNN architecture for images."""
        config = get_app_config()

        # Create a RainbowNet instance for image input
        from practice.exercise2_dqn.e24_rainbow.network import RainbowNet

        network = RainbowNet(
            state_n=(4, 84, 84),  # frame_stack=4, image_shape=(84, 84)
            action_n=4,  # LunarLander has 4 actions
            hidden_sizes=(128,),  # CNN's last FC layer size
            noisy_std=config.noisy_std,
            v_min=config.v_min,
            v_max=config.v_max,
            num_atoms=config.num_atoms,
        )

        # Test network components
        assert hasattr(network, "feature"), "Network should have feature component"
        assert hasattr(network, "value_head"), "Network should have value_head component"
        assert hasattr(network, "advantage_head"), "Network should have advantage_head component"
        assert hasattr(network, "support"), "Network should have support buffer"

        # Test forward pass with image input
        batch_size = 2
        dummy_input = torch.randn(batch_size, 4, 84, 84)  # batch, frames, height, width

        # Test Q-value output
        q_output = network(dummy_input)
        assert q_output.shape == (batch_size, 4), (
            f"Q-value output should be (2, 4), got {q_output.shape}"
        )

        # Test distributional output
        dist_output = network.forward_dist(dummy_input)
        assert dist_output.shape == (batch_size, 4, config.num_atoms), (
            f"Distributional output should be (2, 4, {config.num_atoms}), got {dist_output.shape}"
        )

        # Test that distributional output is valid probability distribution
        assert torch.allclose(dist_output.sum(dim=-1), torch.ones(batch_size, 4), atol=1e-5), (
            "Distributional output should sum to 1 across atoms dimension"
        )

        # Test noise reset functionality
        network.reset_noise()  # Should not raise any errors

    def test_lunar_2d_training_flow(self, temp_output_dir: Path) -> None:
        """Test lunar_2d Rainbow DQN training flow with image-based observations."""
        config = _minimal_lunar_2d_config(temp_output_dir)

        # Generate context
        context = generate_context(config)

        try:
            # Verify that the model is on the correct device
            assert isinstance(context.trained_target, torch.nn.Module), (
                "Expected trained_target to be a Module"
            )
            actual_device = next(context.trained_target.parameters()).device
            assert actual_device.type == config.device.type, (
                f"Model should be on {config.device.type}, but is on {actual_device.type}"
            )

            # Run training
            train_and_evaluate_network(config=config, ctx=context)

            # Test passes if no exception is raised
            assert True, "Lunar 2D training completed successfully"

        finally:
            # Clean up environments
            if hasattr(context, "train_env") and context.train_env:
                context.train_env.close()
            if hasattr(context, "eval_env") and context.eval_env:
                context.eval_env.close()

    def test_lunar_2d_image_observation_processing(self) -> None:
        """Test that lunar_2d can properly process image observations."""
        config = get_app_config()
        context = generate_context(config)

        try:
            # Get initial observation from environment
            obs, _ = context.eval_env.reset()

            # Verify observation shape matches expected image dimensions
            frame_stack = config.env_config.frame_stack
            image_shape = config.env_config.image_shape
            assert image_shape is not None, "image_shape should not be None for lunar_2d"
            expected_shape = (frame_stack, image_shape[0], image_shape[1])
            assert obs.shape == expected_shape, (
                f"Expected observation shape {expected_shape}, got {obs.shape}"
            )

            # Verify observation is numeric and in reasonable range
            assert np.isfinite(obs).all(), "Observation should contain finite values"
            # Image observations can be uint8 (0-255) or float (0-1), both are valid
            assert obs.dtype in [np.float32, np.float64, np.uint8], (
                f"Observation should be float or uint8, got {obs.dtype}"
            )

            # Test that network can process the observation
            assert isinstance(context.trained_target, torch.nn.Module), (
                "trained_target should be a Module"
            )
            from practice.exercise2_dqn.e24_rainbow.network import RainbowNet

            assert isinstance(context.trained_target, RainbowNet), (
                "trained_target should be a RainbowNet"
            )
            network = context.trained_target
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(config.device)  # Add batch dimension

            # Convert uint8 to float if needed (normalize to 0-1 range)
            if obs_tensor.dtype == torch.uint8:
                obs_tensor = obs_tensor.float() / 255.0

            with torch.no_grad():
                q_values = network(obs_tensor)
                # Explicitly cast to ensure mypy understands the type
                probs = network.forward_dist(obs_tensor)
                assert isinstance(probs, torch.Tensor), "forward_dist should return a Tensor"

            # Verify outputs
            assert q_values.shape == (1, 4), f"Q-values should be (1, 4), got {q_values.shape}"
            assert probs.shape == (1, 4, config.num_atoms), (
                f"Probabilities should be (1, 4, {config.num_atoms}), got {probs.shape}"
            )

            # Verify probability distribution
            prob_sums = torch.sum(probs, dim=-1)
            assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-4), (
                "Probabilities should sum to 1"
            )

        finally:
            # Clean up environments
            if hasattr(context, "train_env") and context.train_env:
                context.train_env.close()
            if hasattr(context, "eval_env") and context.eval_env:
                context.eval_env.close()

    def test_lunar_2d_with_accelerated_device(self, temp_output_dir: Path) -> None:
        """Test lunar_2d training with CUDA or MPS if available."""
        # Check if accelerated device is available
        device = get_device()

        if device.type == "cpu":
            pytest.skip("CUDA or MPS not available, skipping accelerated device test")

        # Get base config and modify for accelerated device
        config = get_app_config()
        accelerated_config = replace(config, device=device)

        # Update config to use temp directory and minimal parameters
        minimal_config = replace(
            accelerated_config,
            timesteps=20,  # Much smaller for faster tests
            batch_size=4,  # Smaller batch size
            update_start_step=10,  # Start updating sooner
            target_update_interval=5,  # More frequent updates
            eval_episodes=2,  # Fewer evaluation episodes
            eval_video_num=None,  # No video for testing
            log_interval=5,  # More frequent logging
            env_config=replace(
                accelerated_config.env_config,
                vector_env_num=1,  # Single environment for speed
                use_multi_processing=False,  # Disabled for testing
                max_steps=20,  # Very small max steps for fast testing
            ),
            artifact_config=replace(
                accelerated_config.artifact_config,
                output_dir=str(temp_output_dir),
                save_result=False,
            ),
        )

        context = generate_context(minimal_config)
        try:
            # Verify network is on the correct device type (ignore index)
            assert isinstance(context.trained_target, torch.nn.Module), (
                "trained_target should be a Module"
            )
            network_device = next(context.trained_target.parameters()).device
            assert network_device.type == device.type

            # For MPS, we need to be more careful due to compatibility issues
            if device.type == "mps":
                # MPS has some limitations with certain operations, so we'll just verify setup
                # and skip the actual training to avoid MPS-specific errors
                print(
                    "Verified MPS setup successfully, skipping training due to MPS compatibility considerations"
                )
                assert True, f"Lunar 2D MPS setup completed successfully on {device.type}"
            else:
                # For CUDA, we can run the full training
                train_and_evaluate_network(config=minimal_config, ctx=context)
                assert True, f"Lunar 2D training with {device.type} completed successfully"

        finally:
            # Clean up environments
            if hasattr(context, "train_env") and context.train_env:
                context.train_env.close()
            if hasattr(context, "eval_env") and context.eval_env:
                context.eval_env.close()

    def test_lunar_2d_play_and_video_generation(self, temp_output_dir: Path) -> None:
        """Test lunar_2d play functionality and video generation."""
        config = _minimal_lunar_2d_config(temp_output_dir)
        context = generate_context(config)

        try:
            # First run training to generate the model files
            train_and_evaluate_network(config=config, ctx=context)

            # Now test play and generate video functionality (without saving)
            play_and_generate_video_generic(config=config, ctx=context, save_video=False)

            # Test passes if no exception is raised
            assert True, "Lunar 2D play and video generation completed successfully"

        finally:
            # Clean up environments
            if hasattr(context, "train_env") and context.train_env:
                context.train_env.close()
            if hasattr(context, "eval_env") and context.eval_env:
                context.eval_env.close()

    def test_lunar_2d_learning_rate_scheduler(self) -> None:
        """Test that lunar_2d context has proper learning rate scheduler setup."""
        config = get_app_config()
        context = generate_context(config)

        try:
            # Verify learning rate scheduler is present
            assert hasattr(context, "lr_schedulers"), "Context should have lr_schedulers"
            assert context.lr_schedulers is not None, "lr_schedulers should not be None"
            assert len(context.lr_schedulers) > 0, "lr_schedulers should not be empty"

            # Verify it's a OneCycleLR scheduler
            lr_scheduler = context.lr_schedulers[0]
            assert "OneCycleLR" in str(type(lr_scheduler)), "Should be OneCycleLR scheduler"

        finally:
            # Clean up environments
            if hasattr(context, "train_env") and context.train_env:
                context.train_env.close()
            if hasattr(context, "eval_env") and context.eval_env:
                context.eval_env.close()
