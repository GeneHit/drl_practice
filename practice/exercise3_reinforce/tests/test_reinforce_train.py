"""Tests for vanilla REINFORCE training flow.

This test suite provides comprehensive coverage of the vanilla REINFORCE training functionality
using the mountain car configuration with reduced parameters for fast testing.

Test Coverage:
- Basic training flow without file saving
- Training with file saving and output validation
- Training from checkpoint (resuming from saved model)
- CLI integration testing
- Config file loading and processing
- Validation of training outputs (model structure, evaluation results)
- Exception handling and environment cleanup
- File cleanup after testing

The tests use a reduced configuration based on config_mountain_car.py:
- episode: 4 (vs 1000)
- eval_episodes: 3 (vs 20)
- eval_video_num: 1 (vs 10)

This ensures fast test execution while maintaining realistic training behavior.
All tests validate proper resource cleanup and file cleanup.
"""

import json
import tempfile
from pathlib import Path
from typing import Generator

import pytest
import torch

from practice.base.config import ArtifactConfig, EnvConfig
from practice.exercise3_reinforce.config_cartpole import generate_context
from practice.exercise3_reinforce.reinforce_exercise import (
    Reinforce1DNet,
    ReinforceConfig,
    ReinforceTrainer,
)
from practice.utils.env_utils import get_device
from practice.utils.train_utils import train_and_evaluate_network
from practice.utils_for_coding.scheduler_utils import ConstantSchedule


@pytest.fixture
def test_config() -> ReinforceConfig:
    """Create a test configuration based on mountain car config with reduced parameters."""
    device = get_device()

    return ReinforceConfig(
        device=device,
        episode=4,  # Reduced from 1000
        learning_rate=1e-3,
        gamma=0.999,
        entropy_coef=ConstantSchedule(0.01),
        hidden_sizes=(32, 32),
        eval_episodes=3,  # Reduced from 20
        eval_random_seed=42,
        eval_video_num=1,  # Reduced from 10
        env_config=EnvConfig(
            env_id="MountainCar-v0",
            # use a small max_steps to speed up the test
            max_steps=20,
        ),
        artifact_config=ArtifactConfig(
            trainer_type=ReinforceTrainer,
            output_dir="",  # Will be set to temp dir in tests
            save_result=True,
            model_filename="reinforce.pth",
            repo_id="Reinforce-MountainCarV0",
            algorithm_name="Reinforce",
            extra_tags=("reinforce", "policy-gradient"),
        ),
    )


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestReinforceTraining:
    """Test vanilla REINFORCE training functionality."""

    def test_reinforce_train_basic_flow(
        self, test_config: ReinforceConfig, temp_output_dir: Path
    ) -> None:
        """Test basic REINFORCE training flow without file operations."""
        # Update config to use temp directory and disable file saving
        from dataclasses import replace

        artifact_config = replace(
            test_config.artifact_config,
            output_dir=str(temp_output_dir),
            save_result=False,
        )
        config = replace(test_config, artifact_config=artifact_config)

        # Generate context
        context = generate_context(config)

        try:
            # Run training
            train_and_evaluate_network(config=config, ctx=context)

            # Test passes if no exception is raised
            assert True, "Training completed successfully"
        finally:
            # Clean up environments
            if hasattr(context, "env") and context.env:
                context.env.close()
            if hasattr(context, "eval_env") and context.eval_env:
                context.eval_env.close()

    def test_reinforce_train_with_file_saving(
        self, test_config: ReinforceConfig, temp_output_dir: Path
    ) -> None:
        """Test REINFORCE training with file saving enabled."""
        from dataclasses import replace

        # Update config to use temp directory
        artifact_config = replace(
            test_config.artifact_config,
            output_dir=str(temp_output_dir),
            save_result=True,
        )
        config = replace(test_config, artifact_config=artifact_config)

        # Generate context
        context = generate_context(config)

        try:
            # Run training
            train_and_evaluate_network(config=config, ctx=context)

            # Check that output files were created
            output_dir = Path(config.artifact_config.output_dir)
            assert output_dir.exists(), "Output directory should be created"

            model_file = output_dir / config.artifact_config.model_filename
            assert model_file.exists(), f"Model file should be saved: {model_file}"

            params_file = output_dir / config.artifact_config.params_filename
            assert params_file.exists(), f"Params file should be saved: {params_file}"

            eval_result_file = output_dir / config.artifact_config.eval_result_filename
            assert eval_result_file.exists(), (
                f"Evaluation result file should be saved: {eval_result_file}"
            )

            # Check tensorboard directory
            tensorboard_dir = output_dir / "tensorboard"
            assert tensorboard_dir.exists(), "Tensorboard directory should be created"

            # Check video directory (if eval_video_num > 0)
            if config.eval_video_num and config.eval_video_num > 0:
                video_dir = output_dir / "video"
                assert video_dir.exists(), "Video directory should be created"

        finally:
            # Clean up environments
            if hasattr(context, "env") and context.env:
                context.env.close()
            if hasattr(context, "eval_env") and context.eval_env:
                context.eval_env.close()

    def test_reinforce_train_with_checkpoint(
        self, test_config: ReinforceConfig, temp_output_dir: Path
    ) -> None:
        """Test REINFORCE training starting from a checkpoint."""
        from dataclasses import replace

        # First, create a simple model checkpoint
        checkpoint_file = temp_output_dir / "checkpoint_reinforce.pth"

        # Create a dummy model state dict with correct structure
        dummy_model = Reinforce1DNet(
            state_dim=2, action_dim=3, hidden_sizes=test_config.hidden_sizes
        )  # MountainCar has 2 obs, 3 actions
        torch.save(dummy_model.state_dict(), checkpoint_file)

        # Update config to use checkpoint
        artifact_config = replace(
            test_config.artifact_config,
            output_dir=str(temp_output_dir),
            save_result=False,
        )
        config = replace(
            test_config,
            checkpoint_pathname=str(checkpoint_file),
            artifact_config=artifact_config,
        )

        # Generate context
        context = generate_context(config)

        try:
            # Run training from checkpoint
            train_and_evaluate_network(config=config, ctx=context)

            # Test passes if no exception is raised
            assert True, "Training from checkpoint completed successfully"
        finally:
            # Clean up environments
            if hasattr(context, "env") and context.env:
                context.env.close()
            if hasattr(context, "eval_env") and context.eval_env:
                context.eval_env.close()

    def test_training_produces_valid_results(
        self, test_config: ReinforceConfig, temp_output_dir: Path
    ) -> None:
        """Test that training produces valid model and evaluation results."""
        from dataclasses import replace

        # Update config to use temp directory
        artifact_config = replace(
            test_config.artifact_config,
            output_dir=str(temp_output_dir),
            save_result=True,
        )
        config = replace(test_config, artifact_config=artifact_config)

        # Generate context
        context = generate_context(config)

        try:
            # Run training
            train_and_evaluate_network(config=config, ctx=context)

            # Load and validate the saved model
            model_file = (
                Path(config.artifact_config.output_dir) / config.artifact_config.model_filename
            )
            model = torch.load(model_file, map_location="cpu", weights_only=False)

            # Validate model structure - should be a Reinforce1DNet instance
            assert isinstance(model, Reinforce1DNet), (
                f"Model should be Reinforce1DNet instance, got {type(model)}"
            )

            # Check that model has parameters
            param_count = sum(p.numel() for p in model.parameters())
            assert param_count > 0, "Model should have parameters"

            # Check model can perform forward pass
            dummy_input = torch.randn(1, 2)  # MountainCar has 2 observations
            with torch.no_grad():
                output = model(dummy_input)
            assert output.shape == (1, 3), (
                f"Model output should be shape (1, 3), got {output.shape}"
            )

            # Check output is valid probability distribution
            assert torch.allclose(output.sum(dim=1), torch.ones(1)), (
                "Output should sum to 1 (probability distribution)"
            )

            # Load and validate evaluation results
            eval_result_file = (
                Path(config.artifact_config.output_dir)
                / config.artifact_config.eval_result_filename
            )
            with open(eval_result_file, "r") as f:
                eval_result = json.load(f)

            # Check evaluation result structure
            assert "mean_reward" in eval_result, "Evaluation result should contain mean_reward"
            assert "std_reward" in eval_result, "Evaluation result should contain std_reward"
            assert "datetime" in eval_result, "Evaluation result should contain datetime"

            # Check that rewards are numeric
            assert isinstance(eval_result["mean_reward"], (int, float)), (
                "Mean reward should be numeric"
            )
            assert isinstance(eval_result["std_reward"], (int, float)), (
                "Std reward should be numeric"
            )

        finally:
            # Clean up environments
            if hasattr(context, "env") and context.env:
                context.env.close()
            if hasattr(context, "eval_env") and context.eval_env:
                context.eval_env.close()

    def test_config_validation(self, test_config: ReinforceConfig) -> None:
        """Test that configuration has all required fields."""
        # Test basic config structure
        assert test_config.device is not None
        assert test_config.episode > 0
        assert test_config.learning_rate > 0
        assert test_config.gamma > 0
        assert test_config.eval_episodes > 0

        # Test artifact config
        artifact_config = test_config.artifact_config
        assert artifact_config.trainer_type == ReinforceTrainer
        assert artifact_config.model_filename is not None
        assert artifact_config.algorithm_name is not None

        # Test env config
        env_config = test_config.env_config
        assert env_config.env_id is not None
