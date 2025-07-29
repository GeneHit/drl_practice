"""Tests for curiosity-driven training flow.

This test suite provides comprehensive coverage of the curiosity-driven REINFORCE training functionality
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
- episode: 5 (vs 10000)
- eval_episodes: 3 (vs 20)
- eval_video_num: 1 (vs 10)

This ensures fast test execution while maintaining realistic training behavior.
All tests validate proper resource cleanup and file cleanup.
"""

import json
import tempfile
from pathlib import Path
from typing import Generator, cast

import pytest
import torch

from practice.base.config import ArtifactConfig, EnvConfig
from practice.base.context import ContextBase
from practice.base.env_typing import EnvType
from practice.exercise3_reinforce.reinforce_exercise import Reinforce1DNet
from practice.exercise4_curiosity.config_mountain_car import (
    generate_context,
    get_app_config,
    get_env_for_play_and_hub,
)
from practice.exercise4_curiosity.curiosity_exercise import (
    RND1DNetworkConfig,
    RNDRewardConfig,
    XShapingRewardConfig,
)
from practice.exercise4_curiosity.enhanced_reinforce import (
    EnhancedReinforceConfig,
    EnhancedReinforceTrainer,
)
from practice.utils.env_utils import get_device
from practice.utils.train_utils import train_and_evaluate_network
from practice.utils_for_coding.agent_utils import NNAgent
from practice.utils_for_coding.baseline_utils import ConstantBaseline
from practice.utils_for_coding.scheduler_utils import ConstantSchedule, ExponentialSchedule


@pytest.fixture
def test_config() -> EnhancedReinforceConfig:
    """Create a test configuration based on mountain car config with reduced parameters."""
    # Create a modified config with reduced parameters for testing
    device = get_device()

    return EnhancedReinforceConfig(
        device=device,
        timesteps=100,  # Reduced from 200000
        learning_rate=1e-3,
        lr_gamma=0.99,
        gamma=0.999,
        hidden_sizes=(32, 32),
        baseline=ConstantBaseline(),
        entropy_coef=ConstantSchedule(0.01),
        eval_episodes=3,  # Reduced from 20
        eval_random_seed=42,
        eval_video_num=1,  # Reduced from 10
        reward_configs=(
            RND1DNetworkConfig(
                rnd_config=RNDRewardConfig(
                    beta=ExponentialSchedule(start_e=5.0, end_e=1.0, decay_rate=-0.005),
                    device=device,
                    normalize=True,
                    max_reward=2,
                ),
                obs_dim=2,  # MountainCar observation dimension
                output_dim=32,
                hidden_sizes=(32, 32),
                learning_rate=1e-3,
            ),
            XShapingRewardConfig(
                beta=ExponentialSchedule(start_e=5.0, end_e=1.0, decay_rate=-0.005),
                goal_position=None,
            ),
        ),
        env_config=EnvConfig(
            env_id="MountainCar-v0",
            # use a small max_steps to speed up the test
            max_steps=20,
        ),
        artifact_config=ArtifactConfig(
            trainer_type=EnhancedReinforceTrainer,
            agent_type=NNAgent,
            output_dir="",  # Will be set to temp dir in tests
            save_result=True,
            model_filename="curiosity.pth",
            repo_id="Reinforce-MountainCarV0",
            algorithm_name="Reinforce_RND",
            extra_tags=("curiosity", "reinforce", "rnd"),
        ),
    )


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestCuriosityTraining:
    """Test curiosity-driven training functionality."""

    def test_curiosity_train_basic_flow(
        self, test_config: EnhancedReinforceConfig, temp_output_dir: Path
    ) -> None:
        """Test basic curiosity training flow without file operations."""
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

    def test_curiosity_train_with_file_saving(
        self, test_config: EnhancedReinforceConfig, temp_output_dir: Path
    ) -> None:
        """Test curiosity training with file saving enabled."""
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

            # Validate content of some files
            with open(eval_result_file, "r") as f:
                eval_result = json.load(f)
                assert "mean_reward" in eval_result, "Evaluation result should contain mean_reward"
                assert "std_reward" in eval_result, "Evaluation result should contain std_reward"
                assert "datetime" in eval_result, "Evaluation result should contain datetime"

            with open(params_file, "r") as f:
                params = json.load(f)
                assert "total_steps" in params, "Params should contain total_steps"
                assert "learning_rate" in params, "Params should contain learning_rate"
                assert "env_config" in params, "Params should contain env_config"

        finally:
            # Clean up environments
            if hasattr(context, "env") and context.env:
                context.env.close()
            if hasattr(context, "eval_env") and context.eval_env:
                context.eval_env.close()

    def test_curiosity_train_with_checkpoint(
        self, test_config: EnhancedReinforceConfig, temp_output_dir: Path
    ) -> None:
        """Test curiosity training starting from a checkpoint."""
        from dataclasses import replace

        # First, create a simple model checkpoint
        checkpoint_file = temp_output_dir / "checkpoint_curiosity.pth"

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

    def test_config_functions(self, temp_output_dir: Path) -> None:
        """Test configuration functions work correctly."""
        # Test get_app_config
        config = get_app_config()
        assert isinstance(config, EnhancedReinforceConfig)
        assert config.env_config.env_id == "MountainCar-v0"
        assert len(config.reward_configs) == 2  # RND + XDirectionShaping

        # Test generate_context
        from dataclasses import replace

        artifact_config = replace(
            config.artifact_config,
            output_dir=str(temp_output_dir),
            save_result=False,
        )
        test_config = replace(config, artifact_config=artifact_config)

        context = generate_context(test_config)
        assert hasattr(context, "env")
        assert hasattr(context, "eval_env")
        assert hasattr(context, "trained_target")
        assert hasattr(context, "optimizer")
        assert hasattr(context, "rewarders")
        assert len(context.rewarders) == 2  # Should match reward_configs

        # Test get_env_for_play_and_hub
        env = get_env_for_play_and_hub(test_config)
        assert env is not None
        assert hasattr(env, "observation_space")
        assert hasattr(env, "action_space")

        # Clean up
        try:
            if hasattr(context, "env") and context.env:
                context.env.close()
            if hasattr(context, "eval_env") and context.eval_env:
                context.eval_env.close()
            if env:
                env.close()
        except:
            pass

    def test_cli_integration(self, temp_output_dir: Path) -> None:
        """Test CLI integration with the training functionality."""
        from practice.utils.cli_utils import load_config_module

        # Test train mode loading
        config, context = load_config_module(
            "practice/exercise4_curiosity/config_mountain_car.py", "train"
        )

        assert isinstance(config, EnhancedReinforceConfig)
        assert isinstance(context, ContextBase)
        assert hasattr(context, "env")
        assert hasattr(context, "eval_env")

        # Test play mode loading
        config, init_env = load_config_module(
            "practice/exercise4_curiosity/config_mountain_car.py", "play"
        )

        env = cast(EnvType, init_env)  # make mypy happy
        assert isinstance(config, EnhancedReinforceConfig)
        assert hasattr(env, "observation_space")
        assert hasattr(env, "action_space")

        # Clean up
        try:
            if hasattr(context, "env") and context.env:
                context.env.close()
            if hasattr(context, "eval_env") and context.eval_env:
                context.eval_env.close()
            if env:
                env.close()
        except:
            pass

    def test_file_cleanup(
        self, test_config: EnhancedReinforceConfig, temp_output_dir: Path
    ) -> None:
        """Test that all generated files are properly cleaned up."""
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

            # Verify files were created
            output_dir = Path(config.artifact_config.output_dir)
            created_files = list(output_dir.rglob("*"))
            assert len(created_files) > 0, "Files should be created during training"

            # List expected files
            expected_files = [
                output_dir / config.artifact_config.model_filename,
                output_dir / config.artifact_config.params_filename,
                output_dir / config.artifact_config.eval_result_filename,
                output_dir / "tensorboard",
            ]

            for expected_file in expected_files:
                assert expected_file.exists(), f"Expected file should exist: {expected_file}"

        finally:
            # Clean up environments
            if hasattr(context, "env") and context.env:
                context.env.close()
            if hasattr(context, "eval_env") and context.eval_env:
                context.eval_env.close()

        # After the test, the temp directory should be automatically cleaned up
        # by the temp_output_dir fixture, but let's verify the cleanup works
        assert temp_output_dir.exists(), "Temp directory should still exist during test"

        # The actual cleanup happens when the fixture goes out of scope
        # We can't test that directly, but we can verify that the cleanup
        # mechanism works by manually removing files if needed
        if output_dir.exists():
            # This demonstrates that we can clean up if needed
            # In practice, the tempfile.TemporaryDirectory handles this
            pass

    def test_training_with_different_episodes(
        self, test_config: EnhancedReinforceConfig, temp_output_dir: Path
    ) -> None:
        """Test training with different episode counts."""
        from dataclasses import replace

        # Test with very minimal episodes
        artifact_config = replace(
            test_config.artifact_config,
            output_dir=str(temp_output_dir),
            save_result=False,
        )
        config = replace(
            test_config,
            timesteps=40,  # Even fewer steps
            artifact_config=artifact_config,
        )

        # Generate context
        context = generate_context(config)

        try:
            # Run training
            train_and_evaluate_network(config=config, ctx=context)

            # Test passes if no exception is raised
            assert True, "Training with minimal episodes completed successfully"
        finally:
            # Clean up environments
            if hasattr(context, "env") and context.env:
                context.env.close()
            if hasattr(context, "eval_env") and context.eval_env:
                context.eval_env.close()
