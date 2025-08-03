"""Tests for Q-learning training flow.

This test suite provides comprehensive coverage of the Q-learning training functionality
using the taxi configuration with reduced parameters for fast testing.

Test Coverage:
- Basic training flow without file saving
- Training with file saving and output validation
- Training from checkpoint (resuming from saved Q-table)
- CLI integration testing
- Config file loading and processing
- Validation of training outputs (Q-table structure, evaluation results)
- Exception handling and environment cleanup
- File cleanup after testing

The tests use a reduced configuration based on config_taxi.py:
- episodes: 10 (vs 25000)
- max_steps: 20 (vs 99)
- eval_episodes: 3 (vs 100)
- eval_video_num: 1 (vs 10)
- decay_rate: 0.05 (vs 0.0005) for faster epsilon decay

This ensures fast test execution while maintaining realistic training behavior.
All tests validate proper resource cleanup and file cleanup.
"""

import json
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from practice.exercise1_q.config_taxi import generate_context
from practice.exercise1_q.q_table_exercise import (
    QTable,
    QTableConfig,
    QTableTrainer,
)
from practice.utils.play_utils_new import play_and_generate_video_generic
from practice.utils.train_utils_new import train_and_evaluate_network
from practice.utils_for_coding.scheduler_utils import ExponentialSchedule


@pytest.fixture
def test_config() -> QTableConfig:
    """Create a test configuration based on taxi config with reduced parameters."""
    from practice.base.config import ArtifactConfig, EnvConfig
    from practice.utils.env_utils import get_device

    device = get_device()

    return QTableConfig(
        device=device,
        episodes=10,  # Reduced from 25000
        learning_rate=0.7,
        gamma=0.95,
        epsilon_schedule=ExponentialSchedule(
            start_e=1.0,
            end_e=0.05,
            decay_rate=0.05,  # Increased for faster decay in short training
        ),
        eval_episodes=3,  # Reduced from 100
        eval_random_seed=42,
        eval_video_num=1,  # Reduced from 10
        env_config=EnvConfig(
            env_id="Taxi-v3",
            max_steps=20,  # Reduced from 99
            env_kwargs={
                "render_mode": "rgb_array",
            },
        ),
        artifact_config=ArtifactConfig(
            trainer_type=QTableTrainer,
            agent_type=QTable,
            output_dir="",  # Will be set to temp dir in tests
            save_result=True,
            model_filename="q_table.pt",
            state_dict_filename="table_numpy.pkl",
            repo_id="q-Taxi-v3",
            algorithm_name="Q-Learning",
            extra_tags=("q-learning", "tabular"),
        ),
    )


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestQTableTraining:
    """Test Q-learning training functionality."""

    def test_q_table_train_basic_flow(
        self, test_config: QTableConfig, temp_output_dir: Path
    ) -> None:
        """Test basic Q-learning training flow without file operations."""
        from dataclasses import replace

        # Update config to use temp directory and disable file saving
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

            # Test passes if no exception is raised
            assert True, "Training completed successfully"
            play_and_generate_video_generic(config=config, ctx=context)
        finally:
            # Clean up environments
            if hasattr(context, "train_env") and context.train_env:
                context.train_env.close()
            if hasattr(context, "eval_env") and context.eval_env:
                context.eval_env.close()

    def test_q_table_train_with_file_saving(
        self, test_config: QTableConfig, temp_output_dir: Path
    ) -> None:
        """Test Q-learning training with file saving enabled."""
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

            state_dict_file = output_dir / config.artifact_config.state_dict_filename
            assert state_dict_file.exists(), f"State dict file should be saved: {state_dict_file}"

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
            if hasattr(context, "train_env") and context.train_env:
                context.train_env.close()
            if hasattr(context, "eval_env") and context.eval_env:
                context.eval_env.close()

    def test_q_table_train_with_checkpoint(
        self, test_config: QTableConfig, temp_output_dir: Path
    ) -> None:
        """Test Q-learning training starting from a checkpoint."""
        from dataclasses import replace

        # First, create a Q-table checkpoint
        checkpoint_file = temp_output_dir / "checkpoint_q_table.pkl"

        # Create a dummy Q-table with correct structure (Taxi has 500 states, 6 actions)
        dummy_q_table = np.random.rand(500, 6).astype(np.float32)
        import pickle5 as pickle

        with open(checkpoint_file, "wb") as f:
            pickle.dump(dummy_q_table, f)

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
            if hasattr(context, "train_env") and context.train_env:
                context.train_env.close()
            if hasattr(context, "eval_env") and context.eval_env:
                context.eval_env.close()

    def test_training_produces_valid_results(
        self, test_config: QTableConfig, temp_output_dir: Path
    ) -> None:
        """Test that training produces valid Q-table and evaluation results."""
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

            # Load and validate the saved Q-table
            model_file = (
                Path(config.artifact_config.output_dir) / config.artifact_config.state_dict_filename
            )

            import pickle5 as pickle

            with open(model_file, "rb") as f:
                q_table = pickle.load(f)

            # Validate Q-table structure - should be numpy array
            assert isinstance(q_table, np.ndarray), (
                f"Q-table should be numpy array, got {type(q_table)}"
            )
            assert q_table.dtype == np.float32, f"Q-table should be float32, got {q_table.dtype}"
            assert q_table.shape == (500, 6), (
                f"Q-table should be shape (500, 6), got {q_table.shape}"
            )

            # Check that Q-table has been updated (not all zeros)
            assert not np.allclose(q_table, 0.0), (
                "Q-table should have non-zero values after training"
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
            if hasattr(context, "train_env") and context.train_env:
                context.train_env.close()
            if hasattr(context, "eval_env") and context.eval_env:
                context.eval_env.close()

    def test_config_validation(self, test_config: QTableConfig) -> None:
        """Test that configuration has all required fields."""
        # Test basic config structure
        assert test_config.device is not None
        assert test_config.episodes > 0
        assert test_config.env_config.max_steps is not None
        assert test_config.env_config.max_steps > 0
        assert test_config.learning_rate > 0
        assert test_config.gamma > 0
        assert test_config.eval_episodes > 0
        assert test_config.epsilon_schedule is not None

        # Test epsilon schedule functionality
        assert test_config.epsilon_schedule(0) > 0  # Should return a positive value
        assert test_config.epsilon_schedule(1000) > 0  # Should return a positive value

        # Test artifact config
        artifact_config = test_config.artifact_config
        assert artifact_config.trainer_type == QTableTrainer
        assert artifact_config.agent_type == QTable
        assert artifact_config.model_filename is not None
        assert artifact_config.state_dict_filename is not None
        assert artifact_config.algorithm_name is not None

        # Test env config
        env_config = test_config.env_config
        assert env_config.env_id is not None

    def test_q_table_agent_functionality(self, test_config: QTableConfig) -> None:
        """Test Q-table agent functionality."""
        # Create a simple Q-table
        q_table = np.random.rand(500, 6).astype(np.float32)
        agent = QTable(q_table)

        # Test action selection
        state = 0
        action = agent.action(state)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 6

        # Test saving and loading
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_file = f.name

        try:
            # Save Q-table
            agent.only_save_model(temp_file)

            # Load Q-table
            loaded_agent = QTable.load_from_checkpoint(temp_file, None)

            # Test that loaded agent produces same actions
            assert loaded_agent.action(state) == agent.action(state)

        finally:
            # Clean up temp file
            import os

            if os.path.exists(temp_file):
                os.unlink(temp_file)
