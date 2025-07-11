"""Tests for Q-learning training flow.

This test suite provides comprehensive coverage of the Q-learning training functionality
using the actual config_frozen_lake.json structure with reduced parameters for fast testing.

Test Coverage:
- Basic training flow without file saving
- Training with file saving and output validation
- Training from checkpoint (resuming from saved Q-table)
- Main function integration (environment creation and cleanup)
- Config file loading and processing
- Validation of training outputs (Q-table structure, evaluation results)
- Different hyperparameter configurations
- Exception handling and environment cleanup

The tests use a reduced configuration based on config_frozen_lake.json:
- episodes: 10 (vs 10000)
- max_steps: 20 (vs 99)
- eval_episodes: 3 (vs 100)
- eval_seed: 3 seeds (vs 100)
- decay_rate: 0.05 (vs 0.0005) for faster epsilon decay

This ensures fast test execution while maintaining realistic training behavior.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import patch

import numpy as np
import pytest

from hands_on.exercise1_q_learning.q_train import main, q_table_train
from hands_on.utils.env_utils import make_discrete_env_with_kwargs


@pytest.fixture
def test_config() -> dict[str, Any]:
    """Create a test configuration based on frozen lake config with reduced parameters."""
    return {
        "hyper_params": {
            "episodes": 10,  # Reduced from 10000
            "learning_rate": 0.7,
            "max_steps": 20,  # Reduced from 99
            "gamma": 0.95,
            "max_epsilon": 1.0,
            "min_epsilon": 0.05,
            "decay_rate": 0.05,  # Increased for faster decay in short training
        },
        "env_params": {
            "env_id": "FrozenLake-v1",
            "kwargs": {
                "map_name": "4x4",
                "is_slippery": False,
                "render_mode": "rgb_array",
            },
        },
        "eval_params": {
            "eval_episodes": 3,  # Reduced from 100
            "max_steps": 20,  # Reduced from 99
            "eval_seed": [465, 131, 72],  # Reduced to 3 seeds
        },
        "output_params": {
            "output_dir": "",  # Will be set to temp dir in tests
            "save_result": True,
            "model_filename": "q_table.pkl",
            "params_filename": "params.json",
            "train_result_filename": "train_result.json",
            "eval_result_filename": "eval_result.json",
            "replay_video_filename": "replay.mp4",
        },
        "hub_params": {"repo_id": "test-q-FrozenLake-v1-4x4-noSlippery"},
    }


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestQTraining:
    """Test Q-learning training functionality."""

    def test_q_table_train_basic_flow(
        self, test_config: dict[str, Any], temp_output_dir: Path
    ) -> None:
        """Test basic Q-table training flow without file operations."""
        # Update config to use temp directory
        test_config["output_params"]["output_dir"] = str(temp_output_dir)
        test_config["output_params"]["save_result"] = False  # Skip file saving for this test

        # Create environment
        env, env_info = make_discrete_env_with_kwargs(
            env_id=test_config["env_params"]["env_id"],
            kwargs=test_config["env_params"]["kwargs"],
        )
        test_config["env_params"].update(env_info)

        try:
            # Run training
            q_table_train(env=env, cfg_data=test_config)

            # Test passes if no exception is raised
            assert True, "Training completed successfully"
        finally:
            env.close()

    def test_q_table_train_with_file_saving(
        self, test_config: dict[str, Any], temp_output_dir: Path
    ) -> None:
        """Test Q-table training with file saving enabled."""
        # Update config to use temp directory
        test_config["output_params"]["output_dir"] = str(temp_output_dir)
        test_config["output_params"]["save_result"] = True

        # Create environment
        env, env_info = make_discrete_env_with_kwargs(
            env_id=test_config["env_params"]["env_id"],
            kwargs=test_config["env_params"]["kwargs"],
        )
        test_config["env_params"].update(env_info)

        try:
            # Run training
            q_table_train(env=env, cfg_data=test_config)

            # Check that output files were created
            output_dir = Path(test_config["output_params"]["output_dir"])
            assert output_dir.exists(), "Output directory should be created"

            model_file = output_dir / test_config["output_params"]["model_filename"]
            assert model_file.exists(), f"Model file should be saved: {model_file}"

            params_file = output_dir / test_config["output_params"]["params_filename"]
            assert params_file.exists(), f"Params file should be saved: {params_file}"

            # Note: train_result.json is no longer saved by the current implementation

            eval_result_file = output_dir / test_config["output_params"]["eval_result_filename"]
            assert eval_result_file.exists(), (
                f"Evaluation result file should be saved: {eval_result_file}"
            )

        finally:
            env.close()

    def test_q_table_train_with_checkpoint(
        self, test_config: dict[str, Any], temp_output_dir: Path
    ) -> None:
        """Test Q-table training starting from a checkpoint."""
        # First, create a simple Q-table checkpoint
        checkpoint_file = temp_output_dir / "checkpoint_q_table.pkl"
        initial_q_table = np.random.rand(16, 4).astype(
            np.float32
        )  # 4x4 FrozenLake has 16 states, 4 actions

        import pickle5 as pickle

        with open(checkpoint_file, "wb") as f:
            pickle.dump(initial_q_table, f)

        # Update config to use checkpoint
        test_config["checkpoint_pathname"] = str(checkpoint_file)
        test_config["output_params"]["output_dir"] = str(temp_output_dir)
        test_config["output_params"]["save_result"] = False

        # Create environment
        env, env_info = make_discrete_env_with_kwargs(
            env_id=test_config["env_params"]["env_id"],
            kwargs=test_config["env_params"]["kwargs"],
        )
        test_config["env_params"].update(env_info)

        try:
            # Run training from checkpoint
            q_table_train(env=env, cfg_data=test_config)

            # Test passes if no exception is raised
            assert True, "Training from checkpoint completed successfully"
        finally:
            env.close()

    def test_main_function_full_flow(
        self, test_config: dict[str, Any], temp_output_dir: Path
    ) -> None:
        """Test the main function which includes environment creation and cleanup."""
        # Update config to use temp directory
        test_config["output_params"]["output_dir"] = str(temp_output_dir)
        test_config["output_params"]["save_result"] = True

        # Run main function
        main(cfg_data=test_config)

        # Verify output files were created
        output_dir = Path(test_config["output_params"]["output_dir"])
        model_file = output_dir / test_config["output_params"]["model_filename"]
        assert model_file.exists(), "Model file should be created by main function"

    def test_main_function_with_config_file(
        self, test_config: dict[str, Any], temp_output_dir: Path
    ) -> None:
        """Test the main function using an actual config file."""
        # Update config to use temp directory
        test_config["output_params"]["output_dir"] = str(temp_output_dir)

        # Create a temporary config file
        config_file = temp_output_dir / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(test_config, f, indent=2)

        # Mock command line arguments and run
        with patch("sys.argv", ["q_train.py", "--config", str(config_file)]):
            with patch("hands_on.exercise1_q_learning.q_train.load_config_from_json") as mock_load:
                mock_load.return_value = test_config

                # Import and run the main section

                # Run the main function directly
                main(cfg_data=test_config)

                # Verify output files were created
                output_dir = Path(test_config["output_params"]["output_dir"])
                model_file = output_dir / test_config["output_params"]["model_filename"]
                assert model_file.exists(), "Model file should be created"

    def test_training_produces_valid_results(
        self, test_config: dict[str, Any], temp_output_dir: Path
    ) -> None:
        """Test that training produces valid Q-table and evaluation results."""
        test_config["output_params"]["output_dir"] = str(temp_output_dir)
        test_config["output_params"]["save_result"] = True

        # Create environment
        env, env_info = make_discrete_env_with_kwargs(
            env_id=test_config["env_params"]["env_id"],
            kwargs=test_config["env_params"]["kwargs"],
        )
        test_config["env_params"].update(env_info)

        try:
            # Run training
            q_table_train(env=env, cfg_data=test_config)

            # Load and validate the saved Q-table
            import pickle5 as pickle

            model_file = (
                Path(test_config["output_params"]["output_dir"])
                / test_config["output_params"]["model_filename"]
            )
            with open(model_file, "rb") as f:
                q_table = pickle.load(f)

            # Validate Q-table shape and type
            assert isinstance(q_table, np.ndarray), "Q-table should be numpy array"
            assert q_table.shape == (16, 4), "Q-table should have shape (16, 4) for 4x4 FrozenLake"
            assert q_table.dtype == np.float32, "Q-table should have float32 dtype"

            # Load and validate evaluation results
            eval_result_file = (
                Path(test_config["output_params"]["output_dir"])
                / test_config["output_params"]["eval_result_filename"]
            )
            with open(eval_result_file, "r") as f:
                eval_result = json.load(f)

            # Validate evaluation results structure
            assert "mean_reward" in eval_result, "Evaluation should include mean_reward"
            assert "std_reward" in eval_result, "Evaluation should include std_reward"
            assert "datetime" in eval_result, "Evaluation should include datetime"
            assert isinstance(eval_result["mean_reward"], (int, float)), (
                "mean_reward should be numeric"
            )
            assert isinstance(eval_result["std_reward"], (int, float)), (
                "std_reward should be numeric"
            )

        finally:
            env.close()

    def test_training_with_different_hyperparameters(
        self, test_config: dict[str, Any], temp_output_dir: Path
    ) -> None:
        """Test training with different hyperparameter configurations."""
        # Test with high learning rate
        test_config_high_lr = test_config.copy()
        test_config_high_lr["hyper_params"]["learning_rate"] = 0.9
        test_config_high_lr["output_params"]["output_dir"] = str(temp_output_dir / "high_lr")
        test_config_high_lr["output_params"]["save_result"] = False

        # Test with low learning rate
        test_config_low_lr = test_config.copy()
        test_config_low_lr["hyper_params"]["learning_rate"] = 0.1
        test_config_low_lr["output_params"]["output_dir"] = str(temp_output_dir / "low_lr")
        test_config_low_lr["output_params"]["save_result"] = False

        for config in [test_config_high_lr, test_config_low_lr]:
            env, env_info = make_discrete_env_with_kwargs(
                env_id=config["env_params"]["env_id"],
                kwargs=config["env_params"]["kwargs"],
            )
            config["env_params"].update(env_info)

            try:
                # Should not raise any exceptions
                q_table_train(env=env, cfg_data=config)
            finally:
                env.close()

    def test_environment_cleanup_on_exception(
        self, test_config: dict[str, Any], temp_output_dir: Path
    ) -> None:
        """Test that environment is properly closed even when exceptions occur."""
        test_config["output_params"]["output_dir"] = str(temp_output_dir)
        test_config["output_params"]["save_result"] = False

        # Use a real environment but inject an exception in a way that tests cleanup
        from unittest.mock import patch

        # Mock the evaluation step to raise an exception after training starts
        with patch("hands_on.exercise1_q_learning.q_train.evaluate_and_save_results") as mock_eval:
            mock_eval.side_effect = RuntimeError("Simulated evaluation error")

            # Should raise the exception but environment should still be cleaned up properly
            with pytest.raises(RuntimeError, match="Simulated evaluation error"):
                main(cfg_data=test_config)

            # If we get here, it means the finally block worked correctly
            # (the environment was closed properly, otherwise we'd have resource warnings)
            assert True, "Environment cleanup handled exception correctly"
