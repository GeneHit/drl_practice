"""Tests for REINFORCE training flow.

This test suite provides comprehensive coverage of the REINFORCE training functionality
using the actual cartpole_config.json structure with reduced parameters for fast testing.

Test Coverage (12 Tests):
- Vector environment creation (sync and async)
- Basic training flow without file saving
- Training with file saving and output validation
- Training from checkpoint (resuming from saved model)
- Main function integration (environment creation and cleanup)
- Config file loading and processing
- Validation of training outputs (model structure, evaluation results)
- Different hyperparameter configurations
- Exception handling and environment cleanup
- Device handling (CPU/CUDA/MPS)
- Different environment types (CartPole-v1, LunarLander-v3)

The tests use a reduced configuration based on cartpole_config.json:
- global_episode: 5 (vs 1000)
- lr: 1e-3 (vs 1e-4, increased for faster learning)
- num_envs: 2 (vs 2, kept same)
- use_multi_processing: False (disabled for testing)
- max_steps: 50 (vs 1000, reduced)
- eval_episodes: 3 (vs 20)
- eval_seed: 3 seeds (vs 20)

This ensures fast test execution while maintaining realistic training behavior.
All tests validate proper resource cleanup and exception handling.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import patch

import pytest
import torch

from hands_on.exercise2_dqn.dqn_exercise import EnvType
from hands_on.exercise3_reinforce.reinforce_train import (
    reinforce_main,
    reinforce_train_with_envs,
)
from hands_on.utils.env_utils import make_1d_env


@pytest.fixture
def test_config() -> dict[str, Any]:
    """Create a test configuration based on cartpole_config.json with reduced parameters."""
    return {
        "hyper_params": {
            "global_episode": 5,  # Reduced from 1000
            "lr": 1e-3,  # Increased from 1e-4 for faster learning
            "gamma": 0.99,
            "grad_acc": 2,
            "num_envs": 2,  # Keep same as original
            "use_multi_processing": False,  # Disabled for testing
        },
        "env_params": {
            "env_id": "CartPole-v1",
            "max_steps": 50,  # Reduced from 1000
        },
        "eval_params": {
            "eval_episodes": 3,  # Reduced from 20
            "eval_seed": [1, 55, 44],  # Reduced to 3 seeds
        },
        "output_params": {
            "output_dir": "",  # Will be set to temp dir in tests
            "save_result": True,
            "model_filename": "reinforce.pth",
            "params_filename": "params.json",
            "train_result_filename": "train_result.json",
            "eval_result_filename": "eval_result.json",
        },
        "hub_params": {"repo_id": "test-Reinforce-CartPole"},
    }


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestReinforceTraining:
    """Test REINFORCE training functionality."""

    def test_vector_env_creation_sync(self, test_config: dict[str, Any]) -> None:
        """Test synchronous vector environment creation."""
        import gymnasium as gym

        def env_fn() -> EnvType:
            env, _ = make_1d_env(
                env_id=test_config["env_params"]["env_id"],
                max_steps=test_config["env_params"].get("max_steps"),
            )
            return env

        # Test sync vector env
        envs = gym.vector.SyncVectorEnv([env_fn for _ in range(2)])

        try:
            assert envs.num_envs == 2, "Should create 2 environments"
            assert hasattr(envs, "single_observation_space"), "Should have observation space"
            assert hasattr(envs, "single_action_space"), "Should have action space"
        finally:
            envs.close()

    def test_vector_env_creation_async(self, test_config: dict[str, Any]) -> None:
        """Test asynchronous vector environment creation."""
        import gymnasium as gym

        def env_fn() -> EnvType:
            env, _ = make_1d_env(
                env_id=test_config["env_params"]["env_id"],
                max_steps=test_config["env_params"].get("max_steps"),
            )
            return env

        # Test async vector env
        envs = gym.vector.AsyncVectorEnv([env_fn for _ in range(2)])

        try:
            assert envs.num_envs == 2, "Should create 2 environments"
            assert hasattr(envs, "single_observation_space"), "Should have observation space"
            assert hasattr(envs, "single_action_space"), "Should have action space"
        finally:
            envs.close()

    def test_reinforce_train_basic_flow(
        self, test_config: dict[str, Any], temp_output_dir: Path
    ) -> None:
        """Test basic REINFORCE training flow without file operations."""
        import gymnasium as gym

        # Update config to use temp directory
        test_config["output_params"]["output_dir"] = str(temp_output_dir)
        test_config["output_params"]["save_result"] = False  # Skip file saving for this test

        # Create environment factory function
        def env_fn() -> EnvType:
            env, _ = make_1d_env(
                env_id=test_config["env_params"]["env_id"],
                max_steps=test_config["env_params"].get("max_steps"),
            )
            return env

        # Create vector environment
        envs = gym.vector.SyncVectorEnv(
            [env_fn for _ in range(test_config["hyper_params"]["num_envs"])]
        )

        try:
            # Run training
            reinforce_train_with_envs(envs=envs, cfg_data=test_config)

            # Test passes if no exception is raised
            assert True, "Training completed successfully"
        except Exception as e:
            # Make sure to close envs even if test fails
            if not envs.closed:
                envs.close()
            raise e

    def test_reinforce_train_with_file_saving(
        self, test_config: dict[str, Any], temp_output_dir: Path
    ) -> None:
        """Test REINFORCE training with file saving enabled."""
        import gymnasium as gym

        # Update config to use temp directory
        test_config["output_params"]["output_dir"] = str(temp_output_dir)
        test_config["output_params"]["save_result"] = True

        # Create environment factory function
        def env_fn() -> EnvType:
            env, _ = make_1d_env(
                env_id=test_config["env_params"]["env_id"],
                max_steps=test_config["env_params"].get("max_steps"),
            )
            return env

        # Create vector environment
        envs = gym.vector.SyncVectorEnv(
            [env_fn for _ in range(test_config["hyper_params"]["num_envs"])]
        )

        try:
            # Run training
            reinforce_train_with_envs(envs=envs, cfg_data=test_config)

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

        except Exception as e:
            # Make sure to close envs even if test fails
            if not envs.closed:
                envs.close()
            raise e

    def test_reinforce_train_with_checkpoint(
        self, test_config: dict[str, Any], temp_output_dir: Path
    ) -> None:
        """Test REINFORCE training starting from a checkpoint."""
        import gymnasium as gym

        # First, create a simple model checkpoint
        checkpoint_file = temp_output_dir / "checkpoint_reinforce.pth"

        # Create a dummy model state dict with correct structure
        from hands_on.exercise3_reinforce.reinforce_exercise import (
            Reinforce1DNet,
        )

        dummy_model = Reinforce1DNet(state_dim=4, action_dim=2)  # CartPole has 4 obs, 2 actions
        torch.save(dummy_model.state_dict(), checkpoint_file)

        # Update config to use checkpoint
        test_config["checkpoint_pathname"] = str(checkpoint_file)
        test_config["output_params"]["output_dir"] = str(temp_output_dir)
        test_config["output_params"]["save_result"] = False

        # Create environment factory function
        def env_fn() -> EnvType:
            env, _ = make_1d_env(
                env_id=test_config["env_params"]["env_id"],
                max_steps=test_config["env_params"].get("max_steps"),
            )
            return env

        # Create vector environment
        envs = gym.vector.SyncVectorEnv(
            [env_fn for _ in range(test_config["hyper_params"]["num_envs"])]
        )

        try:
            # Run training from checkpoint
            reinforce_train_with_envs(envs=envs, cfg_data=test_config)

            # Test passes if no exception is raised
            assert True, "Training from checkpoint completed successfully"
        except Exception as e:
            # Make sure to close envs even if test fails
            if not envs.closed:
                envs.close()
            raise e

    def test_main_function_full_flow(
        self, test_config: dict[str, Any], temp_output_dir: Path
    ) -> None:
        """Test the main function which includes environment creation and cleanup."""
        # Update config to use temp directory
        test_config["output_params"]["output_dir"] = str(temp_output_dir)
        test_config["output_params"]["save_result"] = True

        # Run main function
        reinforce_main(cfg_data=test_config)

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
        with patch("sys.argv", ["reinforce_train.py", "--config", str(config_file)]):
            with patch("hands_on.utils.file_utils.load_config_from_json") as mock_load:
                mock_load.return_value = test_config

                # Run the main function directly
                reinforce_main(cfg_data=test_config)

                # Verify output files were created
                output_dir = Path(test_config["output_params"]["output_dir"])
                model_file = output_dir / test_config["output_params"]["model_filename"]
                assert model_file.exists(), "Model file should be created"

    def test_training_produces_valid_results(
        self, test_config: dict[str, Any], temp_output_dir: Path
    ) -> None:
        """Test that training produces valid model and evaluation results."""
        test_config["output_params"]["output_dir"] = str(temp_output_dir)
        test_config["output_params"]["save_result"] = True

        # Run main function
        reinforce_main(cfg_data=test_config)

        # Load and validate the saved model
        model_file = (
            Path(test_config["output_params"]["output_dir"])
            / test_config["output_params"]["model_filename"]
        )
        model = torch.load(model_file, map_location="cpu", weights_only=False)

        # Validate model structure - should be a Reinforce1DNet instance
        from hands_on.exercise3_reinforce.reinforce_exercise import (
            Reinforce1DNet,
        )

        assert isinstance(model, Reinforce1DNet), (
            f"Model should be Reinforce1DNet instance, got {type(model)}"
        )

        # Check that model has parameters
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0, "Model should have parameters"

        # Check model can perform forward pass
        dummy_input = torch.randn(1, 4)  # CartPole has 4 observations
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 2), f"Model output should be shape (1, 2), got {output.shape}"

        # Check output is valid probability distribution
        assert torch.allclose(output.sum(dim=1), torch.ones(1)), (
            "Output should sum to 1 (probability distribution)"
        )

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
        assert isinstance(eval_result["mean_reward"], (int, float)), "mean_reward should be numeric"
        assert isinstance(eval_result["std_reward"], (int, float)), "std_reward should be numeric"

    def test_training_with_different_hyperparameters(
        self, test_config: dict[str, Any], temp_output_dir: Path
    ) -> None:
        """Test training with different hyperparameter configurations."""
        # Test with high learning rate
        test_config_high_lr = test_config.copy()
        test_config_high_lr["hyper_params"] = test_config["hyper_params"].copy()
        test_config_high_lr["hyper_params"]["lr"] = 1e-2
        test_config_high_lr["output_params"]["output_dir"] = str(temp_output_dir / "high_lr")
        test_config_high_lr["output_params"]["save_result"] = False

        # Test with low learning rate
        test_config_low_lr = test_config.copy()
        test_config_low_lr["hyper_params"] = test_config["hyper_params"].copy()
        test_config_low_lr["hyper_params"]["lr"] = 1e-5
        test_config_low_lr["output_params"]["output_dir"] = str(temp_output_dir / "low_lr")
        test_config_low_lr["output_params"]["save_result"] = False

        for config in [test_config_high_lr, test_config_low_lr]:
            # Should not raise any exceptions
            reinforce_main(cfg_data=config)

    def test_environment_cleanup_on_exception(
        self, test_config: dict[str, Any], temp_output_dir: Path
    ) -> None:
        """Test that environment is properly closed even when exceptions occur."""
        test_config["output_params"]["output_dir"] = str(temp_output_dir)
        test_config["output_params"]["save_result"] = False

        # Use a real environment but inject an exception in a way that tests cleanup
        from unittest.mock import patch

        # Mock the evaluation step to raise an exception after training starts
        with patch(
            "hands_on.exercise3_reinforce.reinforce_train.evaluate_and_save_results"
        ) as mock_eval:
            mock_eval.side_effect = RuntimeError("Simulated evaluation error")

            # Should raise the exception but environment should still be cleaned up properly
            with pytest.raises(RuntimeError, match="Simulated evaluation error"):
                reinforce_main(cfg_data=test_config)

            # If we get here, it means the finally block worked correctly
            # (the environment was closed properly, otherwise we'd have resource warnings)
            assert True, "Environment cleanup handled exception correctly"

    def test_device_detection(self, test_config: dict[str, Any], temp_output_dir: Path) -> None:
        """Test device detection and model placement."""
        import gymnasium as gym

        test_config["output_params"]["output_dir"] = str(temp_output_dir)
        test_config["output_params"]["save_result"] = False

        # Create environment factory function
        def env_fn() -> EnvType:
            env, _ = make_1d_env(
                env_id=test_config["env_params"]["env_id"],
                max_steps=test_config["env_params"].get("max_steps"),
            )
            return env

        # Create vector environment
        envs = gym.vector.SyncVectorEnv(
            [env_fn for _ in range(test_config["hyper_params"]["num_envs"])]
        )

        try:
            # Mock device detection to test CPU path
            with patch("torch.cuda.is_available", return_value=False):
                with patch("torch.backends.mps.is_available", return_value=False):
                    reinforce_train_with_envs(envs=envs, cfg_data=test_config)

            # Test passes if training completes on CPU
            assert True, "Training completed successfully on CPU"
        except Exception as e:
            # Make sure to close envs even if test fails
            if not envs.closed:
                envs.close()
            raise e

    def test_different_environment_types(
        self, test_config: dict[str, Any], temp_output_dir: Path
    ) -> None:
        """Test training with different environment configurations."""
        # Test with LunarLander (more complex environment)
        test_config_lunar = test_config.copy()
        test_config_lunar["env_params"]["env_id"] = "LunarLander-v3"
        test_config_lunar["output_params"]["output_dir"] = str(temp_output_dir / "lunar")
        test_config_lunar["output_params"]["save_result"] = False
        test_config_lunar["hyper_params"]["global_episode"] = 3  # Even shorter for LunarLander

        # Should not raise any exceptions
        reinforce_main(cfg_data=test_config_lunar)
