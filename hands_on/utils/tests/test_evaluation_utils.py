"""Tests for evaluation utilities."""

import json
import tempfile
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pytest

from hands_on.base import ActType, AgentBase
from hands_on.utils.env_utils import make_discrete_env_with_kwargs
from hands_on.utils.evaluation_utils import evaluate_and_save_results


class DummyAgent(AgentBase):
    """Dummy agent for testing."""

    def action(self, state: Any) -> ActType:
        """Return a random action."""
        return ActType(np.random.randint(0, 4))

    def only_save_model(self, pathname: str) -> None:
        """Save dummy model data."""
        with open(pathname, "w") as f:
            f.write("dummy_model_data")

    @classmethod
    def load_from_checkpoint(cls, pathname: str, device: Any) -> "DummyAgent":
        """Load dummy agent."""
        return cls()


@pytest.fixture
def test_config() -> Generator[dict[str, Any], None, None]:
    """Create a test configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield {
            "eval_params": {
                "eval_episodes": 3,
                "eval_seed": [1, 2, 3],
                "max_steps": 10,
                "record_video": False,
            },
            "output_params": {
                "output_dir": temp_dir,
                "save_result": True,
                "model_filename": "test_model.txt",
                "params_filename": "test_params.json",
                "train_result_filename": "test_train_result.json",
                "eval_result_filename": "test_eval_result.json",
            },
        }


class TestEvaluateAndSaveResults:
    """Test the centralized evaluation and saving function."""

    def test_evaluate_and_save_results_basic(self, test_config: dict[str, Any]) -> None:
        """Test basic functionality of evaluate_and_save_results."""
        # Create environment
        env, _ = make_discrete_env_with_kwargs(
            env_id="FrozenLake-v1", kwargs={"is_slippery": False}
        )

        # Create dummy agent
        agent = DummyAgent()

        # Create dummy training results
        train_result = {
            "episode_rewards": [1.0, 0.0, 1.0],
            "episode_lengths": [10, 15, 8],
        }

        try:
            # Run evaluation and saving
            evaluate_and_save_results(
                env=env,
                agent=agent,
                cfg_data=test_config,
                train_result=train_result,
            )

            # Verify files were created
            output_dir = Path(test_config["output_params"]["output_dir"])
            assert output_dir.exists()

            model_file = output_dir / test_config["output_params"]["model_filename"]
            assert model_file.exists()

            eval_result_file = output_dir / test_config["output_params"]["eval_result_filename"]
            assert eval_result_file.exists()

            train_result_file = output_dir / test_config["output_params"]["train_result_filename"]
            assert train_result_file.exists()

            params_file = output_dir / test_config["output_params"]["params_filename"]
            assert params_file.exists()

            # Verify eval result content
            with open(eval_result_file, "r") as f:
                eval_result = json.load(f)

            assert "mean_reward" in eval_result
            assert "std_reward" in eval_result
            assert "datetime" in eval_result
            assert isinstance(eval_result["mean_reward"], (int, float))
            assert isinstance(eval_result["std_reward"], (int, float))

        finally:
            env.close()

    def test_evaluate_and_save_results_with_additional_data(
        self, test_config: dict[str, Any]
    ) -> None:
        """Test evaluate_and_save_results with additional evaluation data."""
        # Create environment
        env, _ = make_discrete_env_with_kwargs(
            env_id="FrozenLake-v1", kwargs={"is_slippery": False}
        )

        # Create dummy agent
        agent = DummyAgent()

        # Create dummy training results
        train_result = {
            "episode_rewards": [1.0, 0.0, 1.0],
            "episode_lengths": [10, 15, 8],
        }

        # Additional evaluation data
        additional_eval_data = {
            "custom_metric": 42.0,
            "training_duration": 120.5,
        }

        try:
            # Run evaluation and saving
            evaluate_and_save_results(
                env=env,
                agent=agent,
                cfg_data=test_config,
                train_result=train_result,
                additional_eval_data=additional_eval_data,
            )

            # Verify eval result contains additional data
            output_dir = Path(test_config["output_params"]["output_dir"])
            eval_result_file = output_dir / test_config["output_params"]["eval_result_filename"]

            with open(eval_result_file, "r") as f:
                eval_result = json.load(f)

            assert "mean_reward" in eval_result
            assert "std_reward" in eval_result
            assert "datetime" in eval_result
            assert "custom_metric" in eval_result
            assert "training_duration" in eval_result
            assert eval_result["custom_metric"] == 42.0
            assert eval_result["training_duration"] == 120.5

        finally:
            env.close()

    def test_evaluate_and_save_results_no_save(self, test_config: dict[str, Any]) -> None:
        """Test evaluate_and_save_results when save_result is False."""
        # Modify config to disable saving
        test_config["output_params"]["save_result"] = False

        # Create environment
        env, _ = make_discrete_env_with_kwargs(
            env_id="FrozenLake-v1", kwargs={"is_slippery": False}
        )

        # Create dummy agent
        agent = DummyAgent()

        # Create dummy training results
        train_result = {
            "episode_rewards": [1.0, 0.0, 1.0],
            "episode_lengths": [10, 15, 8],
        }

        try:
            # Run evaluation (should not save)
            evaluate_and_save_results(
                env=env,
                agent=agent,
                cfg_data=test_config,
                train_result=train_result,
            )

            # Verify no files were created (except possibly the temp directory)
            output_dir = Path(test_config["output_params"]["output_dir"])

            # Check that specific files don't exist
            model_file = output_dir / test_config["output_params"]["model_filename"]
            assert not model_file.exists()

            eval_result_file = output_dir / test_config["output_params"]["eval_result_filename"]
            assert not eval_result_file.exists()

        finally:
            env.close()
