"""Tests for push_to_hub_generic function.

This test suite provides comprehensive coverage of the hub functionality
including metadata generation, model card creation, and hub pushing.

Test Coverage:
- Basic hub push functionality with successful upload
- Metadata generation with algorithm-specific tags
- Model card creation with proper formatting
- File path handling and hub parameters
- Error handling for hub operations
- Configuration validation
- Mock verification for external dependencies

The tests use mocking to avoid actual hub interactions while maintaining
realistic test scenarios.

Note: Environment cleanup is now handled by the CLI, not by push_to_hub_generic itself.
"""

import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import Mock, patch

import pytest
import torch

from practice.base.config import ArtifactConfig, BaseConfig, EnvConfig
from practice.base.trainer import TrainerBase
from practice.utils.hub_utils import push_model_to_hub, push_to_hub_generic


@pytest.fixture
def mock_env() -> Mock:
    """Create a mock environment for testing."""
    env = Mock()
    env.observation_space = Mock()
    env.observation_space.shape = (4,)
    env.action_space = Mock()
    env.action_space.n = 2
    env.close = Mock()
    return env


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary output directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_agent() -> Mock:
    """Create a mock agent for testing."""
    agent = Mock()
    agent.load_from_checkpoint = Mock()
    return agent


@pytest.fixture
def test_config(mock_agent: Mock, temp_output_dir: Path) -> BaseConfig:
    """Create a test configuration for hub functionality."""
    from dataclasses import dataclass

    @dataclass(frozen=True, kw_only=True)
    class TestTrainer(TrainerBase):
        def train(self) -> None:
            pass

    artifact_config = ArtifactConfig(
        trainer_type=TestTrainer,
        agent_type=mock_agent,
        output_dir=str(temp_output_dir),
        save_result=True,
        model_filename="test_model.pth",
        params_filename="test_params.json",
        eval_result_filename="test_eval_result.json",
        repo_id="test-repo",
        replay_video_filename="test_replay.mp4",
        fps=30,
        seed=42,
        algorithm_name="TestAlgorithm",
        extra_tags=("test", "hub", "push"),
        usage_instructions="Test usage instructions for hub",
    )

    env_config = EnvConfig(
        env_id="CartPole-v1",
        record_eval_video=True,
        max_steps=100,
    )

    @dataclass(frozen=True, kw_only=True)
    class TestConfig(BaseConfig):
        pass

    return TestConfig(
        env_config=env_config,
        device=torch.device("cpu"),
        learning_rate=1e-3,
        gamma=0.99,
        eval_episodes=5,
        eval_random_seed=42,
        eval_video_num=1,
        artifact_config=artifact_config,
    )


@pytest.fixture
def mock_metadata() -> dict[str, Any]:
    """Create mock metadata for testing."""
    return {
        "algorithm": "testalgorithm",
        "env_id": "CartPole-v1",
        "model_name": "TestAlgorithm",
        "tags": ["test", "hub", "push", "testalgorithm"],
    }


class TestPushToHubGeneric:
    """Test push_to_hub_generic function."""

    @patch("practice.utils.hub_utils.push_model_to_hub")
    @patch("practice.utils.hub_utils.get_env_name_and_metadata")
    def test_push_to_hub_success(
        self,
        mock_get_metadata: Mock,
        mock_push_model: Mock,
        test_config: BaseConfig,
        mock_env: Mock,
        mock_metadata: dict[str, Any],
    ) -> None:
        """Test successful hub push with proper function calls."""
        # Setup mocks
        mock_get_metadata.return_value = mock_metadata
        mock_push_model.return_value = None

        username = "testuser"

        # Call the function
        push_to_hub_generic(test_config, mock_env, username)

        # Verify metadata generation was called
        mock_get_metadata.assert_called_once_with(
            env_id=test_config.env_config.env_id,
            env=mock_env,
            algorithm_name=test_config.artifact_config.algorithm_name.lower(),
            extra_tags=test_config.artifact_config.extra_tags,
        )

        # Verify push_model_to_hub was called
        mock_push_model.assert_called_once()

        # NOTE: Environment cleanup is now handled by the CLI, not by push_to_hub_generic
        # So we don't check for env.close() being called here

        # Verify push_model_to_hub call arguments
        call_args = mock_push_model.call_args
        assert call_args[1]["repo_id"] == f"{username}/{test_config.artifact_config.repo_id}"
        assert call_args[1]["artifact_config"] == test_config.artifact_config
        assert call_args[1]["metadata"] == mock_metadata

        # Verify model card content
        model_card = call_args[1]["model_card"]
        assert f"**{test_config.artifact_config.algorithm_name}**" in model_card
        assert f"**{test_config.env_config.env_id}**" in model_card
        assert test_config.artifact_config.usage_instructions in model_card
        assert test_config.artifact_config.model_filename in model_card

    @patch("practice.utils.hub_utils.push_model_to_hub")
    @patch("practice.utils.hub_utils.get_env_name_and_metadata")
    def test_push_to_hub_with_different_username(
        self,
        mock_get_metadata: Mock,
        mock_push_model: Mock,
        test_config: BaseConfig,
        mock_env: Mock,
        mock_metadata: dict[str, Any],
    ) -> None:
        """Test hub push with different username."""
        # Setup mocks
        mock_get_metadata.return_value = mock_metadata
        mock_push_model.return_value = None

        username = "different_user"

        # Call the function
        push_to_hub_generic(test_config, mock_env, username)

        # Verify repo_id construction
        call_args = mock_push_model.call_args
        assert call_args[1]["repo_id"] == f"{username}/{test_config.artifact_config.repo_id}"

    @patch("practice.utils.hub_utils.push_model_to_hub")
    @patch("practice.utils.hub_utils.get_env_name_and_metadata")
    def test_push_to_hub_with_no_extra_tags(
        self,
        mock_get_metadata: Mock,
        mock_push_model: Mock,
        test_config: BaseConfig,
        mock_env: Mock,
        mock_metadata: dict[str, Any],
    ) -> None:
        """Test hub push when config has no extra tags."""
        # Modify config to have no extra tags
        from dataclasses import replace

        config_no_tags = replace(
            test_config, artifact_config=replace(test_config.artifact_config, extra_tags=())
        )

        # Setup mocks
        mock_get_metadata.return_value = mock_metadata
        mock_push_model.return_value = None

        username = "testuser"

        # Call the function
        push_to_hub_generic(config_no_tags, mock_env, username)

        # Verify metadata generation was called with empty extra_tags
        mock_get_metadata.assert_called_once_with(
            env_id=config_no_tags.env_config.env_id,
            env=mock_env,
            algorithm_name=config_no_tags.artifact_config.algorithm_name.lower(),
            extra_tags=(),
        )

    @patch("practice.utils.hub_utils.push_model_to_hub")
    @patch("practice.utils.hub_utils.get_env_name_and_metadata")
    def test_push_to_hub_metadata_error(
        self,
        mock_get_metadata: Mock,
        mock_push_model: Mock,
        test_config: BaseConfig,
        mock_env: Mock,
    ) -> None:
        """Test error handling when metadata generation fails."""
        # Setup mock to raise error
        mock_get_metadata.side_effect = RuntimeError("Metadata generation failed")

        username = "testuser"

        # Verify the exception is propagated
        with pytest.raises(RuntimeError, match="Metadata generation failed"):
            push_to_hub_generic(test_config, mock_env, username)

        # NOTE: Environment cleanup is now handled by the CLI, not by push_to_hub_generic
        # So we don't check for env.close() being called here

    @patch("practice.utils.hub_utils.push_model_to_hub")
    @patch("practice.utils.hub_utils.get_env_name_and_metadata")
    def test_push_to_hub_push_error(
        self,
        mock_get_metadata: Mock,
        mock_push_model: Mock,
        test_config: BaseConfig,
        mock_env: Mock,
        mock_metadata: dict[str, Any],
    ) -> None:
        """Test error handling when hub push fails."""
        # Setup mocks
        mock_get_metadata.return_value = mock_metadata
        mock_push_model.side_effect = RuntimeError("Hub push failed")

        username = "testuser"

        # Verify the exception is propagated
        with pytest.raises(RuntimeError, match="Hub push failed"):
            push_to_hub_generic(test_config, mock_env, username)

        # NOTE: Environment cleanup is now handled by the CLI, not by push_to_hub_generic
        # So we don't check for env.close() being called here

    @patch("practice.utils.hub_utils.push_model_to_hub")
    @patch("practice.utils.hub_utils.get_env_name_and_metadata")
    def test_model_card_generation(
        self,
        mock_get_metadata: Mock,
        mock_push_model: Mock,
        test_config: BaseConfig,
        mock_env: Mock,
        mock_metadata: dict[str, Any],
    ) -> None:
        """Test model card generation with proper formatting."""
        # Setup mocks
        mock_get_metadata.return_value = mock_metadata
        mock_push_model.return_value = None

        username = "testuser"

        # Call the function
        push_to_hub_generic(test_config, mock_env, username)

        # Verify model card content
        call_args = mock_push_model.call_args
        model_card = call_args[1]["model_card"]

        # Check for expected content
        assert f"**{test_config.artifact_config.algorithm_name}**" in model_card
        assert f"**{test_config.env_config.env_id}**" in model_card
        assert test_config.artifact_config.usage_instructions in model_card
        assert test_config.artifact_config.model_filename in model_card
        assert f"testuser/{test_config.artifact_config.repo_id}" in model_card
        assert f'gym.make("{test_config.env_config.env_id}")' in model_card


class TestPushModelToHub:
    """Test push_model_to_hub function."""

    @patch("practice.utils.hub_utils.push_to_hub")
    def test_push_model_to_hub_success(
        self, mock_push_to_hub: Mock, test_config: BaseConfig, mock_metadata: dict[str, Any]
    ) -> None:
        """Test successful model push to hub."""
        # Setup mock
        mock_push_to_hub.return_value = None

        repo_id = "testuser/test-repo"
        model_card = "Test model card"

        # Call the function
        push_model_to_hub(
            repo_id=repo_id,
            artifact_config=test_config.artifact_config,
            model_card=model_card,
            metadata=mock_metadata,
        )

        # Verify push_to_hub was called with correct parameters
        mock_push_to_hub.assert_called_once()
        call_args = mock_push_to_hub.call_args

        assert call_args[1]["repo_id"] == repo_id
        assert call_args[1]["model_card"] == model_card
        assert call_args[1]["metadata"] == mock_metadata
        assert call_args[1]["copy_file"] is False

        # Verify file paths
        expected_files = [
            str(
                Path(test_config.artifact_config.output_dir)
                / test_config.artifact_config.model_filename
            ),
            str(
                Path(test_config.artifact_config.output_dir)
                / test_config.artifact_config.params_filename
            ),
            str(
                Path(test_config.artifact_config.output_dir)
                / test_config.artifact_config.replay_video_filename
            ),
        ]
        assert call_args[1]["file_pathnames"] == expected_files

    @patch("practice.utils.hub_utils.push_to_hub")
    def test_push_model_to_hub_with_different_files(
        self, mock_push_to_hub: Mock, mock_metadata: dict[str, Any], temp_output_dir: Path
    ) -> None:
        """Test model push with different file configuration."""
        from dataclasses import dataclass

        @dataclass(frozen=True, kw_only=True)
        class TestTrainer(TrainerBase):
            def train(self) -> None:
                pass

        # Create config with different filenames
        artifact_config = ArtifactConfig(
            trainer_type=TestTrainer,
            agent_type=Mock,
            output_dir=str(temp_output_dir),
            save_result=True,
            model_filename="custom_model.pth",
            params_filename="custom_params.json",
            eval_result_filename="custom_eval.json",
            repo_id="custom-repo",
            replay_video_filename="custom_replay.mp4",
            fps=60,
            seed=123,
            algorithm_name="CustomAlgorithm",
            extra_tags=("custom", "test"),
            usage_instructions="Custom usage instructions",
        )

        # Setup mock
        mock_push_to_hub.return_value = None

        repo_id = "testuser/custom-repo"
        model_card = "Custom model card"

        # Call the function
        push_model_to_hub(
            repo_id=repo_id,
            artifact_config=artifact_config,
            model_card=model_card,
            metadata=mock_metadata,
        )

        # Verify custom file paths
        call_args = mock_push_to_hub.call_args
        expected_files = [
            str(temp_output_dir / "custom_model.pth"),
            str(temp_output_dir / "custom_params.json"),
            str(temp_output_dir / "custom_replay.mp4"),
        ]
        assert call_args[1]["file_pathnames"] == expected_files
        assert call_args[1]["eval_result_pathname"] == str(temp_output_dir / "custom_eval.json")

    @patch("practice.utils.hub_utils.push_to_hub")
    def test_push_model_to_hub_error(
        self, mock_push_to_hub: Mock, test_config: BaseConfig, mock_metadata: dict[str, Any]
    ) -> None:
        """Test error handling in push_model_to_hub."""
        # Setup mock to raise error
        mock_push_to_hub.side_effect = RuntimeError("Hub connection failed")

        repo_id = "testuser/test-repo"
        model_card = "Test model card"

        # Verify the exception is propagated
        with pytest.raises(RuntimeError, match="Hub connection failed"):
            push_model_to_hub(
                repo_id=repo_id,
                artifact_config=test_config.artifact_config,
                model_card=model_card,
                metadata=mock_metadata,
            )

    def test_file_path_construction(self, test_config: BaseConfig, temp_output_dir: Path) -> None:
        """Test that file paths are constructed correctly."""
        artifact_config = test_config.artifact_config

        # Verify path construction logic
        expected_model_path = temp_output_dir / artifact_config.model_filename
        expected_params_path = temp_output_dir / artifact_config.params_filename
        expected_video_path = temp_output_dir / artifact_config.replay_video_filename
        expected_eval_path = temp_output_dir / artifact_config.eval_result_filename

        # These should match what push_model_to_hub constructs
        output_dir = Path(artifact_config.output_dir)
        assert str(output_dir / artifact_config.model_filename) == str(expected_model_path)
        assert str(output_dir / artifact_config.params_filename) == str(expected_params_path)
        assert str(output_dir / artifact_config.replay_video_filename) == str(expected_video_path)
        assert str(output_dir / artifact_config.eval_result_filename) == str(expected_eval_path)


class TestHubUtilsIntegration:
    """Integration tests for hub utilities."""

    @patch("practice.utils.hub_utils.push_to_hub")
    @patch("practice.utils.hub_utils.get_env_name_and_metadata")
    def test_full_hub_workflow(
        self,
        mock_get_metadata: Mock,
        mock_push_to_hub: Mock,
        test_config: BaseConfig,
        mock_env: Mock,
        mock_metadata: dict[str, Any],
    ) -> None:
        """Test the full workflow from config to hub push."""
        # Setup mocks
        mock_get_metadata.return_value = mock_metadata
        mock_push_to_hub.return_value = None

        username = "testuser"

        # Call the function
        push_to_hub_generic(test_config, mock_env, username)

        # Verify the entire workflow
        mock_get_metadata.assert_called_once()
        mock_push_to_hub.assert_called_once()

        # NOTE: Environment cleanup is now handled by the CLI, not by push_to_hub_generic
        # So we don't check for env.close() being called here

        # Verify the flow of data
        metadata_call_args = mock_get_metadata.call_args
        push_call_args = mock_push_to_hub.call_args

        # Verify metadata generation parameters
        assert metadata_call_args[1]["env_id"] == test_config.env_config.env_id
        assert metadata_call_args[1]["env"] == mock_env
        assert (
            metadata_call_args[1]["algorithm_name"]
            == test_config.artifact_config.algorithm_name.lower()
        )
        assert metadata_call_args[1]["extra_tags"] == test_config.artifact_config.extra_tags

        # Verify push parameters
        assert push_call_args[1]["repo_id"] == f"{username}/{test_config.artifact_config.repo_id}"
        assert push_call_args[1]["metadata"] == mock_metadata
        assert push_call_args[1]["copy_file"] is False

    def test_config_validation(self, test_config: BaseConfig) -> None:
        """Test that configuration has all required fields for hub operations."""
        artifact_config = test_config.artifact_config
        env_config = test_config.env_config

        # Verify required fields are present
        assert artifact_config.output_dir is not None
        assert artifact_config.model_filename is not None
        assert artifact_config.params_filename is not None
        assert artifact_config.eval_result_filename is not None
        assert artifact_config.replay_video_filename is not None
        assert artifact_config.repo_id is not None
        assert artifact_config.algorithm_name is not None
        assert artifact_config.extra_tags is not None
        assert artifact_config.usage_instructions is not None

        # Verify env config
        assert env_config.env_id is not None

        # Verify config types
        assert isinstance(artifact_config.extra_tags, tuple)
        assert isinstance(artifact_config.usage_instructions, str)
        assert isinstance(artifact_config.algorithm_name, str)
        assert isinstance(artifact_config.repo_id, str)
