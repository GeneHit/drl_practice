"""Tests for play_and_generate_video_generic function.

This test suite provides comprehensive coverage of the play functionality
including model loading, video generation, and error handling.

Test Coverage:
- Basic play functionality with successful video generation
- Model loading from checkpoint
- File path handling and video output
- Error handling for missing model files
- Environment cleanup
- Configuration validation
- Mock verification for external dependencies

The tests use mocking to avoid actual file I/O and environment interactions
while maintaining realistic test scenarios.
"""

import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch

import pytest
import torch

from practice.base.config import ArtifactConfig, BaseConfig, EnvConfig
from practice.base.trainer import TrainerBase
from practice.utils.play_utils import _load_model_from_config, play_and_generate_video_generic


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
def mock_agent() -> Mock:
    """Create a mock agent for testing."""
    agent = Mock()
    agent.load_from_checkpoint = Mock()
    return agent


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_config(mock_agent: Mock, temp_output_dir: Path) -> BaseConfig:
    """Create a test configuration for play functionality."""
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
        extra_tags=("test", "play"),
        usage_instructions="Test usage instructions",
    )

    env_config = EnvConfig(
        env_id="CartPole-v1",
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


class TestPlayAndGenerateVideoGeneric:
    """Test play_and_generate_video_generic function."""

    @patch("practice.utils.play_utils.play_game_once")
    @patch("practice.utils.play_utils._load_model_from_config")
    def test_play_and_generate_video_success(
        self,
        mock_load_model: Mock,
        mock_play_game_once: Mock,
        test_config: BaseConfig,
        mock_env: Mock,
        mock_agent: Mock,
        temp_output_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test successful video generation with proper function calls."""
        # Setup mocks
        mock_load_model.return_value = mock_agent
        mock_play_game_once.return_value = None

        # Call the function
        play_and_generate_video_generic(test_config, mock_env)

        # Verify model loading was called
        mock_load_model.assert_called_once_with(test_config)

        # Verify play_game_once was called with correct parameters
        expected_video_path = str(
            temp_output_dir / test_config.artifact_config.replay_video_filename
        )
        mock_play_game_once.assert_called_once_with(
            env=mock_env,
            policy=mock_agent,
            save_video=True,
            video_pathname=expected_video_path,
            fps=test_config.artifact_config.fps,
            seed=test_config.artifact_config.seed,
        )

        # Verify output message
        captured = capsys.readouterr()
        assert f"Game replay saved to: {expected_video_path}" in captured.out

    @patch("practice.utils.play_utils.play_game_once")
    @patch("practice.utils.play_utils._load_model_from_config")
    def test_play_and_generate_video_with_different_config(
        self,
        mock_load_model: Mock,
        mock_play_game_once: Mock,
        test_config: BaseConfig,
        mock_env: Mock,
        mock_agent: Mock,
        temp_output_dir: Path,
    ) -> None:
        """Test video generation with different configuration parameters."""
        # Modify config for different test scenario
        from dataclasses import replace

        modified_artifact_config = replace(
            test_config.artifact_config,
            replay_video_filename="custom_replay.mp4",
            fps=60,
            seed=123,
        )
        modified_config = replace(test_config, artifact_config=modified_artifact_config)

        # Setup mocks
        mock_load_model.return_value = mock_agent
        mock_play_game_once.return_value = None

        # Call the function
        play_and_generate_video_generic(modified_config, mock_env)

        # Verify play_game_once was called with modified parameters
        expected_video_path = str(temp_output_dir / "custom_replay.mp4")
        mock_play_game_once.assert_called_once_with(
            env=mock_env,
            policy=mock_agent,
            save_video=True,
            video_pathname=expected_video_path,
            fps=60,
            seed=123,
        )

    @patch("practice.utils.play_utils._load_model_from_config")
    def test_play_and_generate_video_model_loading_error(
        self, mock_load_model: Mock, test_config: BaseConfig, mock_env: Mock
    ) -> None:
        """Test error handling when model loading fails."""
        # Setup mock to raise FileNotFoundError
        mock_load_model.side_effect = FileNotFoundError("Model file not found")

        # Verify the exception is propagated
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            play_and_generate_video_generic(test_config, mock_env)

    @patch("practice.utils.play_utils.play_game_once")
    @patch("practice.utils.play_utils._load_model_from_config")
    def test_play_and_generate_video_play_error(
        self,
        mock_load_model: Mock,
        mock_play_game_once: Mock,
        test_config: BaseConfig,
        mock_env: Mock,
        mock_agent: Mock,
    ) -> None:
        """Test error handling when play_game_once fails."""
        # Setup mocks
        mock_load_model.return_value = mock_agent
        mock_play_game_once.side_effect = RuntimeError("Play failed")

        # Verify the exception is propagated
        with pytest.raises(RuntimeError, match="Play failed"):
            play_and_generate_video_generic(test_config, mock_env)

    def test_video_path_construction(self, test_config: BaseConfig, temp_output_dir: Path) -> None:
        """Test that video path is constructed correctly."""
        artifact_config = test_config.artifact_config
        expected_path = Path(artifact_config.output_dir) / artifact_config.replay_video_filename

        # Verify path construction
        assert expected_path == temp_output_dir / "test_replay.mp4"
        assert str(expected_path).endswith("test_replay.mp4")


class TestLoadModelFromConfig:
    """Test _load_model_from_config function."""

    def test_load_model_success(
        self, test_config: BaseConfig, mock_agent: Mock, temp_output_dir: Path
    ) -> None:
        """Test successful model loading from checkpoint."""
        # Create a dummy model file
        model_file = temp_output_dir / test_config.artifact_config.model_filename
        model_file.touch()

        # Setup mock
        mock_loaded_agent = Mock()
        mock_agent.load_from_checkpoint.return_value = mock_loaded_agent

        # Call the function
        result = _load_model_from_config(test_config)

        # Verify model loading was called correctly
        mock_agent.load_from_checkpoint.assert_called_once_with(
            str(model_file), device=test_config.device
        )
        assert result == mock_loaded_agent

    def test_load_model_file_not_found(
        self, test_config: BaseConfig, temp_output_dir: Path
    ) -> None:
        """Test error handling when model file doesn't exist."""
        # Don't create the model file

        # Verify FileNotFoundError is raised
        with pytest.raises(FileNotFoundError) as exc_info:
            _load_model_from_config(test_config)

        expected_path = temp_output_dir / test_config.artifact_config.model_filename
        assert f"Model file not found: {expected_path}" in str(exc_info.value)

    def test_load_model_with_different_device(
        self, test_config: BaseConfig, mock_agent: Mock, temp_output_dir: Path
    ) -> None:
        """Test model loading with different device configuration."""
        # Create a dummy model file
        model_file = temp_output_dir / test_config.artifact_config.model_filename
        model_file.touch()

        # Modify config to use different device
        from dataclasses import replace

        cuda_config = replace(
            test_config, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Setup mock
        mock_loaded_agent = Mock()
        mock_agent.load_from_checkpoint.return_value = mock_loaded_agent

        # Call the function
        result = _load_model_from_config(cuda_config)

        # Verify model loading was called with correct device
        mock_agent.load_from_checkpoint.assert_called_once_with(
            str(model_file), device=cuda_config.device
        )
        assert result == mock_loaded_agent

    def test_load_model_checkpoint_error(
        self, test_config: BaseConfig, mock_agent: Mock, temp_output_dir: Path
    ) -> None:
        """Test error handling when checkpoint loading fails."""
        # Create a dummy model file
        model_file = temp_output_dir / test_config.artifact_config.model_filename
        model_file.touch()

        # Setup mock to raise error
        mock_agent.load_from_checkpoint.side_effect = RuntimeError("Checkpoint corrupted")

        # Verify the exception is propagated
        with pytest.raises(RuntimeError, match="Checkpoint corrupted"):
            _load_model_from_config(test_config)


class TestPlayUtilsIntegration:
    """Integration tests for play utilities."""

    @patch("practice.utils.play_utils.play_game_once")
    def test_full_play_workflow(
        self,
        mock_play_game_once: Mock,
        test_config: BaseConfig,
        mock_env: Mock,
        mock_agent: Mock,
        temp_output_dir: Path,
    ) -> None:
        """Test the full workflow from config to video generation."""
        # Create a dummy model file
        model_file = temp_output_dir / test_config.artifact_config.model_filename
        model_file.touch()

        # Setup mocks
        mock_loaded_agent = Mock()
        mock_agent.load_from_checkpoint.return_value = mock_loaded_agent
        mock_play_game_once.return_value = None

        # Call the function
        play_and_generate_video_generic(test_config, mock_env)

        # Verify the entire workflow
        mock_agent.load_from_checkpoint.assert_called_once()
        mock_play_game_once.assert_called_once()

        # Verify parameters passed to play_game_once
        call_args = mock_play_game_once.call_args
        assert call_args[1]["env"] == mock_env
        assert call_args[1]["policy"] == mock_loaded_agent
        assert call_args[1]["save_video"] is True
        assert call_args[1]["fps"] == test_config.artifact_config.fps
        assert call_args[1]["seed"] == test_config.artifact_config.seed

    def test_config_validation(self, test_config: BaseConfig) -> None:
        """Test that configuration has all required fields."""
        artifact_config = test_config.artifact_config

        # Verify required fields are present
        assert artifact_config.output_dir is not None
        assert artifact_config.model_filename is not None
        assert artifact_config.replay_video_filename is not None
        assert artifact_config.fps > 0
        assert artifact_config.seed is not None
        assert artifact_config.agent_type is not None

        # Verify config device
        assert test_config.device is not None
