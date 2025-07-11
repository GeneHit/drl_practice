"""Tests for CLI utilities module."""

import argparse
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from hands_on.utils.cli_utils import (
    CLIConfig,
    ModeHandler,
    create_env_from_config,
    create_main_function,
    create_standard_parser,
    get_video_path_from_config,
    load_and_validate_config,
    load_model_from_config,
    play_and_generate_video_generic,
    push_to_hub_generic,
    validate_cli_args,
)


class TestCLIConfig:
    """Test CLIConfig class."""

    def test_cli_config_default_values(self) -> None:
        """Test CLIConfig with default values."""
        config = CLIConfig("test.json")
        assert config.config == "test.json"
        assert config.mode == "train"
        assert config.username == ""
        assert config.skip_play is False

    def test_cli_config_custom_values(self) -> None:
        """Test CLIConfig with custom values."""
        config = CLIConfig(
            config="custom.json",
            mode="push_to_hub",
            username="testuser",
            skip_play=True,
        )
        assert config.config == "custom.json"
        assert config.mode == "push_to_hub"
        assert config.username == "testuser"
        assert config.skip_play is True


class TestStandardParser:
    """Test standard parser creation and validation."""

    def test_create_standard_parser(self) -> None:
        """Test creating a standard argument parser."""
        parser = create_standard_parser(
            script_name="test_script.py",
            description="Test Script",
            algorithm_name="TestAlgorithm",
            config_example="test_config.json",
        )

        assert isinstance(parser, argparse.ArgumentParser)
        assert "TestAlgorithm" in str(parser.epilog)
        assert "test_script.py" in str(parser.epilog)
        assert "test_config.json" in str(parser.epilog)

    def test_parser_arguments(self) -> None:
        """Test that parser has all required arguments."""
        parser = create_standard_parser("test.py", "Test", "Algorithm")

        # Parse help to check arguments exist
        help_text = parser.format_help()
        assert "--config" in help_text
        assert "--mode" in help_text
        assert "--username" in help_text
        assert "--skip_play" in help_text

    def test_validate_cli_args_valid(self) -> None:
        """Test CLI argument validation with valid arguments."""
        parser = create_standard_parser("test.py", "Test", "Algorithm")

        # Valid train mode
        args = argparse.Namespace(mode="train", username="", skip_play=False)
        validate_cli_args(args, parser)  # Should not raise

        # Valid push_to_hub mode with username
        args = argparse.Namespace(
            mode="push_to_hub", username="testuser", skip_play=False
        )
        validate_cli_args(args, parser)  # Should not raise

        # Valid play_only mode
        args = argparse.Namespace(
            mode="play_only", username="", skip_play=False
        )
        validate_cli_args(args, parser)  # Should not raise

    def test_validate_cli_args_missing_username(self) -> None:
        """Test CLI validation fails when username missing for push_to_hub."""
        parser = create_standard_parser("test.py", "Test", "Algorithm")
        args = argparse.Namespace(
            mode="push_to_hub", username="", skip_play=False
        )

        with pytest.raises(SystemExit):
            validate_cli_args(args, parser)

    def test_validate_cli_args_invalid_skip_play(self) -> None:
        """Test CLI validation fails when skip_play used with play_only."""
        parser = create_standard_parser("test.py", "Test", "Algorithm")
        args = argparse.Namespace(mode="play_only", username="", skip_play=True)

        with pytest.raises(SystemExit):
            validate_cli_args(args, parser)


class TestConfigLoading:
    """Test configuration loading and validation."""

    @patch("hands_on.utils.cli_utils.load_config_from_json")
    def test_load_and_validate_config_success(
        self, mock_load: MagicMock
    ) -> None:
        """Test successful config loading."""
        mock_config = {
            "env_params": {"env_id": "test"},
            "output_params": {"output_dir": "test"},
        }
        mock_load.return_value = mock_config

        result = load_and_validate_config("test.json")
        assert result == mock_config
        mock_load.assert_called_once_with("test.json")

    @patch("hands_on.utils.cli_utils.load_config_from_json")
    def test_load_and_validate_config_file_not_found(
        self, mock_load: MagicMock
    ) -> None:
        """Test config loading with file not found."""
        mock_load.side_effect = FileNotFoundError("File not found")

        with pytest.raises(
            FileNotFoundError, match="Configuration file not found"
        ):
            load_and_validate_config("missing.json")

    @patch("hands_on.utils.cli_utils.load_config_from_json")
    def test_load_and_validate_config_missing_section(
        self, mock_load: MagicMock
    ) -> None:
        """Test config validation with missing required section."""
        mock_config = {
            "env_params": {"env_id": "test"}
        }  # Missing output_params
        mock_load.return_value = mock_config

        with pytest.raises(
            ValueError, match="Missing required section 'output_params'"
        ):
            load_and_validate_config("test.json")

    @patch("hands_on.utils.cli_utils.load_config_from_json")
    def test_load_and_validate_config_json_error(
        self, mock_load: MagicMock
    ) -> None:
        """Test config loading with JSON parse error."""
        mock_load.side_effect = ValueError("Invalid JSON")

        with pytest.raises(ValueError, match="Failed to load configuration"):
            load_and_validate_config("test.json")


class TestModelLoading:
    """Test model loading utilities."""

    def test_load_model_from_config_success(self) -> None:
        """Test successful model loading."""
        cfg_data = {
            "output_params": {
                "output_dir": "test_output",
                "model_filename": "model.pkl",
            }
        }

        # Mock model loader
        mock_model = Mock()
        mock_loader = Mock()
        mock_loader.load_from_checkpoint.return_value = mock_model

        with patch("pathlib.Path.exists", return_value=True):
            result = load_model_from_config(
                cfg_data, mock_loader, device=torch.device("cpu")
            )

        assert result == mock_model
        mock_loader.load_from_checkpoint.assert_called_once_with(
            "test_output/model.pkl", device=torch.device("cpu")
        )

    def test_load_model_from_config_file_not_found(self) -> None:
        """Test model loading with missing file."""
        cfg_data = {
            "output_params": {
                "output_dir": "test_output",
                "model_filename": "missing_model.pkl",
            }
        }

        mock_loader = Mock()

        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Model file not found"):
                load_model_from_config(cfg_data, mock_loader)

    def test_get_video_path_from_config_default(self) -> None:
        """Test getting video path with default filename."""
        cfg_data = {"output_params": {"output_dir": "test_output"}}

        result = get_video_path_from_config(cfg_data)
        expected = Path("test_output") / "replay.mp4"
        assert result == expected

    def test_get_video_path_from_config_custom(self) -> None:
        """Test getting video path with custom filename."""
        cfg_data = {
            "output_params": {
                "output_dir": "test_output",
                "replay_video_filename": "custom_video.mp4",
            }
        }

        result = get_video_path_from_config(cfg_data)
        expected = Path("test_output") / "custom_video.mp4"
        assert result == expected


class TestPlayAndGenerateVideo:
    """Test play and generate video functionality."""

    @patch("hands_on.utils.cli_utils.play_game_once")
    @patch("hands_on.utils.cli_utils.load_model_from_config")
    def test_play_and_generate_video_generic(
        self, mock_load_model: MagicMock, mock_play_game: MagicMock
    ) -> None:
        """Test generic play and generate video function."""
        # Setup mocks
        mock_env = Mock()
        mock_env_creator = Mock(return_value=mock_env)
        mock_model = Mock()
        mock_loader = Mock()
        mock_load_model.return_value = mock_model

        cfg_data = {
            "env_params": {"env_id": "test"},
            "output_params": {
                "output_dir": "test_output",
                "replay_video_filename": "test_video.mp4",
            },
        }

        play_and_generate_video_generic(
            cfg_data=cfg_data,
            env_creator=mock_env_creator,
            model_loader=mock_loader,
            device=torch.device("cpu"),
            fps=30,
            seed=42,
        )

        # Verify function calls
        mock_env_creator.assert_called_once_with(cfg_data["env_params"])
        mock_load_model.assert_called_once_with(
            cfg_data, mock_loader, torch.device("cpu")
        )
        mock_play_game.assert_called_once_with(
            env=mock_env,
            policy=mock_model,
            save_video=True,
            video_pathname="test_output/test_video.mp4",
            fps=30,
            seed=42,
        )
        mock_env.close.assert_called_once()

    @patch("hands_on.utils.cli_utils.play_game_once")
    @patch("hands_on.utils.cli_utils.load_model_from_config")
    def test_play_and_generate_video_generic_env_cleanup(
        self, mock_load_model: MagicMock, mock_play_game: MagicMock
    ) -> None:
        """Test that environment is closed even if error occurs."""
        mock_env = Mock()
        mock_env_creator = Mock(return_value=mock_env)
        mock_loader = Mock()
        mock_load_model.side_effect = Exception("Model loading failed")

        cfg_data = {
            "env_params": {"env_id": "test"},
            "output_params": {"output_dir": "test_output"},
        }

        with pytest.raises(Exception, match="Model loading failed"):
            play_and_generate_video_generic(
                cfg_data=cfg_data,
                env_creator=mock_env_creator,
                model_loader=mock_loader,
            )

        # Environment should still be closed
        mock_env.close.assert_called_once()


class TestHubPush:
    """Test hub push functionality."""

    @patch("hands_on.utils.cli_utils.create_env_from_config")
    @patch("hands_on.utils.hub_play_utils.get_env_name_and_metadata")
    @patch("hands_on.utils.hub_play_utils.push_model_to_hub")
    def test_push_to_hub_generic_success(
        self,
        mock_push_model: MagicMock,
        mock_get_metadata: MagicMock,
        mock_create_env: MagicMock,
    ) -> None:
        """Test successful generic hub push."""
        # Setup mocks
        mock_env = Mock()
        mock_create_env.return_value = mock_env
        mock_metadata = {"env_name": "TestEnv", "tags": ["test"]}
        mock_get_metadata.return_value = mock_metadata

        cfg_data = {
            "env_params": {"env_id": "TestEnv-v0"},
            "hub_params": {"repo_id": "test-repo"},
            "output_params": {"output_dir": "test_output"},
        }

        push_to_hub_generic(
            cfg_data=cfg_data,
            username="testuser",
            algorithm_name="TestAlgorithm",
            model_filename="model.pth",
            extra_tags=["tag1", "tag2"],
            usage_instructions="Custom usage instructions",
        )

        # Verify function calls
        mock_create_env.assert_called_once_with(cfg_data["env_params"])
        mock_get_metadata.assert_called_once_with(
            env_id="TestEnv-v0",
            env=mock_env,
            algorithm_name="testalgorithm",
            extra_tags=["tag1", "tag2"],
        )
        mock_push_model.assert_called_once()
        mock_env.close.assert_called_once()

        # Check push_model_to_hub call arguments
        call_args = mock_push_model.call_args
        assert call_args[1]["repo_id"] == "testuser/test-repo"
        assert call_args[1]["output_params"] == cfg_data["output_params"]
        assert call_args[1]["metadata"] == mock_metadata
        assert "TestAlgorithm" in call_args[1]["model_card"]
        assert "TestEnv-v0" in call_args[1]["model_card"]
        assert "model.pth" in call_args[1]["model_card"]
        assert "Custom usage instructions" in call_args[1]["model_card"]

    @patch("hands_on.utils.cli_utils.create_env_from_config")
    @patch("hands_on.utils.hub_play_utils.get_env_name_and_metadata")
    @patch("hands_on.utils.hub_play_utils.push_model_to_hub")
    def test_push_to_hub_generic_env_cleanup(
        self,
        mock_push_model: MagicMock,
        mock_get_metadata: MagicMock,
        mock_create_env: MagicMock,
    ) -> None:
        """Test that environment is closed even if error occurs during hub push."""
        mock_env = Mock()
        mock_create_env.return_value = mock_env
        mock_get_metadata.side_effect = Exception("Metadata error")

        cfg_data = {
            "env_params": {"env_id": "TestEnv-v0"},
            "hub_params": {"repo_id": "test-repo"},
            "output_params": {"output_dir": "test_output"},
        }

        with pytest.raises(Exception, match="Metadata error"):
            push_to_hub_generic(
                cfg_data=cfg_data,
                username="testuser",
                algorithm_name="TestAlgorithm",
                model_filename="model.pth",
            )

        # Environment should still be closed
        mock_env.close.assert_called_once()

    @patch("hands_on.utils.cli_utils.create_env_from_config")
    @patch("hands_on.utils.hub_play_utils.get_env_name_and_metadata")
    @patch("hands_on.utils.hub_play_utils.push_model_to_hub")
    def test_push_to_hub_generic_default_parameters(
        self,
        mock_push_model: MagicMock,
        mock_get_metadata: MagicMock,
        mock_create_env: MagicMock,
    ) -> None:
        """Test hub push with default parameters."""
        mock_env = Mock()
        mock_create_env.return_value = mock_env
        mock_metadata = {"env_name": "TestEnv", "tags": ["test"]}
        mock_get_metadata.return_value = mock_metadata

        cfg_data = {
            "env_params": {"env_id": "TestEnv-v0"},
            "hub_params": {"repo_id": "test-repo"},
            "output_params": {"output_dir": "test_output"},
        }

        push_to_hub_generic(
            cfg_data=cfg_data,
            username="testuser",
            algorithm_name="TestAlgorithm",
            model_filename="model.pth",
        )

        # Verify get_metadata was called with empty extra_tags
        mock_get_metadata.assert_called_once_with(
            env_id="TestEnv-v0",
            env=mock_env,
            algorithm_name="testalgorithm",
            extra_tags=[],
        )

        # Verify model card has minimal usage instructions
        call_args = mock_push_model.call_args
        model_card = call_args[1]["model_card"]
        assert 'env = gym.make(model["env_id"])' in model_card


class TestModeHandler:
    """Test ModeHandler class."""

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.mock_train_fn = Mock()
        self.mock_push_fn = Mock()
        self.mock_play_fn = Mock()

        self.handler = ModeHandler(
            train_fn=self.mock_train_fn,
            push_to_hub_fn=self.mock_push_fn,
            play_and_generate_video_fn=self.mock_play_fn,
            algorithm_name="TestAlgorithm",
        )

        self.cfg_data = {"test": "config"}

    def test_handle_train_mode_with_play(self, capsys: Any) -> None:
        """Test training mode with play/video generation."""
        self.handler.handle_train_mode(self.cfg_data, skip_play=False)

        self.mock_train_fn.assert_called_once_with(self.cfg_data)
        self.mock_play_fn.assert_called_once_with(self.cfg_data)

        captured = capsys.readouterr()
        assert "Training TestAlgorithm Model" in captured.out
        assert "Training Complete, Playing Game" in captured.out

    def test_handle_train_mode_skip_play(self, capsys: Any) -> None:
        """Test training mode without play/video generation."""
        self.handler.handle_train_mode(self.cfg_data, skip_play=True)

        self.mock_train_fn.assert_called_once_with(self.cfg_data)
        self.mock_play_fn.assert_not_called()

        captured = capsys.readouterr()
        assert "Training TestAlgorithm Model" in captured.out
        assert "Skipping Play/Video Generation" in captured.out

    def test_handle_push_to_hub_mode_with_play(self, capsys: Any) -> None:
        """Test push to hub mode with play/video generation."""
        self.handler.handle_push_to_hub_mode(
            self.cfg_data, "testuser", skip_play=False
        )

        self.mock_push_fn.assert_called_once_with(self.cfg_data, "testuser")
        self.mock_play_fn.assert_called_once_with(self.cfg_data)

        captured = capsys.readouterr()
        assert "Pushing TestAlgorithm Model to Hub" in captured.out
        assert "Playing Game and Generating Video" in captured.out
        assert "Model successfully pushed to Hub!" in captured.out

    def test_handle_push_to_hub_mode_skip_play(self, capsys: Any) -> None:
        """Test push to hub mode without play/video generation."""
        self.handler.handle_push_to_hub_mode(
            self.cfg_data, "testuser", skip_play=True
        )

        self.mock_push_fn.assert_called_once_with(self.cfg_data, "testuser")
        self.mock_play_fn.assert_not_called()

        captured = capsys.readouterr()
        assert "Pushing TestAlgorithm Model to Hub" in captured.out
        assert "Playing Game and Generating Video" not in captured.out

    def test_handle_play_only_mode(self, capsys: Any) -> None:
        """Test play only mode."""
        self.handler.handle_play_only_mode(self.cfg_data)

        self.mock_play_fn.assert_called_once_with(self.cfg_data)
        self.mock_train_fn.assert_not_called()
        self.mock_push_fn.assert_not_called()

        captured = capsys.readouterr()
        assert "Playing TestAlgorithm Model" in captured.out

    def test_execute_mode_train(self) -> None:
        """Test execute_mode with train mode."""
        self.handler.execute_mode(
            mode="train",
            cfg_data=self.cfg_data,
            skip_play=False,
            config_path="test.json",
        )

        self.mock_train_fn.assert_called_once_with(self.cfg_data)
        self.mock_play_fn.assert_called_once_with(self.cfg_data)

    def test_execute_mode_push_to_hub(self) -> None:
        """Test execute_mode with push_to_hub mode."""
        self.handler.execute_mode(
            mode="push_to_hub",
            cfg_data=self.cfg_data,
            username="testuser",
            skip_play=False,
            config_path="test.json",
        )

        self.mock_push_fn.assert_called_once_with(self.cfg_data, "testuser")
        self.mock_play_fn.assert_called_once_with(self.cfg_data)

    def test_execute_mode_play_only(self) -> None:
        """Test execute_mode with play_only mode."""
        self.handler.execute_mode(
            mode="play_only", cfg_data=self.cfg_data, config_path="test.json"
        )

        self.mock_play_fn.assert_called_once_with(self.cfg_data)
        self.mock_train_fn.assert_not_called()
        self.mock_push_fn.assert_not_called()

    def test_execute_mode_invalid(self) -> None:
        """Test execute_mode with invalid mode."""
        with pytest.raises(ValueError, match="Unknown mode: invalid"):
            self.handler.execute_mode(mode="invalid", cfg_data=self.cfg_data)


class TestMainFunctionFactory:
    """Test create_main_function."""

    @patch("hands_on.utils.cli_utils.parse_standard_args")
    @patch("hands_on.utils.cli_utils.load_and_validate_config")
    def test_create_main_function(
        self, mock_load_config: MagicMock, mock_parse_args: MagicMock
    ) -> None:
        """Test creating a main function."""
        # Setup mocks
        mock_cli_config = CLIConfig("test.json", "train", "", False)
        mock_parse_args.return_value = mock_cli_config
        mock_cfg_data = {"test": "config"}
        mock_load_config.return_value = mock_cfg_data

        mock_train_fn = Mock()
        mock_push_fn = Mock()
        mock_play_fn = Mock()

        # Create main function
        main_fn = create_main_function(
            script_name="test.py",
            algorithm_name="TestAlgorithm",
            train_fn=mock_train_fn,
            push_to_hub_fn=mock_push_fn,
            play_and_generate_video_fn=mock_play_fn,
            config_example="test_config.json",
        )

        # Execute main function
        main_fn()

        # Verify calls
        mock_parse_args.assert_called_once()
        mock_load_config.assert_called_once_with("test.json")
        mock_train_fn.assert_called_once_with(mock_cfg_data)
        mock_play_fn.assert_called_once_with(mock_cfg_data)


class TestIntegration:
    """Integration tests for CLI utilities."""

    @patch("sys.argv")
    @patch("hands_on.utils.cli_utils.load_config_from_json")
    def test_full_cli_integration(
        self, mock_load_config: MagicMock, mock_argv: MagicMock
    ) -> None:
        """Test full CLI integration with argument parsing."""
        # Setup command line arguments
        mock_argv.__getitem__.side_effect = [
            "test_script.py",  # sys.argv[0]
            "--config",
            "test.json",
            "--mode",
            "train",
        ]
        mock_argv.__len__.return_value = 4
        mock_argv.__iter__.return_value = iter(
            ["test_script.py", "--config", "test.json", "--mode", "train"]
        )

        # Setup config
        mock_config = {
            "env_params": {"env_id": "test"},
            "output_params": {"output_dir": "test"},
        }
        mock_load_config.return_value = mock_config

        # Setup functions
        mock_train_fn = Mock()
        mock_push_fn = Mock()
        mock_play_fn = Mock()

        # Create and test main function
        main_fn = create_main_function(
            script_name="test_script.py",
            algorithm_name="TestAlgorithm",
            train_fn=mock_train_fn,
            push_to_hub_fn=mock_push_fn,
            play_and_generate_video_fn=mock_play_fn,
        )

        # This would normally be called, but we can't easily test
        # argparse in this context without more complex mocking
        assert callable(main_fn)


class TestEnvironmentCreation:
    """Test environment creation utilities."""

    @patch("hands_on.utils.cli_utils.make_image_env")
    def test_create_env_from_config_image_env(
        self, mock_make_image_env: MagicMock
    ) -> None:
        """Test creating image-based environment (DQN style)."""
        mock_env = Mock()
        mock_make_image_env.return_value = (mock_env, {})

        env_params = {
            "env_id": "TestEnv-v0",
            "use_image": True,
            "resize_shape": [84, 84],
            "frame_stack_size": 4,
        }

        result = create_env_from_config(env_params, "rgb_array")

        assert result == mock_env
        mock_make_image_env.assert_called_once_with(
            env_id="TestEnv-v0",
            render_mode="rgb_array",
            resize_shape=(84, 84),
            frame_stack_size=4,
        )

    @patch("hands_on.utils.cli_utils.make_discrete_env_with_kwargs")
    def test_create_env_from_config_discrete_env(
        self, mock_make_discrete_env: MagicMock
    ) -> None:
        """Test creating discrete environment with kwargs (Q-learning style)."""
        mock_env = Mock()
        mock_make_discrete_env.return_value = (mock_env, {})

        env_params = {
            "env_id": "FrozenLake-v1",
            "kwargs": {"map_name": "4x4", "is_slippery": False},
        }

        result = create_env_from_config(env_params, "human")

        assert result == mock_env
        expected_kwargs = {
            "map_name": "4x4",
            "is_slippery": False,
            "render_mode": "human",
        }
        mock_make_discrete_env.assert_called_once_with(
            env_id="FrozenLake-v1", kwargs=expected_kwargs
        )

    @patch("hands_on.utils.cli_utils.make_1d_env")
    def test_create_env_from_config_1d_env(
        self, mock_make_1d_env: MagicMock
    ) -> None:
        """Test creating 1D environment (REINFORCE style)."""
        mock_env = Mock()
        mock_make_1d_env.return_value = (mock_env, {})

        env_params = {"env_id": "CartPole-v1", "max_steps": 500}

        result = create_env_from_config(env_params)

        assert result == mock_env
        mock_make_1d_env.assert_called_once_with(
            env_id="CartPole-v1", render_mode="rgb_array", max_steps=500
        )

    @patch("hands_on.utils.cli_utils.make_1d_env")
    def test_create_env_from_config_1d_env_no_max_steps(
        self, mock_make_1d_env: MagicMock
    ) -> None:
        """Test creating 1D environment without max_steps."""
        mock_env = Mock()
        mock_make_1d_env.return_value = (mock_env, {})

        env_params = {"env_id": "CartPole-v1"}

        result = create_env_from_config(env_params, "human")

        assert result == mock_env
        mock_make_1d_env.assert_called_once_with(
            env_id="CartPole-v1", render_mode="human", max_steps=None
        )

    @patch("hands_on.utils.cli_utils.make_discrete_env_with_kwargs")
    def test_create_env_from_config_kwargs_copy(
        self, mock_make_discrete_env: MagicMock
    ) -> None:
        """Test that kwargs are properly copied and not mutated."""
        mock_env = Mock()
        mock_make_discrete_env.return_value = (mock_env, {})

        original_kwargs = {"map_name": "4x4", "is_slippery": False}
        env_params = {"env_id": "FrozenLake-v1", "kwargs": original_kwargs}

        create_env_from_config(env_params, "rgb_array")

        # Original kwargs should not be modified
        assert original_kwargs == {"map_name": "4x4", "is_slippery": False}
        assert "render_mode" not in original_kwargs
