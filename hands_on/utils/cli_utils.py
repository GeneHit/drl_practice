"""Common CLI utilities for handling argument parsing and modes across different RL exercises.

This module provides reusable components for:
1. Standard argument parsing (config, mode, username, skip_play)
2. Mode validation logic
3. Common mode descriptions and examples
4. Configuration loading and validation
5. Common play and video generation utilities
6. Model loading utilities
7. Common environment creation utilities
"""

import argparse
from pathlib import Path
from typing import Any, Callable, Protocol, Union

import torch

from hands_on.utils.env_utils import (
    make_1d_env,
    make_discrete_env_with_kwargs,
    make_image_env,
)
from hands_on.utils.evaluation_utils import play_game_once
from hands_on.utils.file_utils import load_config_from_json


class ModelLoader(Protocol):
    """Protocol for model loaders that can load checkpoints."""

    @classmethod
    def load_from_checkpoint(
        cls, pathname: str, device: Union[torch.device, None]
    ) -> Any:
        """Load model from checkpoint file."""
        ...


def load_model_from_config(
    cfg_data: dict[str, Any],
    model_loader: ModelLoader,
    device: Union[torch.device, None] = None,
) -> Any:
    """Load a trained model from configuration.

    Args:
        cfg_data: Configuration data containing output_params
        model_loader: Model class with load_from_checkpoint method
        device: Device to load the model on

    Returns:
        Loaded model instance

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    output_params = cfg_data["output_params"]
    output_dir = Path(output_params["output_dir"])
    model_path = output_dir / output_params["model_filename"]

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return model_loader.load_from_checkpoint(str(model_path), device=device)


def get_video_path_from_config(cfg_data: dict[str, Any]) -> Path:
    """Get video path from configuration.

    Args:
        cfg_data: Configuration data containing output_params

    Returns:
        Path object for the video file
    """
    output_params = cfg_data["output_params"]
    output_dir = Path(output_params["output_dir"])
    video_filename = output_params.get("replay_video_filename", "replay.mp4")
    return Path(output_dir / video_filename)


def play_and_generate_video_generic(
    cfg_data: dict[str, Any],
    env_creator: Callable[[dict[str, Any]], Any],
    model_loader: ModelLoader,
    device: Union[torch.device, None] = None,
    fps: int = 10,
    seed: int = 99,
) -> None:
    """Generic function to play game and generate video.

    Args:
        cfg_data: Configuration data
        env_creator: Function that creates environment from env_params
        model_loader: Model class with load_from_checkpoint method
        device: Device to load the model on
        fps: Video frame rate
        seed: Random seed for gameplay
    """
    print("=== Playing Game and Generating Video ===")

    # Create environment from config
    env = env_creator(cfg_data["env_params"])

    try:
        # Load the trained model
        model = load_model_from_config(cfg_data, model_loader, device)

        # Get video path
        video_path = get_video_path_from_config(cfg_data)

        # Play the game and save video
        play_game_once(
            env=env,
            policy=model,
            save_video=True,
            video_pathname=str(video_path),
            fps=fps,
            seed=seed,
        )
        print(f"Game replay saved to: {video_path}")
    finally:
        env.close()


def create_env_from_config(
    env_params: dict[str, Any], render_mode: str = "rgb_array"
) -> Any:
    """Create environment from configuration parameters.

    This is a universal environment factory that can handle different environment types
    based on the configuration parameters.

    Args:
        env_params: Environment parameters dictionary containing:
            - env_id: Environment ID to create
            - use_image (optional): If True, creates image-based environment
            - kwargs (optional): Additional kwargs for discrete environments
            - max_steps (optional): Maximum steps per episode
            - resize_shape (optional): Shape for image resizing
            - frame_stack_size (optional): Number of frames to stack
        render_mode: Render mode for the environment (default: "rgb_array")

    Returns:
        The created environment
    """
    env_id = env_params["env_id"]

    # Check if this is an image-based environment (DQN style)
    if env_params.get("use_image", False):
        env, _ = make_image_env(
            env_id=env_id,
            render_mode=render_mode,
            resize_shape=tuple(env_params["resize_shape"]),
            frame_stack_size=env_params["frame_stack_size"],
        )
    # Check if this is a discrete environment with custom kwargs (Q-learning style)
    elif "kwargs" in env_params:
        kwargs = env_params["kwargs"].copy()
        kwargs["render_mode"] = render_mode
        env, _ = make_discrete_env_with_kwargs(  # type: ignore[assignment]
            env_id=env_id,
            kwargs=kwargs,
        )
    # Default to 1D environment (REINFORCE style)
    else:
        env, _ = make_1d_env(
            env_id=env_id,
            render_mode=render_mode,
            max_steps=env_params.get("max_steps", None),
        )

    return env


def push_to_hub_generic(
    cfg_data: dict[str, Any],
    username: str,
    algorithm_name: str,
    model_filename: str,
    extra_tags: Union[list[str], None] = None,
    usage_instructions: str = "",
) -> None:
    """Generic function to push any model to Hugging Face Hub.

    Args:
        cfg_data: Configuration data containing env_params, hub_params, output_params
        username: Hugging Face username
        algorithm_name: Name of the algorithm (e.g., "Q-Learning", "DQN", "REINFORCE")
        model_filename: Name of the model file (e.g., "q-learning.pkl", "dqn.pth")
        extra_tags: Additional tags for the model (default: None)
        usage_instructions: Additional usage instructions for the model card
    """
    from hands_on.utils.hub_play_utils import (
        get_env_name_and_metadata,
        push_model_to_hub,
    )

    env_id = cfg_data["env_params"]["env_id"]

    # Create environment for metadata generation
    env = create_env_from_config(cfg_data["env_params"])

    try:
        # Get metadata with algorithm-specific tags
        metadata = get_env_name_and_metadata(
            env_id=env_id,
            env=env,
            algorithm_name=algorithm_name.lower(),
            extra_tags=extra_tags or [],
        )

        # Create repo_id
        repo_id = f"{username}/{cfg_data['hub_params']['repo_id']}"

        # Create model card with algorithm-specific content
        model_card = f"""
    # **{algorithm_name}** Agent playing **{env_id}**
    This is a trained model of a **{algorithm_name}** agent playing **{env_id}**.

    ## Usage

    model = load_from_hub(repo_id="{repo_id}", filename="{model_filename}")

    {usage_instructions}
    env = gym.make(model["env_id"])
    ...
    """

        # Push to hub
        push_model_to_hub(
            repo_id=repo_id,
            output_params=cfg_data["output_params"],
            model_card=model_card,
            metadata=metadata,
        )
    finally:
        env.close()


class CLIConfig:
    """Configuration container for CLI arguments."""

    def __init__(
        self,
        config: str,
        mode: str = "train",
        username: str = "",
        skip_play: bool = False,
    ):
        self.config = config
        self.mode = mode
        self.username = username
        self.skip_play = skip_play


def create_standard_parser(
    script_name: str,
    description: str,
    algorithm_name: str,
    config_example: str = "config.json",
) -> argparse.ArgumentParser:
    """Create a standard argument parser with common arguments.

    Args:
        script_name: Name of the script (e.g., "q_main.py", "dqn_main.py")
        description: Description for the parser
        algorithm_name: Name of the algorithm (e.g., "Q-learning", "DQN", "REINFORCE")
        config_example: Example config filename for help text

    Returns:
        Configured ArgumentParser instance
    """
    epilog = f"""
        Mode descriptions:
        train (default): Train the {algorithm_name} model, then play game and generate video
        push_to_hub: Push model to hub, optionally play game and generate video
        play_only: Only play game and generate video (no training or hub operations)

        Examples:
        # Train mode (default) - train model then play game and generate video
        python {script_name} --config {config_example}

        # Train mode without play/video generation
        python {script_name} --config {config_example} --skip_play

        # Push to hub with play/video
        python {script_name} --config {config_example} --mode push_to_hub --username myuser

        # Push to hub without play/video
        python {script_name} --config {config_example} --mode push_to_hub --username myuser --skip_play

        # Only play and generate video
        python {script_name} --config {config_example} --mode play_only
    """

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration JSON file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "push_to_hub", "play_only"],
        default="train",
        help="Mode of operation (default: train)",
    )
    parser.add_argument(
        "--username",
        type=str,
        default="",
        help="Hugging Face username (required for push_to_hub mode)",
    )
    parser.add_argument(
        "--skip_play",
        action="store_true",
        default=False,
        help="Skip playing the game and video generation",
    )

    return parser


def validate_cli_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """Validate CLI arguments with common validation logic.

    Args:
        args: Parsed arguments
        parser: ArgumentParser instance for error reporting

    Raises:
        SystemExit: If validation fails
    """
    # Validate username requirement for push_to_hub mode
    if args.mode == "push_to_hub" and not args.username:
        parser.error("--username is required when using push_to_hub mode")

    # Validate skip_play usage
    if args.skip_play and args.mode not in ["push_to_hub", "train"]:
        parser.error(
            "--skip_play can only be used with push_to_hub or train mode"
        )


def parse_standard_args(
    script_name: str,
    description: str,
    algorithm_name: str,
    config_example: str = "config.json",
) -> CLIConfig:
    """Parse standard CLI arguments and return validated configuration.

    Args:
        script_name: Name of the script
        description: Description for the parser
        algorithm_name: Name of the algorithm
        config_example: Example config filename

    Returns:
        CLIConfig instance with validated arguments
    """
    parser = create_standard_parser(
        script_name=script_name,
        description=description,
        algorithm_name=algorithm_name,
        config_example=config_example,
    )

    args = parser.parse_args()
    validate_cli_args(args, parser)

    return CLIConfig(
        config=args.config,
        mode=args.mode,
        username=args.username,
        skip_play=args.skip_play,
    )


def load_and_validate_config(config_path: str) -> dict[str, Any]:
    """Load configuration from JSON file with validation.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    try:
        cfg_data = load_config_from_json(config_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except Exception as e:
        raise ValueError(f"Failed to load configuration: {e}")

    # Basic validation of required sections
    required_sections = ["env_params", "output_params"]
    for section in required_sections:
        if section not in cfg_data:
            raise ValueError(f"Missing required section '{section}' in config")

    return cfg_data


class ModeHandler:
    """Handler for different execution modes with common logic."""

    def __init__(
        self,
        train_fn: Callable[[dict[str, Any]], None],
        push_to_hub_fn: Callable[[dict[str, Any], str], None],
        play_and_generate_video_fn: Callable[[dict[str, Any]], None],
        algorithm_name: str,
    ):
        """Initialize mode handler.

        Args:
            train_fn: Function to call for training (takes cfg_data)
            push_to_hub_fn: Function to call for pushing to hub (takes cfg_data, username)
            play_and_generate_video_fn: Function to call for playing/video generation
            algorithm_name: Name of the algorithm for logging
        """
        self.train_fn = train_fn
        self.push_to_hub_fn = push_to_hub_fn
        self.play_and_generate_video_fn = play_and_generate_video_fn
        self.algorithm_name = algorithm_name

    def handle_train_mode(
        self, cfg_data: dict[str, Any], skip_play: bool = False
    ) -> None:
        """Handle training mode with optional play/video generation."""
        print(f"=== Training {self.algorithm_name} Model ===")
        self.train_fn(cfg_data)

        if skip_play:
            print("=== Training Complete (Skipping Play/Video Generation) ===")
        else:
            print(
                "=== Training Complete, Playing Game and Generating Video ==="
            )
            self.play_and_generate_video_fn(cfg_data)

    def handle_push_to_hub_mode(
        self,
        cfg_data: dict[str, Any],
        username: str,
        skip_play: bool = False,
    ) -> None:
        """Handle push to hub mode with optional play/video generation."""
        print(f"=== Pushing {self.algorithm_name} Model to Hub ===")

        if not skip_play:
            print("=== Playing Game and Generating Video ===")
            self.play_and_generate_video_fn(cfg_data)

        self.push_to_hub_fn(cfg_data, username)
        print("Model successfully pushed to Hub!")

    def handle_play_only_mode(self, cfg_data: dict[str, Any]) -> None:
        """Handle play only mode."""
        print(f"=== Playing {self.algorithm_name} Model ===")
        self.play_and_generate_video_fn(cfg_data)

    def execute_mode(
        self,
        mode: str,
        cfg_data: dict[str, Any],
        username: str = "",
        skip_play: bool = False,
        config_path: str = "",
    ) -> None:
        """Execute the specified mode with configuration.

        Args:
            mode: Mode to execute ("train", "push_to_hub", "play_only")
            cfg_data: Configuration data
            username: Username for hub operations
            skip_play: Whether to skip play/video generation
            config_path: Path to config file (for logging)
        """
        log_config = f" with config: {config_path}" if config_path else ""

        if mode == "train":
            print(f"=== Training {self.algorithm_name} Model{log_config} ===")
            self.handle_train_mode(cfg_data, skip_play)
        elif mode == "push_to_hub":
            print(
                f"=== Pushing {self.algorithm_name} Model to Hub{log_config} ==="
            )
            self.handle_push_to_hub_mode(cfg_data, username, skip_play)
        elif mode == "play_only":
            print(f"=== Playing {self.algorithm_name} Model{log_config} ===")
            self.handle_play_only_mode(cfg_data)
        else:
            raise ValueError(f"Unknown mode: {mode}")


def create_main_function(
    script_name: str,
    algorithm_name: str,
    train_fn: Callable[[dict[str, Any]], None],
    push_to_hub_fn: Callable[[dict[str, Any], str], None],
    play_and_generate_video_fn: Callable[[dict[str, Any]], None],
    config_example: str = "config.json",
) -> Callable[[], None]:
    """Create a standardized main function for RL exercise scripts.

    Args:
        script_name: Name of the script
        algorithm_name: Name of the algorithm
        train_fn: Training function
        push_to_hub_fn: Hub push function
        play_and_generate_video_fn: Play/video generation function
        config_example: Example config filename

    Returns:
        Main function that can be called directly
    """

    def main() -> None:
        # Parse arguments
        cli_config = parse_standard_args(
            script_name=script_name,
            description=f"{algorithm_name} Main Script with Multiple Modes",
            algorithm_name=algorithm_name,
            config_example=config_example,
        )

        # Load and validate configuration
        cfg_data = load_and_validate_config(cli_config.config)

        # Create mode handler and execute
        mode_handler = ModeHandler(
            train_fn=train_fn,
            push_to_hub_fn=push_to_hub_fn,
            play_and_generate_video_fn=play_and_generate_video_fn,
            algorithm_name=algorithm_name,
        )

        mode_handler.execute_mode(
            mode=cli_config.mode,
            cfg_data=cfg_data,
            username=cli_config.username,
            skip_play=cli_config.skip_play,
            config_path=cli_config.config,
        )

    return main
