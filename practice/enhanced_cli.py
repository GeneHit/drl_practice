"""Enhanced CLI for training RL algorithms with multiple modes.

Supports modes:
- train: Train model, then optionally play and generate video
- train --skip_play: Train model only, skip playing
- push_to_hub: Push model to hub, optionally play and generate video
- play: Only play game and generate video (no training)
- push_to_hub --skip_play: Push to hub only, skip playing
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Type, Union, Callable, Protocol

# Add the parent directory to Python path for imports when running directly
if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

import gymnasium as gym
import torch

from practice.base import BaseConfig, TrainerBase, AgentBase


class ModelLoader(Protocol):
    """Protocol for model loaders that can load checkpoints."""

    @classmethod
    def load_from_checkpoint(
        cls, pathname: str, device: Union[torch.device, None] = None
    ) -> AgentBase[Any]:
        """Load model from checkpoint file."""
        ...


def load_config_from_file(config_path: str, config_class: Type[BaseConfig]) -> BaseConfig:
    """Load configuration from file."""
    
    if config_path.endswith('.json'):
        # Load from JSON (legacy support)
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Extract relevant parameters based on the original structure
        hyper_params = config_data.get("hyper_params", {})
        env_params = config_data.get("env_params", {})
        output_params = config_data.get("output_params", {})
        eval_params = config_data.get("eval_params", {})
        hub_params = config_data.get("hub_params", {})
        
        # Build config dict for the new config classes
        config_dict = {}
        config_dict.update(hyper_params)
        
        # Environment parameters
        config_dict["env_id"] = env_params.get("env_id", "")
        # Extract env_kwargs properly, excluding env_id
        env_kwargs = env_params.copy()
        env_kwargs.pop("env_id", None)  # Remove env_id from kwargs
        config_dict["env_kwargs"] = env_kwargs
        
        # Output parameters
        config_dict["output_dir"] = output_params.get("output_dir", "results/")
        config_dict["save_result"] = output_params.get("save_result", True)
        config_dict["model_filename"] = output_params.get("model_filename", "model.pth")
        config_dict["params_filename"] = output_params.get("params_filename", "params.json")
        config_dict["train_result_filename"] = output_params.get("train_result_filename", "train_result.json")
        config_dict["eval_result_filename"] = output_params.get("eval_result_filename", "eval_result.json")
        
        # Evaluation parameters
        config_dict["eval_episodes"] = eval_params.get("eval_episodes", 100)
        config_dict["eval_seed"] = eval_params.get("eval_seed", None)
        
        # Hub parameters
        config_dict["repo_id"] = hub_params.get("repo_id", "")
        
        return config_class(**config_dict)
    
    elif config_path.endswith('.py'):
        # Load from Python file
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load config from {config_path}")
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Look for config instance in the module
        for name in dir(config_module):
            obj = getattr(config_module, name)
            if isinstance(obj, config_class) or (hasattr(obj, '__class__') and issubclass(obj.__class__, config_class)):
                return obj
        
        # If not found, look for any config object that might be the right type
        for name in dir(config_module):
            obj = getattr(config_module, name)
            if hasattr(obj, 'validate') and hasattr(obj, 'env_id'):
                # Try to convert to the expected config class
                try:
                    if hasattr(obj, 'to_dict'):
                        config_dict = obj.to_dict()
                    elif hasattr(obj, '__dict__'):
                        config_dict = obj.__dict__
                    else:
                        continue
                    return config_class(**config_dict)
                except Exception:
                    continue
        
        raise ValueError(f"No {config_class.__name__} instance found in {config_path}")
    
    else:
        raise ValueError("Config file must be .json or .py")


def create_env_from_config(config: BaseConfig, render_mode: str = "rgb_array") -> gym.Env[Any, Any]:
    """Create environment from configuration."""
    env_kwargs = config.env_kwargs.copy()
    if render_mode:
        env_kwargs["render_mode"] = render_mode
    
    return gym.make(config.env_id, **env_kwargs)


def load_model_from_config(config: BaseConfig, model_loader: ModelLoader, device: torch.device | None = None) -> AgentBase[Any]:
    """Load a trained model from configuration."""
    model_path = Path(config.output_dir) / config.model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return model_loader.load_from_checkpoint(str(model_path), device)


def play_and_generate_video(
    config: BaseConfig,
    model_loader: ModelLoader,
    device: torch.device | None = None,
    fps: int = 10,
    seed: int = 99
) -> None:
    """Play the game using trained model and generate video."""
    print(f"Playing {config.env_id} with trained model...")
    
    # Load the trained model
    agent = load_model_from_config(config, model_loader, device)
    
    # Create environment for playing
    env = create_env_from_config(config, render_mode="rgb_array")
    
    # Play one episode and record
    frames = []
    state, _ = env.reset(seed=seed)
    episode_reward = 0.0
    
    for step in range(1000):  # Max steps to prevent infinite loops
        action = agent.action(state)
        
        # Record frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        
        if terminated or truncated:
            break
    
    env.close()
    
    # Save video if frames were captured
    if frames:
        import imageio
        video_path = Path(config.output_dir) / "replay.mp4"
        imageio.mimsave(str(video_path), frames, fps=fps)
        print(f"Video saved to: {video_path}")
    
    print(f"Episode reward: {episode_reward}")


def push_to_hub(config: BaseConfig, username: str, algorithm_name: str) -> None:
    """Push model to Hugging Face Hub."""
    print(f"Pushing {algorithm_name} model to hub...")
    
    # This is a placeholder - in real implementation, you would:
    # 1. Load the model
    # 2. Create a model card
    # 3. Upload to Hugging Face Hub
    # 4. Handle authentication
    
    repo_id = f"{username}/{config.repo_id}" if config.repo_id else f"{username}/{algorithm_name.lower()}-model"
    print(f"Would push to: {repo_id}")
    print("Hub push functionality not implemented yet.")


class ModeHandler:
    """Handles different execution modes."""
    
    def __init__(
        self,
        algorithm_name: str,
        config_class: Type[BaseConfig],
        trainer_class: Type[TrainerBase[Any]],
        model_loader: ModelLoader,
        network_factory: Callable[[BaseConfig, torch.device], Any] | None = None
    ):
        self.algorithm_name = algorithm_name
        self.config_class = config_class
        self.trainer_class = trainer_class
        self.model_loader = model_loader
        self.network_factory = network_factory
    
    def handle_train_mode(self, config: BaseConfig, skip_play: bool = False) -> None:
        """Handle train mode."""
        print(f"Training {self.algorithm_name}...")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Create network if factory is provided
        network = None
        if self.network_factory is not None:
            network = self.network_factory(config, device)
        
        # Create trainer
        trainer = self.trainer_class(config, network, device)
        
        # Create log directory
        log_dir = f"logs/{self.algorithm_name.lower()}"
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Train the model
        trained_agent = trainer.train(log_dir)
        
        # Save results
        trainer.save_config()
        trainer.save_training_results()
        
        print(f"Training completed! Model saved to: {config.output_dir}/{config.model_filename}")
        
        # Play and generate video if not skipped
        if not skip_play:
            try:
                play_and_generate_video(config, self.model_loader, device)
            except Exception as e:
                print(f"Warning: Could not generate video: {e}")
    
    def handle_push_to_hub_mode(self, config: BaseConfig, username: str, skip_play: bool = False) -> None:
        """Handle push to hub mode."""
        push_to_hub(config, username, self.algorithm_name)
        
        # Play and generate video if not skipped
        if not skip_play:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                play_and_generate_video(config, self.model_loader, device)
            except Exception as e:
                print(f"Warning: Could not generate video: {e}")
    
    def handle_play_only_mode(self, config: BaseConfig) -> None:
        """Handle play only mode."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        play_and_generate_video(config, self.model_loader, device)
    
    def execute_mode(
        self,
        mode: str,
        config: BaseConfig,
        username: str = "",
        skip_play: bool = False
    ) -> None:
        """Execute the specified mode."""
        if mode == "train":
            self.handle_train_mode(config, skip_play)
        elif mode == "push_to_hub":
            if not username:
                raise ValueError("Username is required for push_to_hub mode")
            self.handle_push_to_hub_mode(config, username, skip_play)
        elif mode == "play":
            self.handle_play_only_mode(config)
        else:
            raise ValueError(f"Unknown mode: {mode}")


def create_enhanced_parser(algorithm_name: str, config_example: str = "config.json") -> argparse.ArgumentParser:
    """Create argument parser with mode support."""
    parser = argparse.ArgumentParser(
        description=f"{algorithm_name} training and evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Modes:
  train         : Train model, then optionally play and generate video
  push_to_hub   : Push model to hub, optionally play and generate video  
  play          : Only play game and generate video (no training)

Examples:
  # Train model and play
  python {algorithm_name.lower()}_exercise.py --config {config_example} --mode train
  
  # Train model only (skip playing)
  python {algorithm_name.lower()}_exercise.py --config {config_example} --mode train --skip_play
  
  # Push to hub and play
  python {algorithm_name.lower()}_exercise.py --config {config_example} --mode push_to_hub --username your_username
  
  # Push to hub only (skip playing)
  python {algorithm_name.lower()}_exercise.py --config {config_example} --mode push_to_hub --username your_username --skip_play
  
  # Play only (using existing model)
  python {algorithm_name.lower()}_exercise.py --config {config_example} --mode play
"""
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to configuration file (.json or .py)"
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["train", "push_to_hub", "play"],
        default="train",
        help="Execution mode (default: train)"
    )
    
    parser.add_argument(
        "--username", "-u",
        type=str,
        default="",
        help="Hugging Face username (required for push_to_hub mode)"
    )
    
    parser.add_argument(
        "--skip_play",
        action="store_true",
        help="Skip playing game and generating video"
    )
    
    return parser


def create_enhanced_main_function(
    algorithm_name: str,
    config_class: Type[BaseConfig],
    trainer_class: Type[TrainerBase[Any]],
    model_loader: ModelLoader,
    network_factory: Callable[[BaseConfig, torch.device], Any] | None = None,
    config_example: str = "config.json"
) -> Callable[[], None]:
    """Create main function with enhanced mode support."""
    
    def main() -> None:
        # Parse arguments
        parser = create_enhanced_parser(algorithm_name, config_example)
        args = parser.parse_args()
        
        # Validate arguments
        if args.mode == "push_to_hub" and not args.username:
            parser.error("--username is required when using --mode push_to_hub")
        
        if not Path(args.config).exists():
            parser.error(f"Config file not found: {args.config}")
        
        try:
            # Load configuration
            config = load_config_from_file(args.config, config_class)
            
            # Create mode handler
            mode_handler = ModeHandler(
                algorithm_name, config_class, trainer_class, model_loader, network_factory
            )
            
            # Execute mode
            mode_handler.execute_mode(args.mode, config, args.username, args.skip_play)
            
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    return main


def detect_algorithm_from_config(config_path: str) -> str:
    """Detect algorithm type from configuration file."""
    if config_path.endswith('.json'):
        # Load from JSON to detect algorithm
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Try to detect from hyper_params or file structure
        hyper_params = config_data.get("hyper_params", {})
        
        # Check for DQN-specific parameters
        if any(key in hyper_params for key in ["learning_rate", "epsilon_start", "replay_buffer_capacity", "batch_size"]):
            return "dqn"
        # Check for Q-table specific parameters  
        elif any(key in hyper_params for key in ["min_epsilon", "max_epsilon", "decay_rate"]):
            return "qtable"
        # Check for REINFORCE parameters
        elif any(key in hyper_params for key in ["baseline", "entropy_coef", "policy_lr"]):
            return "reinforce"
        else:
            # Default to DQN for unknown JSON configs
            return "dqn"
    
    elif config_path.endswith('.py'):
        # Detect from path or content
        if "qtable" in config_path.lower():
            return "qtable"
        elif "dqn" in config_path.lower():
            return "dqn"
        elif "reinforce" in config_path.lower():
            return "reinforce"
        else:
            # Try to import and inspect the config
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", config_path)
            if spec is None or spec.loader is None:
                raise ValueError(f"Could not load config from {config_path}")
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            # Look for config class name
            for name in dir(config_module):
                obj = getattr(config_module, name)
                if hasattr(obj, '__class__'):
                    class_name = obj.__class__.__name__.lower()
                    if "qtable" in class_name:
                        return "qtable"
                    elif "dqn" in class_name:
                        return "dqn"
                    elif "reinforce" in class_name:
                        return "reinforce"
            
            # Default to qtable if can't detect
            return "qtable"
    
    else:
        raise ValueError("Config file must be .json or .py")


def create_unified_parser() -> argparse.ArgumentParser:
    """Create unified argument parser for all algorithms."""
    parser = argparse.ArgumentParser(
        description="Unified RL training and evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  train         : Train model, then optionally play and generate video
  push_to_hub   : Push model to hub, optionally play and generate video  
  play          : Only play game and generate video (no training)

Examples:
  # Train Q-table
  python practice/enhanced_cli.py --config practice/exercise1_qtable/qtable_config.py --mode train
  
  # Train DQN and skip playing
  python practice/enhanced_cli.py --config practice/exercise2_dqn/obs_1d_config.py --mode train --skip_play
  
  # Push DQN to hub
  python practice/enhanced_cli.py --config practice/exercise2_dqn/test_config.py --mode push_to_hub --username your_username
  
  # Play only (using existing model)
  python practice/enhanced_cli.py --config practice/exercise1_qtable/qtable_config.py --mode play

  # Legacy JSON support
  python practice/enhanced_cli.py --config practice/exercise2_dqn/obs_1d_config.json --mode train
"""
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to configuration file (.json or .py)"
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["train", "push_to_hub", "play"],
        default="train",
        help="Execution mode (default: train)"
    )
    
    parser.add_argument(
        "--username", "-u",
        type=str,
        default="",
        help="Hugging Face username (required for push_to_hub mode)"
    )
    
    parser.add_argument(
        "--skip_play",
        action="store_true",
        help="Skip playing game and generating video"
    )
    
    parser.add_argument(
        "--algorithm", "-a",
        choices=["qtable", "dqn", "reinforce", "enhanced_reinforce"],
        help="Force specific algorithm (auto-detected if not provided)"
    )
    
    return parser


def get_algorithm_components(algorithm: str) -> tuple[Type[BaseConfig], Type[TrainerBase[Any]], ModelLoader, Any]:
    """Get the config class, trainer class, model loader, and network factory for an algorithm."""
    if algorithm == "qtable":
        from practice.exercise1_qtable.qtable_config import QTableConfig
        from practice.exercise1_qtable.qtable_exercise import QTableTrainer, QTable
        return QTableConfig, QTableTrainer, QTable, None
    
    elif algorithm == "dqn":
        from practice.exercise2_dqn.dqn_config import DQNConfig
        from practice.exercise2_dqn.dqn_exercise import DQNTrainer, DQNAgent, create_dqn_network
        return DQNConfig, DQNTrainer, DQNAgent, create_dqn_network
    
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def unified_main() -> None:
    """Unified main function for all algorithms."""
    # Parse arguments
    parser = create_unified_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "push_to_hub" and not args.username:
        parser.error("--username is required when using --mode push_to_hub")
    
    if not Path(args.config).exists():
        parser.error(f"Config file not found: {args.config}")
    
    try:
        # Detect algorithm if not specified
        algorithm = args.algorithm
        if algorithm is None:
            algorithm = detect_algorithm_from_config(args.config)
            print(f"Auto-detected algorithm: {algorithm.upper()}")
        
        # Get algorithm components
        config_class, trainer_class, model_loader, network_factory = get_algorithm_components(algorithm)
        
        # Load configuration
        config = load_config_from_file(args.config, config_class)
        
        # Create mode handler
        mode_handler = ModeHandler(
            algorithm.upper(), config_class, trainer_class, model_loader, network_factory
        )
        
        # Execute mode
        mode_handler.execute_mode(args.mode, config, args.username, args.skip_play)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    unified_main() 