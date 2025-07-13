"""Unified CLI for training RL algorithms."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Type, Union

import gymnasium as gym
import torch

from practice.base import BaseConfig, TrainerBase
from practice.qtable import QTableConfig, QTableTrainer
from practice.dqn import DQNConfig, DQNTrainer  
from practice.reinforce import ReinforceConfig, ReinforceTrainer
from practice.enhanced_reinforce import EnhancedReinforceConfig, EnhancedReinforceTrainer


# Algorithm registry
ALGORITHMS = {
    "qtable": {
        "config_class": QTableConfig,
        "trainer_class": QTableTrainer,
        "requires_discrete_obs": True,
        "requires_discrete_action": True,
    },
    "dqn": {
        "config_class": DQNConfig,
        "trainer_class": DQNTrainer,
        "requires_discrete_obs": False,
        "requires_discrete_action": True,
    },
    "reinforce": {
        "config_class": ReinforceConfig,
        "trainer_class": ReinforceTrainer,
        "requires_discrete_obs": False,
        "requires_discrete_action": True,
    },
    "enhanced_reinforce": {
        "config_class": EnhancedReinforceConfig,
        "trainer_class": EnhancedReinforceTrainer,
        "requires_discrete_obs": False,
        "requires_discrete_action": True,
    },
}


def load_config_from_file(config_path: str, algorithm: str) -> BaseConfig:
    """Load configuration from file."""
    config_class = ALGORITHMS[algorithm]["config_class"]
    
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
        config_dict["env_kwargs"] = env_params.get("kwargs", {})
        
        # Output parameters
        config_dict["output_dir"] = output_params.get("output_dir", "results/")
        config_dict["save_result"] = output_params.get("save_result", True)
        config_dict["model_filename"] = output_params.get("model_filename", "model.pkl")
        
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
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Look for config instance in the module
        for name in dir(config_module):
            obj = getattr(config_module, name)
            if isinstance(obj, config_class):
                return obj
        
        raise ValueError(f"No {config_class.__name__} instance found in {config_path}")
    
    else:
        raise ValueError("Config file must be .json or .py")


def create_trainer(algorithm: str, env: gym.Env, device: torch.device) -> TrainerBase:
    """Create trainer based on algorithm and environment."""
    algo_info = ALGORITHMS[algorithm]
    trainer_class = algo_info["trainer_class"]
    
    # Get environment dimensions
    if algo_info["requires_discrete_obs"]:
        if not hasattr(env.observation_space, 'n'):
            raise ValueError(f"Algorithm {algorithm} requires discrete observation space")
        obs_dim = env.observation_space.n
    else:
        if hasattr(env.observation_space, 'shape'):
            obs_dim = env.observation_space.shape[0]
        else:
            raise ValueError(f"Unsupported observation space for {algorithm}")
    
    if algo_info["requires_discrete_action"]:
        if not hasattr(env.action_space, 'n'):
            raise ValueError(f"Algorithm {algorithm} requires discrete action space")
        action_dim = env.action_space.n
    else:
        if hasattr(env.action_space, 'shape'):
            action_dim = env.action_space.shape[0]
        else:
            raise ValueError(f"Unsupported action space for {algorithm}")
    
    # Create trainer
    if algorithm == "qtable":
        return trainer_class(obs_dim, action_dim)
    else:
        return trainer_class(obs_dim, action_dim, device)


def train_algorithm(config: BaseConfig, algorithm: str, log_dir: str) -> None:
    """Train an algorithm with the given configuration."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment for introspection
    temp_env = gym.make(config.env_id, **config.env_kwargs)
    
    # Create trainer
    trainer = create_trainer(algorithm, temp_env, device)
    temp_env.close()
    
    # Train the algorithm
    print(f"Training {algorithm} with config:")
    print(f"  Environment: {config.env_id}")
    print(f"  Episodes: {config.episodes}")
    print(f"  Output directory: {config.output_dir}")
    
    trained_agent = trainer.train(config, log_dir)
    
    print(f"Training completed! Model saved to: {config.output_dir}/{config.model_filename}")


def create_example_configs() -> None:
    """Create example configuration files for each algorithm."""
    examples_dir = Path("practice/configs/examples")
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    # Q-table example
    qtable_config = QTableConfig(
        env_id="FrozenLake-v1",
        env_kwargs={"map_name": "4x4", "is_slippery": False, "render_mode": "rgb_array"},
        episodes=1000,
        max_steps=99,
        gamma=0.95,
        eval_episodes=100,
        eval_seed=None,
        output_dir="results/qtable/frozen_lake/",
        save_result=True,
        model_filename="qtable.pkl",
        repo_id="",
        learning_rate=0.7,
        min_epsilon=0.05,
        max_epsilon=1.0,
        decay_rate=0.0005
    )
    
    with open(examples_dir / "qtable_config.py", "w") as f:
        f.write(f"""\"\"\"Example Q-table configuration for FrozenLake.\"\"\"

from practice.qtable import QTableConfig

config = {repr(qtable_config)}
""")
    
    # DQN example
    dqn_config = DQNConfig(
        env_id="CartPole-v1",
        env_kwargs={"render_mode": "rgb_array"},
        episodes=500,
        max_steps=500,
        gamma=0.99,
        eval_episodes=100,
        eval_seed=None,
        output_dir="results/dqn/cartpole/",
        save_result=True,
        model_filename="dqn.pth",
        repo_id="",
        learning_rate=0.001,
        min_epsilon=0.05,
        max_epsilon=1.0,
        epsilon_decay_duration=10000,
        target_network_frequency=1000,
        batch_size=32,
        buffer_size=10000,
        learning_starts=1000,
        train_frequency=4
    )
    
    with open(examples_dir / "dqn_config.py", "w") as f:
        f.write(f"""\"\"\"Example DQN configuration for CartPole.\"\"\"

from practice.dqn import DQNConfig

config = {repr(dqn_config)}
""")
    
    # REINFORCE example
    reinforce_config = ReinforceConfig(
        env_id="CartPole-v1",
        env_kwargs={"render_mode": "rgb_array"},
        episodes=1000,
        max_steps=500,
        gamma=0.99,
        eval_episodes=100,
        eval_seed=None,
        output_dir="results/reinforce/cartpole/",
        save_result=True,
        model_filename="reinforce.pth",
        repo_id="",
        learning_rate=0.003,
        use_baseline=True,
        baseline_decay=0.99,
        entropy_coef=0.01
    )
    
    with open(examples_dir / "reinforce_config.py", "w") as f:
        f.write(f"""\"\"\"Example REINFORCE configuration for CartPole.\"\"\"

from practice.reinforce import ReinforceConfig

config = {repr(reinforce_config)}
""")
    
    # Enhanced REINFORCE example
    enhanced_config = EnhancedReinforceConfig(
        env_id="MountainCar-v0",
        env_kwargs={"render_mode": "rgb_array"},
        episodes=1000,
        max_steps=200,
        gamma=0.99,
        eval_episodes=100,
        eval_seed=None,
        output_dir="results/enhanced_reinforce/mountain_car/",
        save_result=True,
        model_filename="enhanced_reinforce.pth",
        repo_id="",
        learning_rate=0.003,
        use_baseline=True,
        baseline_decay=0.99,
        entropy_coef=0.01,
        use_beta_scheduler=True,
        initial_beta=0.1,
        final_beta=0.001,
        beta_decay_duration=500,
        use_curiosity=True,
        curiosity_coef=0.1,
        use_reward_shaping=False
    )
    
    with open(examples_dir / "enhanced_reinforce_config.py", "w") as f:
        f.write(f"""\"\"\"Example Enhanced REINFORCE configuration for MountainCar.\"\"\"

from practice.enhanced_reinforce import EnhancedReinforceConfig

config = {repr(enhanced_config)}
""")
    
    print(f"Example configurations created in {examples_dir}")


def main() -> None:
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Unified CLI for training RL algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available algorithms: {', '.join(ALGORITHMS.keys())}

Examples:
  # Train Q-table on FrozenLake
  python practice/cli.py --algorithm qtable --config practice/configs/examples/qtable_config.py
  
  # Train DQN on CartPole  
  python practice/cli.py --algorithm dqn --config practice/configs/examples/dqn_config.py
  
  # Train REINFORCE on CartPole
  python practice/cli.py --algorithm reinforce --config practice/configs/examples/reinforce_config.py
  
  # Train Enhanced REINFORCE on MountainCar
  python practice/cli.py --algorithm enhanced_reinforce --config practice/configs/examples/enhanced_reinforce_config.py
  
  # Create example configurations
  python practice/cli.py --create-examples
"""
    )
    
    parser.add_argument(
        "--algorithm", "-a",
        choices=list(ALGORITHMS.keys()),
        help="Algorithm to train"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file (.json or .py)"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/",
        help="Directory for tensorboard logs (default: logs/)"
    )
    
    parser.add_argument(
        "--create-examples",
        action="store_true",
        help="Create example configuration files"
    )
    
    args = parser.parse_args()
    
    if args.create_examples:
        create_example_configs()
        return
    
    if not args.algorithm or not args.config:
        parser.error("Both --algorithm and --config are required (unless using --create-examples)")
    
    if args.algorithm not in ALGORITHMS:
        parser.error(f"Unknown algorithm: {args.algorithm}")
    
    if not Path(args.config).exists():
        parser.error(f"Config file not found: {args.config}")
    
    try:
        # Load configuration
        config = load_config_from_file(args.config, args.algorithm)
        
        # Create log directory
        log_dir = f"{args.log_dir}/{args.algorithm}/{Path(args.config).stem}"
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Train algorithm
        train_algorithm(config, args.algorithm, log_dir)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 