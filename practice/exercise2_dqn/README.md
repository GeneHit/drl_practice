# DQN Implementation

This directory contains the implementation of the Deep Q-Network (DQN) algorithm for reinforcement learning.

## Overview

DQN is a value-based reinforcement learning algorithm that combines Q-learning with deep neural networks. It uses experience replay and target networks to stabilize training. This implementation follows the structure established in the practice exercises, providing a clean and extensible framework for deep Q-learning.

## Files

- `dqn_exercise.py`: Main implementation containing:
  - `DQNConfig`: Configuration dataclass for the algorithm
  - `QNet1D`: Q-network for 1D observations
  - `QNet2D`: Q-network for 2D/image observations
  - `DQNTrainer`: Main trainer class implementing the training loop
  - `_DQNPod`: Internal class handling the core DQN logic

- `config_lunar_1d.py`: Configuration file for LunarLander-v3 with 1D observations
- `config_lunar_2d.py`: Configuration file for LunarLander-v3 with 2D/image observations
- `tests/test_dqn_train.py`: Comprehensive test suite

## Key Features

### Algorithm Features
- **Experience Replay**: Stores transitions in a replay buffer for stable learning
- **Target Network**: Separate target network for computing TD targets
- **Epsilon-Greedy Exploration**: Configurable exploration strategy
- **Multi-Environment Training**: Support for parallel environment training
- **Double DQN**: Can be extended to support Double DQN

### Implementation Features
- **Modular Design**: Clean separation of concerns with trainer, pod, and config classes
- **Type Safety**: Full type hints and mypy compliance
- **Testing**: Comprehensive test coverage
- **Logging**: TensorBoard integration for training metrics
- **Checkpointing**: Save and load model states
- **Vector Environments**: Support for both sync and async vector environments

## Usage

### Basic Training

```python
from practice.exercise2_dqn.config_lunar_1d import (
    generate_context,
    get_app_config,
)
from practice.utils.train_utils import train_and_evaluate_network

# Get configuration
config = get_app_config()

# Generate context
context = generate_context(config)

# Run training
train_and_evaluate_network(config=config, ctx=context)
```

### Running via CLI

```bash
# Train the model on LunarLander-v3 with 1D observations
python practice/cli.py --config practice/exercise2_dqn/config_lunar_1d.py

# Train the model on LunarLander-v3 with 2D/image observations
python practice/cli.py --config practice/exercise2_dqn/config_lunar_2d.py

# Play with trained model (when implemented)
python practice/cli.py --config practice/exercise2_dqn/config_lunar_1d.py --mode play

# Alternative using the utils CLI
python -m practice.utils.cli_utils train practice/exercise2_dqn/config_lunar_1d.py
python -m practice.utils.cli_utils play practice/exercise2_dqn/config_lunar_1d.py --checkpoint_pathname results/exercise2_dqn/lunar_1d/dqn.pth
```

### Configuration

The algorithm can be configured through the `DQNConfig` dataclass:

```python
config = DQNConfig(
    device=torch.device("cpu"),
    timesteps=200000,           # Number of training timesteps
    learning_rate=1e-4,         # Learning rate for optimizer
    gamma=0.99,                 # Discount factor
    start_epsilon=1.0,          # Initial exploration rate
    end_epsilon=0.01,           # Final exploration rate
    exploration_fraction=0.1,   # Fraction of training for exploration decay
    replay_buffer_capacity=120000,  # Size of replay buffer
    batch_size=64,              # Batch size for training
    train_interval=1,           # Training frequency
    target_update_interval=250, # Target network update frequency
    update_start_step=1000,     # When to start training
    env_config=EnvConfig(
        env_id="LunarLander-v3",
        vector_env_num=6,       # Number of parallel environments
        use_multi_processing=True,  # Use multiprocessing for environments
        record_eval_video=True,
    ),
    # ... other config parameters
)
```

### Environment Configuration

The environment setup is handled through the `EnvConfig` within the base configuration:

- **vector_env_num**: Number of parallel environments for training (required for DQN)
- **use_multi_processing**: Whether to use AsyncVectorEnv (True) or SyncVectorEnv (False)
- **use_image**: Enable image-based observations (2D)
- **training_render_mode**: Render mode for training environments
- **image_shape**: Shape for image observations (height, width)
- **frame_stack**: Number of frames to stack for temporal information

## Testing

Run the test suite:

```bash
python -m pytest practice/exercise2_dqn/tests/test_dqn_train.py -v
```

Test coverage includes:
- Basic training flow
- File saving and loading
- Checkpoint resumption
- CLI integration
- Model validation
- Configuration validation

## Architecture

### Training Flow

1. **Environment Setup**: Initialize vector environments and Q-network
2. **Experience Collection**: Collect experiences using epsilon-greedy policy
3. **Replay Buffer**: Store transitions in replay buffer
4. **Training Updates**: Sample batches and update Q-network using TD loss
5. **Target Network Updates**: Periodically sync target network
6. **Logging**: Record training metrics and performance

### Key Components

- **DQNTrainer**: Orchestrates the training process
- **_DQNPod**: Handles action selection and Q-network updates
- **QNet1D/QNet2D**: Q-networks for different observation types
- **ReplayBuffer**: Stores and samples experiences (from hands_on utilities)
- **LinearSchedule**: Epsilon decay schedule (from hands_on utilities)

## Environments

### LunarLander-v3 (1D)
- **Observation Space**: 8D continuous (position, velocity, angle, etc.)
- **Action Space**: 4 discrete actions (do nothing, fire left, fire main, fire right)
- **Goal**: Land the lunar module safely
- **Configuration**: `config_lunar_1d.py`

### LunarLander-v3 (2D/Image)
- **Observation Space**: 84x84x4 image stack (rendered environment)
- **Action Space**: 4 discrete actions (same as 1D)
- **Goal**: Land the lunar module safely using visual input
- **Configuration**: `config_lunar_2d.py`

## Comparison with Hands-On Version

This practice implementation differs from the hands-on version by:

- **Unified Framework**: Uses the practice framework structure with `BaseConfig` and `ContextBase`
- **Environment Management**: Environments are created through `env_config` and provided via context, not created in trainer
- **Configuration Structure**: Uses `EnvConfig` for environment parameters instead of trainer-specific fields
- **Simplified Interface**: Clean config and context pattern with automatic environment setup
- **Better Testing**: Comprehensive test coverage with proper environment handling
- **Type Safety**: Full mypy compliance with proper type annotations
- **Modular Design**: Clear separation of concerns between configuration, context, and training logic

## Dependencies

- PyTorch: Neural network implementation
- Gymnasium: Environment interface
- NumPy: Numerical computations
- TensorBoard: Logging and visualization
- tqdm: Progress bars

## Environment Support

Currently supports:
- **1D Observations**: Continuous state vectors
- **2D Observations**: Image-based observations with frame stacking
- **Discrete Actions**: Finite action spaces
- **Vector Environments**: Parallel environment training

## Future Enhancements

Potential improvements:
- Double DQN implementation
- Dueling DQN architecture
- Prioritized Experience Replay
- Rainbow DQN (combining multiple improvements)
- Continuous action spaces (DDPG/TD3)
- Distributed training support
