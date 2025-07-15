# Vanilla REINFORCE Implementation

This directory contains the implementation of the vanilla REINFORCE algorithm for reinforcement learning.

## Overview

REINFORCE is a policy gradient method that directly optimizes the policy by using the gradient of the expected return. This implementation follows the structure established in the practice exercises, providing a clean and extensible framework for policy gradient learning.

## Files

- `reinforce_exercise.py`: Main implementation containing:
  - `ReinforceConfig`: Configuration dataclass for the algorithm
  - `EpisodeBuffer`: Simple buffer for storing episode data
  - `ReinforceTrainer`: Main trainer class implementing the training loop
  - `_ReinforcePod`: Internal class handling the core REINFORCE logic

- `config_mountain_car.py`: Configuration file for MountainCar-v0 environment
- `config_cartpole.py`: Configuration file for CartPole-v1 environment
- `tests/test_reinforce_train.py`: Comprehensive test suite

## Key Features

### Algorithm Features
- **Vanilla REINFORCE**: Pure policy gradient implementation without baseline
- **Entropy Regularization**: Configurable entropy bonus for exploration
- **Gradient Accumulation**: Support for accumulating gradients over multiple episodes

### Implementation Features
- **Modular Design**: Clean separation of concerns with trainer, pod, and config classes
- **Type Safety**: Full type hints and mypy compliance
- **Testing**: Comprehensive test coverage
- **Logging**: TensorBoard integration for training metrics
- **Checkpointing**: Save and load model states

## Usage

### Basic Training

```python
from practice.exercise3_reinforce.config_mountain_car import (
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
# Train the model on MountainCar-v0
python practice/cli.py --config practice/exercise3_reinforce/config_mountain_car.py

# Train the model on CartPole-v1
python practice/cli.py --config practice/exercise3_reinforce/config_cartpole.py

# Play with trained model (when implemented)
python practice/cli.py --config practice/exercise3_reinforce/config_mountain_car.py --mode play

# Alternative using the utils CLI
python -m practice.utils.cli_utils train practice/exercise3_reinforce/config_mountain_car.py
python -m practice.utils.cli_utils play practice/exercise3_reinforce/config_mountain_car.py --checkpoint_pathname results/exercise3_reinforce/mountain_car/reinforce.pth
```

### Configuration

The algorithm can be configured through the `ReinforceConfig` dataclass:

```python
config = ReinforceConfig(
    device=torch.device("cpu"),
    episode=1000,           # Number of training episodes
    learning_rate=1e-3,     # Learning rate for optimizer
    gamma=0.999,            # Discount factor
    grad_acc=1,             # Gradient accumulation steps
    entropy_coef=0.01,      # Entropy regularization coefficient
    # ... other config parameters
)
```

## Testing

Run the test suite:

```bash
python -m pytest practice/exercise3_reinforce/tests/test_reinforce_train.py -v
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

1. **Environment Setup**: Initialize environment and policy network
2. **Episode Collection**: Run episodes and collect trajectories
3. **Return Calculation**: Compute discounted returns for each step
4. **Policy Update**: Compute policy gradient and update parameters (no baseline)
5. **Logging**: Record training metrics and performance

### Key Components

- **ReinforceTrainer**: Orchestrates the training process
- **_ReinforcePod**: Handles action sampling and policy updates
- **EpisodeBuffer**: Stores episode data (rewards, log_probs, entropies)
- **Reinforce1DNet**: Policy network (from hands_on implementation)

## Environments

### MountainCar-v0
- **Observation Space**: 2D (position, velocity)
- **Action Space**: 3 discrete actions (left, no-op, right)
- **Goal**: Reach the flag on the right side of the mountain
- **Configuration**: `config_mountain_car.py`

### CartPole-v1
- **Observation Space**: 4D (position, velocity, angle, angular velocity)
- **Action Space**: 2 discrete actions (left, right)
- **Goal**: Balance the pole on the cart
- **Configuration**: `config_cartpole.py`

## Comparison with Enhanced Version

This vanilla implementation differs from the enhanced version in `practice/exercise4_curiosity/` by:

- **No Curiosity Rewards**: Pure environment rewards only
- **No Baseline**: Pure vanilla REINFORCE without variance reduction
- **Simpler Configuration**: No reward configuration system
- **Focused Scope**: Single algorithm implementation
- **Educational Purpose**: Clear, straightforward REINFORCE implementation

## Dependencies

- PyTorch: Neural network implementation
- Gymnasium: Environment interface
- NumPy: Numerical computations
- TensorBoard: Logging and visualization
- tqdm: Progress bars

## Environment Support

Currently supports 1D observation environments (like MountainCar-v0 and CartPole-v1). The implementation can be extended to support:
- Different observation spaces
- Different action spaces
- Custom environments

## Future Enhancements

Potential improvements:
- Actor-Critic variants (with baseline)
- Natural policy gradients
- Proximal Policy Optimization (PPO)
- Multi-environment training
- Distributed training support
