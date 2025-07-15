# Exercise 1: Q-Learning Implementation

This directory contains a complete implementation of the Q-learning algorithm adapted to the practice framework. The implementation follows the same structure as other exercises in the practice directory, providing a unified interface for training, evaluation, and deployment.

## Overview

Q-learning is a model-free reinforcement learning algorithm that learns the quality of actions, telling an agent what action to take under what circumstances. It does not require a model of the environment and can handle problems with stochastic transitions and rewards.

## Key Features

- **Tabular Q-learning**: Uses a Q-table to store state-action values
- **Epsilon-greedy exploration**: Balances exploration and exploitation
- **Exponential epsilon decay**: Reduces exploration over time
- **Discrete environment support**: Works with environments having discrete observation and action spaces
- **Checkpoint support**: Can resume training from saved Q-tables
- **Comprehensive evaluation**: Includes performance metrics and video recording
- **Framework integration**: Fully integrated with the practice framework

## Algorithm Components

### Q-Table
- Stores Q-values for all state-action pairs
- Updated using the Bellman equation: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
- Saved as pickle files for persistence

### Epsilon-Greedy Policy
- **Exploration**: Random action with probability ε
- **Exploitation**: Greedy action (argmax Q(s,a)) with probability 1-ε
- **Decay**: ε decreases exponentially over episodes

### Training Process
1. Initialize Q-table with zeros
2. For each episode:
   - Reset environment
   - For each step:
     - Choose action using epsilon-greedy policy
     - Execute action and observe reward and next state
     - Update Q-table using Bellman equation
     - Update state
   - Decay epsilon

## Usage

### Training

Train the Q-learning agent using the CLI:

```bash
# Train on Taxi environment
python practice/cli.py --config practice/exercise1_q/config_taxi.py

# Train on FrozenLake environment
python practice/cli.py --config practice/exercise1_q/config_frozen_lake.py
```

### Playing

Test the trained agent:

```bash
# Play with trained agent
python practice/cli.py --config practice/exercise1_q/config_taxi.py --mode play
```

### Evaluation

Evaluate the trained agent:

```bash
# Evaluate performance
python practice/cli.py --config practice/exercise1_q/config_taxi.py --mode eval
```

## Configuration

### Taxi Configuration (`config_taxi.py`)

Based on the Taxi-v3 environment with the following parameters:
- **Episodes**: 25,000 training episodes
- **Max Steps**: 99 steps per episode
- **Learning Rate**: 0.7
- **Gamma**: 0.95 (discount factor)
- **Epsilon**: 1.0 → 0.05 (exploration decay)
- **Decay Rate**: 0.0005 (exponential decay)

### FrozenLake Configuration (`config_frozen_lake.py`)

Based on the FrozenLake-v1 environment with the following parameters:
- **Episodes**: 10,000 training episodes
- **Max Steps**: 99 steps per episode
- **Learning Rate**: 0.7
- **Gamma**: 0.95 (discount factor)
- **Epsilon**: 1.0 → 0.05 (exploration decay)
- **Decay Rate**: 0.0005 (exponential decay)
- **Environment**: 4x4 grid, non-slippery

## File Structure

```
practice/exercise1_q/
├── q_table_exercise.py      # Main Q-learning implementation
├── config_taxi.py           # Taxi environment configuration
├── config_frozen_lake.py    # FrozenLake environment configuration
├── tests/
│   └── test_q_table_train.py # Comprehensive test suite
└── README.md                # This file
```

## Implementation Details

### QTableConfig
Dataclass extending `BaseConfig` with Q-learning specific parameters:
- `episodes`: Number of training episodes
- `max_steps`: Maximum steps per episode
- `learning_rate`: Learning rate (α)
- `gamma`: Discount factor (γ)
- `min_epsilon`: Minimum exploration rate
- `max_epsilon`: Maximum exploration rate
- `decay_rate`: Exponential decay rate for epsilon

### QTable Class
Agent class implementing the `AgentBase` interface:
- `action(state)`: Returns greedy action for given state
- `only_save_model(pathname)`: Saves Q-table to pickle file
- `load_from_checkpoint(pathname, device)`: Loads Q-table from pickle file

### QTableTrainer Class
Trainer class implementing the `TrainerBase` interface:
- `train()`: Main training loop with epsilon-greedy exploration
- Logs training metrics (TD error, epsilon, episode rewards)
- Uses TensorBoard for visualization

### _QTablePod Class
Internal class handling training logic:
- `action(state, episode)`: Epsilon-greedy action selection
- `update(state, action, reward, next_state)`: Q-table update using Bellman equation
- `get_epsilon(episode)`: Current epsilon value

## Environment Support

The implementation supports discrete environments with:
- **Discrete observation space**: States represented as integers
- **Discrete action space**: Actions represented as integers
- **Episodic tasks**: Episodes have clear start and end

### Supported Environments
- **Taxi-v3**: 500 states, 6 actions (pickup/dropoff passengers)
- **FrozenLake-v1**: 16 states (4x4 grid), 4 actions (up/down/left/right)

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest practice/exercise1_q/tests/ -v

# Run specific test
python -m pytest practice/exercise1_q/tests/test_q_table_train.py::TestQTableTraining::test_q_table_train_basic_flow -v
```

### Test Coverage
- Basic training flow without file operations
- Training with file saving and output validation
- Training from checkpoint (resuming from saved Q-table)
- CLI integration testing
- Configuration validation
- Q-table agent functionality
- Exception handling and cleanup

## Comparison with Hands-On Version

### Similarities
- Same Q-learning algorithm and Bellman equation
- Same epsilon-greedy exploration strategy
- Same exponential epsilon decay
- Same environment support (Taxi, FrozenLake)

### Differences
- **Framework Integration**: Uses practice framework base classes
- **Configuration**: Python-based config instead of JSON
- **Context Management**: Uses `ContextBase` for environment and Q-table management
- **Type Safety**: Full type annotations and mypy compatibility
- **Testing**: Comprehensive test suite with pytest
- **CLI Integration**: Unified CLI interface with other exercises

## Performance Notes

- **Memory Usage**: O(|S| × |A|) for Q-table storage
- **Computation**: O(1) per step for Q-table updates
- **Convergence**: Guaranteed convergence to optimal policy under certain conditions
- **Scalability**: Limited to discrete environments with reasonable state/action space sizes

## Troubleshooting

### Common Issues

1. **Memory Error**: Q-table too large for environment
   - Solution: Use function approximation (DQN) for large state spaces

2. **Slow Convergence**: Agent not learning effectively
   - Solution: Adjust learning rate, epsilon decay, or increase episodes

3. **Poor Performance**: Low evaluation rewards
   - Solution: Tune hyperparameters or increase training episodes

4. **Environment Errors**: Discrete space requirements
   - Solution: Ensure environment has discrete observation and action spaces

### Debug Tips

- Monitor epsilon decay in TensorBoard
- Check Q-table convergence by examining value changes
- Verify environment reset and step functions
- Use smaller environments for debugging (e.g., FrozenLake 4x4)

## Future Enhancements

- **Multi-step Q-learning**: N-step returns for faster learning
- **Double Q-learning**: Reduce overestimation bias
- **Prioritized experience**: Focus on important transitions
- **Function approximation**: Support for continuous/large state spaces
- **Distributed training**: Parallel environment execution
