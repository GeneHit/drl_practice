# Practice RL Algorithms

A well-architected package for training and experimenting with reinforcement learning algorithms.

## Features

### 🎯 **Unified Configuration System**
- **Python-based configurations**: Easy-to-read, type-safe configuration classes
- **Legacy JSON support**: Backward compatibility with existing JSON configs
- **Automatic validation**: Built-in parameter validation with helpful error messages
- **Extensible**: Easy to add new algorithms and parameters

### 🚀 **Unified CLI Interface**
- **Single entry point**: `python practice/cli.py` for all algorithms
- **Algorithm registry**: Automatic discovery and registration of algorithms
- **Example generation**: Auto-create example configurations
- **Clear error messages**: Helpful feedback for configuration issues

### 🧠 **Implemented Algorithms**

#### 1. **Q-Table** (`qtable`)
- Classic tabular Q-learning for discrete environments
- Epsilon-greedy exploration with exponential decay
- Perfect for small discrete state/action spaces like FrozenLake

#### 2. **Deep Q-Network (DQN)** (`dqn`) 
- Neural network-based Q-learning
- Experience replay buffer
- Target network for stability
- Suitable for environments like CartPole

#### 3. **REINFORCE** (`reinforce`)
- Policy gradient method
- Optional baseline for variance reduction
- Entropy regularization
- Monte Carlo returns

#### 4. **Enhanced REINFORCE** (`enhanced_reinforce`)
- **Advanced baseline**: Exponential moving average baseline
- **Beta scheduling**: Adaptive entropy coefficient
- **Curiosity-driven learning**: Intrinsic motivation module
- **Reward shaping**: Additional reward signals
- Perfect for challenging environments like MountainCar

### 🏗️ **Architecture Highlights**

- **Type safety**: Full type annotations and mypy compliance
- **Modular design**: Easy to extend with new algorithms
- **Clean abstractions**: Base classes for agents, trainers, and configs
- **Automatic logging**: TensorBoard integration
- **Device agnostic**: Automatic CPU/GPU detection

## Quick Start

### 1. Create Example Configurations
```bash
python practice/cli.py --create-examples
```

### 2. Train Algorithms

```bash
# Train Q-table on FrozenLake
python practice/cli.py --algorithm qtable --config practice/configs/examples/qtable_config.py

# Train DQN on CartPole
python practice/cli.py --algorithm dqn --config practice/configs/examples/dqn_config.py

# Train REINFORCE on CartPole
python practice/cli.py --algorithm reinforce --config practice/configs/examples/reinforce_config.py

# Train Enhanced REINFORCE on MountainCar
python practice/cli.py --algorithm enhanced_reinforce --config practice/configs/examples/enhanced_reinforce_config.py
```

### 3. Monitor Training
```bash
tensorboard --logdir logs/
```

## Configuration Examples

### Python Configuration (Recommended)
```python
from practice.qtable import QTableConfig

config = QTableConfig(
    env_id="FrozenLake-v1",
    env_kwargs={"map_name": "4x4", "is_slippery": False},
    episodes=1000,
    max_steps=99,
    gamma=0.95,
    learning_rate=0.7,
    min_epsilon=0.05,
    max_epsilon=1.0,
    decay_rate=0.0005,
    output_dir="results/qtable/frozen_lake/"
)
```

### Enhanced REINFORCE with Curiosity
```python
from practice.enhanced_reinforce import EnhancedReinforceConfig

config = EnhancedReinforceConfig(
    env_id="MountainCar-v0",
    env_kwargs={"render_mode": "rgb_array"},
    episodes=1000,
    max_steps=200,
    gamma=0.99,
    learning_rate=0.003,
    use_baseline=True,
    use_curiosity=True,
    curiosity_coef=0.1,
    use_beta_scheduler=True,
    initial_beta=0.1,
    final_beta=0.001
)
```

## Adding New Algorithms

### 1. Create Algorithm Module
```python
# practice/my_algorithm.py
from practice.base import BaseConfig, AgentBase, TrainerBase

@dataclass
class MyAlgorithmConfig(BaseConfig):
    # Algorithm-specific parameters
    my_param: float
    
    def validate(self) -> None:
        # Validation logic
        pass

class MyAgent(AgentBase):
    # Implementation
    pass

class MyTrainer(TrainerBase):
    # Implementation  
    pass
```

### 2. Register in CLI
```python
# practice/cli.py
ALGORITHMS["my_algorithm"] = {
    "config_class": MyAlgorithmConfig,
    "trainer_class": MyTrainer,
    "requires_discrete_obs": False,
    "requires_discrete_action": True,
}
```

## Project Structure

```
practice/
├── __init__.py              # Package initialization
├── README.md               # This file
├── cli.py                  # Unified CLI interface
├── base.py                 # Base classes and interfaces
├── qtable.py              # Q-table implementation
├── dqn.py                 # DQN implementation  
├── reinforce.py           # REINFORCE implementation
├── enhanced_reinforce.py  # Enhanced REINFORCE with advanced features
├── utils/
│   ├── __init__.py
│   └── schedules.py       # Scheduling utilities
└── configs/
    └── examples/          # Auto-generated example configs
        ├── qtable_config.py
        ├── dqn_config.py
        ├── reinforce_config.py
        └── enhanced_reinforce_config.py
```

## Algorithm Comparison

| Algorithm | Type | Exploration | Experience Replay | Target Network | Baseline | Curiosity |
|-----------|------|-------------|-------------------|---------------|----------|-----------|
| Q-Table | Value-based | ε-greedy | ❌ | ❌ | ❌ | ❌ |
| DQN | Value-based | ε-greedy | ✅ | ✅ | ❌ | ❌ |
| REINFORCE | Policy-based | Stochastic policy | ❌ | ❌ | Optional | ❌ |
| Enhanced REINFORCE | Policy-based | Stochastic policy | ❌ | ❌ | ✅ | ✅ |

## Environment Compatibility

- **Q-Table**: Discrete observation & action spaces (e.g., FrozenLake, Taxi)
- **DQN**: Continuous observation, discrete action spaces (e.g., CartPole, Atari)
- **REINFORCE**: Continuous observation, discrete action spaces (e.g., CartPole)
- **Enhanced REINFORCE**: Continuous observation, discrete action spaces (e.g., MountainCar)

## Future Extensions

The architecture is designed to easily accommodate:
- **A2C/A3C**: Actor-Critic methods
- **PPO**: Proximal Policy Optimization
- **DDPG/TD3**: Continuous action algorithms
- **SAC**: Soft Actor-Critic
- **Custom environments**: Easy environment integration
- **Multi-agent settings**: Extend base classes for multi-agent scenarios 