# Enhanced Practice RL Exercises

This directory contains enhanced implementations of reinforcement learning algorithms with improved architecture, unified CLI, and comprehensive mode support.

## 🎯 **Key Improvements**

### 1. **Enhanced Trainer Architecture**
- **Config & Network Initialization**: Trainers now take config and network in `__init__`
- **Automatic Result Saving**: Training results and parameters saved to JSON automatically
- **Unified Base Classes**: Consistent interface across all algorithms

### 2. **Multi-Mode CLI Support**
Each exercise supports multiple execution modes:
- `--mode train`: Train model, then optionally play and generate video
- `--mode train --skip_play`: Train model only, skip playing
- `--mode push_to_hub --username <name>`: Push model to hub, optionally play
- `--mode push_to_hub --username <name> --skip_play`: Push to hub only
- `--mode play`: Only play game and generate video (no training)

### 3. **Unified CLI Entry Point**
- **Single Command**: Use `python practice/enhanced_cli.py` for all algorithms
- **Auto-Detection**: Automatically detects algorithm from configuration file
- **Backward Compatible**: Supports both Python and JSON configurations

### 4. **Multi-Environment Support (DQN)**
- **Vectorized Training**: Uses multiple environments for faster training
- **1D/2D Observations**: Automatically detects and handles both vector and image observations
- **Configurable**: Supports both multiprocessing and sync vectorization

### 5. **Comprehensive Configuration System**
- **Python Configurations**: Type-safe configuration classes
- **JSON Compatibility**: Backward compatible with existing JSON configs
- **Automatic Validation**: Built-in parameter validation with helpful error messages

## 📁 **Project Structure**

```
practice/
├── base.py                     # Enhanced base classes with config/network support
├── enhanced_cli.py             # Multi-mode CLI system
├── utils/
│   └── schedules.py           # Schedule implementations
├── exercise1_qtable/
│   ├── qtable_config.py       # Q-table configuration
│   └── qtable_exercise.py     # Q-table implementation with enhanced CLI
├── exercise2_dqn/
│   ├── dqn_config.py          # DQN configuration
│   ├── dqn_exercise.py        # Enhanced DQN with multi-env support
│   ├── obs_1d_config.py       # 1D observation config
│   ├── obs_2d_config.py       # 2D observation config
│   ├── obs_1d_config.json     # Legacy JSON config (1D)
│   ├── obs_2d_config.json     # Legacy JSON config (2D)
│   └── test_config.py         # Test configuration
├── exercise3_reinforce/        # [Future] REINFORCE exercise
└── exercise4_enhanced_reinforce/ # [Future] Enhanced REINFORCE exercise
```

## 🚀 **Usage Examples**

### Unified CLI (Recommended)

```bash
# Train Q-table on FrozenLake
python practice/enhanced_cli.py \
    --config practice/exercise1_qtable/qtable_config.py \
    --mode train

# Train DQN with 1D observations (skip playing)
python practice/enhanced_cli.py \
    --config practice/exercise2_dqn/obs_1d_config.py \
    --mode train --skip_play

# Train DQN with 2D observations
python practice/enhanced_cli.py \
    --config practice/exercise2_dqn/obs_2d_config.py \
    --mode train

# Quick test with minimal config
python practice/enhanced_cli.py \
    --config practice/exercise2_dqn/test_config.py \
    --mode train --skip_play

# Play with trained model
python practice/enhanced_cli.py \
    --config practice/exercise1_qtable/qtable_config.py \
    --mode play

# Push to Hugging Face Hub
python practice/enhanced_cli.py \
    --config practice/exercise2_dqn/obs_1d_config.py \
    --mode push_to_hub --username your_username

# Legacy JSON config support
python practice/enhanced_cli.py \
    --config practice/exercise2_dqn/obs_1d_config.json \
    --mode train --skip_play

# Force specific algorithm (if auto-detection fails)
python practice/enhanced_cli.py \
    --config my_custom_config.py \
    --algorithm dqn --mode train
```

### Alternative: Module-Based CLI

```bash
# Q-Table
python -m practice.exercise1_qtable.qtable_exercise \
    --config practice/exercise1_qtable/qtable_config.py \
    --mode train --skip_play

# DQN
python -m practice.exercise2_dqn.dqn_exercise \
    --config practice/exercise2_dqn/test_config.py \
    --mode train --skip_play
```

## 🧠 **Algorithm Features**

### Q-Table
- **Discrete environments**: FrozenLake, Taxi, etc.
- **Epsilon-greedy exploration**: Exponential decay schedule
- **Tabular Q-learning**: Classic reinforcement learning

### DQN (Deep Q-Network)
- **Multi-environment training**: 2-6 parallel environments
- **1D observations**: Vector inputs (LunarLander, CartPole)
- **2D observations**: Image inputs with CNN architecture
- **Experience replay**: Efficient memory usage
- **Target network**: Stable training
- **Configurable architecture**: Hidden dimensions, layers, etc.

## 🔧 **Configuration System**

### Python Configuration (Recommended)
```python
from practice.exercise2_dqn.dqn_config import DQNConfig

config = DQNConfig(
    env_id="CartPole-v1",
    env_kwargs={},
    timesteps=10000,
    learning_rate=1e-4,
    num_envs=4,
    output_dir="results/my_experiment/"
)
```

### Automatic Result Saving
All training runs automatically save:
- **`params.json`**: Complete configuration parameters
- **`train_result.json`**: Training statistics (rewards, losses, etc.)
- **`model.pth`**: Trained model weights
- **`replay.mp4`**: Gameplay video (if not skipped)

## 📊 **Training Results**

### Saved Statistics
- **Episode rewards**: Per-episode cumulative rewards
- **Episode lengths**: Number of steps per episode
- **Training losses**: Loss values during training
- **Exploration parameters**: Epsilon values, schedules
- **Custom metrics**: Algorithm-specific statistics

### TensorBoard Integration
```bash
# Monitor training in real-time
tensorboard --logdir logs/

# View specific algorithm
tensorboard --logdir logs/dqn/
```

## 🎮 **Multi-Environment Details (DQN)**

### Configuration
```python
config = DQNConfig(
    num_envs=6,                    # Number of parallel environments
    use_multi_processing=True,     # Use AsyncVectorEnv vs SyncVectorEnv
    timesteps=200000,              # Total training steps
    batch_size=64,                 # Replay buffer batch size
    train_interval=1,              # Update frequency
    target_update_interval=250,    # Target network sync frequency
)
```

### Observation Types
- **1D (Vector)**: Direct state representation
- **2D (Image)**: CNN-based processing with frame stacking

### Automatic Network Selection
The system automatically chooses the appropriate network architecture:
- **QNet1D**: For vector observations
- **QNet2D**: For image observations (CNN-based)

## 🔍 **Backward Compatibility**

### JSON Configuration Support
All exercises support legacy JSON configurations:
```bash
python -m practice.exercise2_dqn.dqn_exercise \
    --config hands_on/exercise2_dqn/obs_1d_config.json \
    --mode train
```

### Original Functionality
All original training capabilities are preserved while adding new features.

## 🚦 **Error Handling & Validation**

### Configuration Validation
```python
# Automatic validation on creation
config = DQNConfig(
    learning_rate=2.0,  # Error: learning_rate must be in (0, 1]
    batch_size=-1,      # Error: batch_size must be positive
)
```

### Runtime Checks
- Environment compatibility verification
- Network architecture validation
- Model loading error handling

## 🎯 **Best Practices**

### 1. **Development Workflow**
```bash
# 1. Test with minimal config
python -m practice.exercise2_dqn.dqn_exercise \
    --config practice/exercise2_dqn/test_config.py \
    --mode train --skip_play

# 2. Train full model
python -m practice.exercise2_dqn.dqn_exercise \
    --config practice/exercise2_dqn/obs_1d_config.py \
    --mode train

# 3. Test trained model
python -m practice.exercise2_dqn.dqn_exercise \
    --config practice/exercise2_dqn/obs_1d_config.py \
    --mode play
```

### 2. **Configuration Management**
- Use Python configs for new experiments
- Start with provided examples and modify
- Keep JSON configs for reproducibility

### 3. **Resource Management**
- Use `--skip_play` for faster iteration
- Reduce `num_envs` for memory-constrained systems
- Use `test_config.py` for quick validation

## 🔄 **Migration from Original**

### From `hands_on/exercise2_dqn/`
```bash
# Old way
cd hands_on/exercise2_dqn
python dqn_cli.py config.json

# New way
cd /path/to/drl_practice
python -m practice.exercise2_dqn.dqn_exercise \
    --config practice/exercise2_dqn/obs_1d_config.py \
    --mode train
```

### Key Differences
1. **Module execution**: Use `python -m` from project root
2. **Enhanced modes**: More execution options
3. **Automatic saving**: Results saved by default
4. **Multi-environment**: Better training efficiency
5. **Type safety**: Configuration validation

## 🎊 **Future Extensions**

The architecture is designed for easy extension:
- **New algorithms**: Implement base classes
- **Custom environments**: Add environment wrappers
- **Advanced features**: Extend configuration classes
- **Integration**: Connect with existing ML pipelines

## 📝 **Example Output**

```
Training DQN...
Using device: cpu
Training DQN with 6 environments...
Training DQN: 100%|██████████| 200000/200000 [02:15<00:00, 1480.32it/s]
Training completed! Model saved to: results/exercise2_dqn/lunar_1d/dqn.pth

Files saved:
- results/exercise2_dqn/lunar_1d/dqn.pth (model weights)
- results/exercise2_dqn/lunar_1d/params.json (configuration)
- results/exercise2_dqn/lunar_1d/train_result.json (training statistics)
- results/exercise2_dqn/lunar_1d/replay.mp4 (gameplay video)
```

This enhanced system provides a professional-grade foundation for reinforcement learning research and experimentation while maintaining full backward compatibility with existing code.
