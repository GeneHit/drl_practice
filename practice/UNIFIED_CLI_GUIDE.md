# Unified CLI Quick Reference

## ✨ **New Unified Entry Point**

Use `python practice/enhanced_cli.py` for all algorithms with automatic algorithm detection!

## 🎯 **Command Format**

```bash
python practice/enhanced_cli.py --config <config_file> --mode <mode> [options]
```

## 📋 **Available Modes**

- `train` - Train model, then optionally play and generate video
- `train --skip_play` - Train model only, skip playing
- `play` - Only play game and generate video (no training)
- `push_to_hub --username <name>` - Push model to hub, optionally play
- `push_to_hub --username <name> --skip_play` - Push to hub only

## 🚀 **Working Examples**

### Q-Table Training

```bash
# Train Q-table (auto-detects from config)
python practice/enhanced_cli.py \
    --config practice/exercise1_qtable/qtable_config.py \
    --mode train

# Train only (skip playing)
python practice/enhanced_cli.py \
    --config practice/exercise1_qtable/qtable_config.py \
    --mode train --skip_play

# Play with trained model
python practice/enhanced_cli.py \
    --config practice/exercise1_qtable/qtable_config.py \
    --mode play
```

### DQN Training

```bash
# Train DQN with 1D observations (LunarLander)
python practice/enhanced_cli.py \
    --config practice/exercise2_dqn/obs_1d_config.py \
    --mode train --skip_play

# Train DQN with 2D observations (images)
python practice/enhanced_cli.py \
    --config practice/exercise2_dqn/obs_2d_config.py \
    --mode train

# Quick test with minimal config (CartPole)
python practice/enhanced_cli.py \
    --config practice/exercise2_dqn/test_config.py \
    --mode train --skip_play
```

### Legacy JSON Support

```bash
# Use existing JSON configurations
python practice/enhanced_cli.py \
    --config practice/exercise2_dqn/obs_1d_config.json \
    --mode train --skip_play
```

### Hub Integration

```bash
# Push to Hugging Face Hub (placeholder)
python practice/enhanced_cli.py \
    --config practice/exercise2_dqn/test_config.py \
    --mode push_to_hub --username your_username --skip_play
```

## 🔍 **Algorithm Auto-Detection**

The CLI automatically detects the algorithm from:
- **File path**: `qtable` → Q-Table, `dqn` → DQN
- **Config class**: `QTableConfig` → Q-Table, `DQNConfig` → DQN
- **JSON parameters**: Checks for algorithm-specific parameters

### Force Algorithm (if needed)

```bash
python practice/enhanced_cli.py \
    --config my_config.py \
    --algorithm dqn \
    --mode train
```

## 📁 **Output Structure**

All training runs automatically save:
```
results/
├── exercise1_qtable/frozen_lake/
│   ├── qtable.pkl              # Trained Q-table
│   ├── params.json             # Configuration
│   ├── train_result.json       # Training stats
│   └── replay.mp4              # Gameplay video
│
├── exercise2_dqn/lunar_1d/
│   ├── dqn.pth                 # Trained DQN model
│   ├── params.json             # Configuration
│   ├── train_result.json       # Training stats
│   └── replay.mp4              # Gameplay video
│
└── test_dqn/
    ├── dqn.pth
    ├── params.json
    ├── train_result.json
    └── replay.mp4
```

## 🎮 **Example Training Output**

```bash
$ python practice/enhanced_cli.py --config practice/exercise1_qtable/qtable_config.py --mode train --skip_play

Auto-detected algorithm: QTABLE
Training QTABLE...
Using device: cpu
Training Q-table on FrozenLake-v1...
Training Q-table: 100%|████████| 1000/1000 [00:00<00:00, 1327.59it/s]
Training completed! Model saved to: results/exercise1_qtable/frozen_lake/qtable.pkl
```

## 📊 **Available Algorithms**

- ✅ **Q-Table**: Tabular Q-learning for discrete environments
- ✅ **DQN**: Deep Q-Network with multi-environment support
- 🔄 **REINFORCE**: Coming soon
- 🔄 **Enhanced REINFORCE**: Coming soon

## 🆘 **Help & Options**

```bash
python practice/enhanced_cli.py --help
```

Shows all available options and example commands.

## 🔄 **Migration from Module Commands**

### Old Way
```bash
python -m practice.exercise1_qtable.qtable_exercise --config ...
python -m practice.exercise2_dqn.dqn_exercise --config ...
```

### New Way (Unified)
```bash
python practice/enhanced_cli.py --config ...
```

Both approaches still work, but the unified CLI is recommended for consistency and ease of use! 