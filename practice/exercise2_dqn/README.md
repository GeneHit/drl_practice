# DQN Implementation

Deep Q-Network (DQN) is a value-based reinforcement learning algorithm that combines Q-learning with deep neural networks.

## File Structure:
- `dqn_exercise.py`: Main `EXERCISE` containing:
    - `DQNConfig`: Configuration dataclass for the algorithm
    - `QNet1D`: Q-network for 1D observations
    - `QNet2D`: Q-network for 2D/image observations
    - `DQNTrainer`: Main trainer class implementing the training loop
    - `_DQNPod`: Internal class handling the core DQN logic

- `config_lunar_1d.py`: Configuration file for LunarLander-v3 with 1D observations
- `config_lunar_2d.py`: Configuration file for LunarLander-v3 with 2D/image observations. `You Should Find A Better Config`.
- `tests/test_dqn_train.py`: Comprehensive test suite

`You Need Find A Better Config for LunarLander-v3 2D`.

## Command
Train the Q-learning agent using the CLI:
```bash
# Train on LunarLander-v3 1D environment
python practice/cli.py --config practice/exercise2_dqn/config_lunar_1d.py

# Train on LunarLander-v3 2D  environment
python practice/cli.py --config practice/exercise2_dqn/config_lunar_2d.py
```

Test the trained agent:
```bash
# Play with trained agent and generate video
python practice/cli.py --config practice/exercise2_dqn/config_lunar_1d.py --mode play
```

Push to hub
```bash
# generate video and push to hub
python practice/cli.py --config practice/exercise2_dqn/config_lunar_1d.py --push_to_hub --username myuser

# only push to hub
python practice/cli.py --config practice/exercise2_dqn/config_lunar_1d.py --push_to_hub --username myuser --skip_play
```
**Replace `myuser` with your HuggingFace account name.**

Run the comprehensive test suite:
```bash
# Run all tests
python -m pytest practice/exercise2_dqn/tests/ -v
```


## Key Features

Algorithm Features
- **Experience Replay**: Stores transitions in a replay buffer for stable learning
- **Target Network**: Separate target network for computing TD targets
- **Epsilon-Greedy Exploration**: Configurable exploration strategy
- **Multi-Environment Training**: Support for parallel environment training
- **Double DQN**: Can be extended to support Double DQN

Training Flow
1. **Environment Setup**: Initialize vector environments and Q-network
2. **Experience Collection**: Collect experiences using epsilon-greedy policy
3. **Replay Buffer**: Store transitions in replay buffer
4. **Training Updates**: Sample batches and update Q-network using TD loss
5. **Target Network Updates**: Periodically sync target network
6. **Logging**: Record training metrics and performance

Key Components
- **DQNTrainer**: Orchestrates the training process
- **_DQNPod**: Handles action selection and Q-network updates
- **QNet1D/QNet2D**: Q-networks for different observation types
- **ReplayBuffer**: Stores and samples experiences (from hands_on utilities)
- **LinearSchedule**: Epsilon decay schedule (from hands_on utilities)


## Environments

LunarLander-v3 (1D)
- **Observation Space**: 8D continuous (position, velocity, angle, etc.)
- **Action Space**: 4 discrete actions (do nothing, fire left, fire main, fire right)
- **Goal**: Land the lunar module safely
- **Configuration**: `config_lunar_1d.py`

LunarLander-v3 (2D/Image)
- **Observation Space**: 4x84x84 image stack (rendered environment)
- **Action Space**: 4 discrete actions (same as 1D)
- **Goal**: Land the lunar module safely using visual input
- **Configuration**: `config_lunar_2d.py`


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
