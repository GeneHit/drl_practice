# DQN Exercise

Deep Q-Network (DQN) is a value-based reinforcement learning algorithm that combines Q-learning with deep neural networks.

## File Structure:
- `dqn_exercise.py`: Main `EXERCISE`.
- `config_lunar_1d.py`: Configuration file for LunarLander-v3 with 1D observations.
- `config_lunar_2d.py`: Configuration file for LunarLander-v3 with 2D/image observations.
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
python practice/cli.py --config practice/exercise2_dqn/config_lunar_1d.py --mode push_to_hub --username myuser

# only push to hub
python practice/cli.py --config practice/exercise2_dqn/config_lunar_1d.py --mode push_to_hub --username myuser --skip_play
```
**Replace `myuser` with your HuggingFace account name.**

Run the comprehensive test suite:
```bash
# Run all tests
python -m pytest practice/exercise2_dqn/tests/ -v
```


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


Supporting **Vector Environments**: Parallel environment training.

## Future Enhancements

Potential improvements:
- Double DQN implementation
- Dueling DQN architecture
- Prioritized Experience Replay
- Rainbow DQN (combining multiple improvements)
- Continuous action spaces (DDPG/TD3)
- Distributed training support
