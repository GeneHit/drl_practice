# Double DQN Exercise

Double Deep Q-Network (Double DQN):
- Double DQN addresses the overestimation bias in standard DQN
- decouples action selection from action evaluation to provide more stable and accurate Q-value estimates.

| Aspect | Standard DQN | Double DQN |
|--------|--------------|------------|
| Action Selection | Target Network | Current Network |
| Action Evaluation | Target Network | Target Network |
| Bias Reduction | ❌ Overestimation bias | ✅ Reduced overestimation |
| Stability | Moderate | Improved |
| Performance | Good | Better |

## File Structure:
- `double_dqn_exercise.py`: Main `EXERCISE` with Double DQN implementation.
- `config_lunar_1d.py`: Configuration file for LunarLander-v3 with 1D observations.
- `config_lunar_2d.py`: Configuration file for LunarLander-v3 with 2D/image observations.
- `tests/test_double_dqn_train.py`: Comprehensive test suite

## Command
Train the Double DQN agent using the CLI:
```bash
# Train on LunarLander-v3 1D environment
python practice/cli.py --config practice/exercise2_dqn/e22_double_dqn/config_lunar_1d.py

# Train on LunarLander-v3 2D environment
python practice/cli.py --config practice/exercise2_dqn/e22_double_dqn/config_lunar_2d.py
```

Test the trained agent:
```bash
# Play with trained agent and generate video
python practice/cli.py --config practice/exercise2_dqn/e22_double_dqn/config_lunar_1d.py --mode play
```

Push to hub:
```bash
# Generate video and push to hub
python practice/cli.py --config practice/exercise2_dqn/e22_double_dqn/config_lunar_1d.py --mode push_to_hub --username myuser

# Only push to hub
python practice/cli.py --config practice/exercise2_dqn/e22_double_dqn/config_lunar_1d.py --mode push_to_hub --username myuser --skip_play
```
**Replace `myuser` with your HuggingFace account name.**

Run the comprehensive test suite:
```bash
# Run all tests
python -m pytest practice/exercise2_dqn/e22_double_dqn/tests/ -v
```


## Future Enhancements

Potential improvements and extensions:
- **Dueling DQN**: Separate value and advantage streams
- **Prioritized Experience Replay**: Prioritize important transitions
- **Rainbow DQN**: Combine multiple DQN improvements
- **Noisy Networks**: Replace epsilon-greedy with parameter noise
- **Distributional DQN**: Learn full value distribution
- **Multi-step Learning**: N-step bootstrapping
- **Continuous Action Spaces**: DDPG/TD3 for continuous control
