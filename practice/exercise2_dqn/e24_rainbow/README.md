# Rainbow DQN Exercise

Rainbow:
- Rainbow DQN combines six key improvements to standard DQN for superior performance
- Integrates Double DQN, Prioritized Experience Replay, Dueling DQN, Multi-step Learning, Distributional DQN, and Noisy Networks

|        Aspect         |   Standard DQN    |         Rainbow DQN           |
|-----------------------|-------------------|-------------------------------|
| Action Selection      | Target Network    | Current Network (Double DQN)  |
| Action Evaluation     | Target Network    | Target Network (Double DQN)   |
| Experience Replay     | Uniform sampling  | Prioritized sampling          |
| Network Architecture  | Single Q-stream   | Dueling (Value + Advantage)   |
| Learning Method       | Single-step       | Multi-step bootstrapping      |
| Value Representation  | Expected value    | Full value distribution       |
| Exploration           | Epsilon-greedy    | Parameter noise (Noisy Networks) |
| Bias Reduction        | ❌ Overestimation bias | ✅ Significantly reduced  |
| Stability             | Moderate          | Highly improved               |
| Performance           | Good              | State-of-the-art              |

## File Structure:
- `per_exercise.py`: `EXERCISE` for the Prioritied Experience Replay.
- `rainbow_exercise`: `EXERCISE` for Rainbow algorithm.
- `config_lunar_1d.py`: Configuration file for LunarLander-v3 with 1D observations.
- `config_lunar_2d.py`: Configuration file for LunarLander-v3 with 2D/image observations.
- `config_car_race.py`: Configuration for the Car Racing.

## Command
Train the Rainbow DQN agent using the CLI:
```bash
# Train on LunarLander-v3 1D environment
python practice/cli.py --config practice/exercise2_dqn/e24_rainbow/config_lunar_1d.py

# Train on LunarLander-v3 2D environment
python practice/cli.py --config practice/exercise2_dqn/e24_rainbow/config_lunar_2d.py

# Train on the Car Racing environment
python practice/cli.py --config practice/exercise2_dqn/e24_rainbow/config_car_race.py
```

Test the trained agent:
```bash
# Play with trained agent and generate video
python practice/cli.py --config practice/exercise2_dqn/e24_rainbow/config_lunar_1d.py --mode play
```

Push to hub:
```bash
# Generate video and push to hub
python practice/cli.py --config practice/exercise2_dqn/e24_rainbow/config_lunar_1d.py --mode push_to_hub --username myuser

# Only push to hub
python practice/cli.py --config practice/exercise2_dqn/e24_rainbow/config_lunar_1d.py --mode push_to_hub --username myuser --skip_play
```
**Replace `myuser` with your HuggingFace account name.**

Run the comprehensive test suite:
```bash
# Run all tests
python -m pytest practice/exercise2_dqn/e24_rainbow/tests/ -v
```


## Future Enhancements

Potential improvements and extensions beyond Rainbow DQN:
- **Continuous Action Spaces**: DDPG/TD3/SAC for continuous control
- **Hierarchical RL**: Option-critic and meta-learning approaches
- **Model-based RL**: World models and planning
- **Multi-agent RL**: MADDPG, QMIX for cooperative/competitive scenarios
- **Offline RL**: Conservative Q-learning and behavior cloning
- **Meta-RL**: Learning to learn across multiple environments
- **Transformer-based**: Attention mechanisms for complex state representations
