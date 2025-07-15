# Vanilla REINFORCE Implementation

REINFORCE is a policy gradient method that directly optimizes the policy by using the gradient of the expected return.

## Files
- `reinforce_exercise.py`: Main `EXERCISE` containing:
  - `ReinforceConfig`: Configuration dataclass for the algorithm
  - `EpisodeBuffer`: Simple buffer for storing episode data
  - `ReinforceTrainer`: Main trainer class implementing the training loop
  - `_ReinforcePod`: Internal class handling the core REINFORCE logic

- `config_cartpole.py`: Configuration file for CartPole-v1 environment
- `config_mountain_car.py`: Configuration file for MountainCar-v0 environment
- `tests/test_reinforce_train.py`: Comprehensive test suite

`You Need Find A Better Config for MountainCar-v0, it's hard`.

## Command
Train
```bash
# Train the model on CartPole-v1
python practice/cli.py --config practice/exercise3_reinforce/config_cartpole.py

# Train the model on MountainCar-v0 ***hard to get a good result***
python practice/cli.py --config practice/exercise3_reinforce/config_mountain_car.py
```

Play with trained model and generate video
```bash
python practice/cli.py --config practice/exercise3_reinforce/config_cartpole.py --mode play
```

Push to hub
```bash
# generate video and push to hub
python practice/cli.py --config practice/exercise3_reinforce/config_lunar_1d.py --push_to_hub --username myuser

# only push to hub
python practice/cli.py --config practice/exercise3_reinforce/config_lunar_1d.py --push_to_hub --username myuser --skip_play
```
**Replace `myuser` with your HuggingFace account name.**

Run the comprehensive test suite:
```bash
# Run all tests
python -m pytest practice/exercise3_reinforce/tests/ -v
```

## Architecture

Algorithm Features
- **Vanilla REINFORCE**: Pure policy gradient implementation without baseline
- **Entropy Regularization**: Configurable entropy bonus for exploration
- **Gradient Accumulation**: Support for accumulating gradients over multiple episodes

Training Flow
1. **Environment Setup**: Initialize environment and policy network
2. **Episode Collection**: Run episodes and collect trajectories
3. **Return Calculation**: Compute discounted returns for each step
4. **Policy Update**: Compute policy gradient and update parameters
5. **Logging**: Record training metrics and performance

Key Components
- **ReinforceTrainer**: Orchestrates the training process
- **_ReinforcePod**: Handles action sampling and policy updates
- **EpisodeBuffer**: Stores episode data (rewards, log_probs, entropies)
- **Reinforce1DNet**: Policy network (from hands_on implementation)

## Environments

CartPole-v1
- **Observation Space**: 4D (position, velocity, angle, angular velocity)
- **Action Space**: 2 discrete actions (left, right)
- **Goal**: Balance the pole on the cart
- **Configuration**: `config_cartpole.py`

MountainCar-v0
- **Observation Space**: 2D (position, velocity)
- **Action Space**: 3 discrete actions (left, no-op, right)
- **Goal**: Reach the flag on the right side of the mountain
- **Configuration**: `config_mountain_car.py`


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
