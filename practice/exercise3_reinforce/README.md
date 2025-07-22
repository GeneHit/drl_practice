# Vanilla REINFORCE Exercise

REINFORCE is a policy gradient method that directly optimizes the policy by using the gradient of the expected return.

## Files
- `reinforce_exercise.py`: Main `EXERCISE` containing:
- `config_cartpole.py`: Configuration file for CartPole-v1 environment
- `config_mountain_car.py`: Configuration file for MountainCar-v0 environment
- `tests/test_reinforce_train.py`: Comprehensive test suite

`You Need Find A Better Config for MountainCar-v0, it's hard`.

## Command
Train  (change `cartpole` to `lunar_1d` if needed)
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
python practice/cli.py --config practice/exercise3_reinforce/config_cartpole.py --mode push_to_hub --username myuser

# only push to hub
python practice/cli.py --config practice/exercise3_reinforce/config_cartpole.py --mode push_to_hub --username myuser --skip_play
```
**Replace `myuser` with your HuggingFace account name.**

Run the comprehensive test suite:
```bash
# Run all tests
python -m pytest practice/exercise3_reinforce/tests/ -v
```

## Architecture

Algorithm Features:
- Vanilla REINFORCE, Entropy Regularization, Gradient Accumulation

Training Flow
1. **Environment Setup**: Initialize environment and policy network
2. **Episode Collection**: Run episodes and collect trajectories
3. **Return Calculation**: Compute discounted returns for each step
4. **Policy Update**: Compute policy gradient and update parameters
5. **Logging**: Record training metrics and performance


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


## Future Enhancements

Potential improvements:
- Actor-Critic variants (with baseline)
- Natural policy gradients
- Proximal Policy Optimization (PPO)
- Multi-environment training
- Distributed training support
